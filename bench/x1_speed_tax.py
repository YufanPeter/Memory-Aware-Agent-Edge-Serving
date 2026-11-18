"""X1: does KV that nobody is using slow down the agents that are decoding? ("speed tax")

llama.cpp b11960 with --kv-unified runs attention over every cell up to the highest occupied
one (n_kv, padded to 256), and for a head-dim-128 model the CUDA kernel does not skip masked
cells during decode (docs/phase2_notes.md). If that costs real time, KV held by idle or finished
agents taxes every decoding agent, and *where* that KV sits in the pool decides whether freeing
it helps.

For each condition we start a fresh server (clean cell layout), place B "active" slots and N
"idle" slots of L tokens each in a given order, optionally erase or relocate some of them, then
let the B active slots decode together and record decode speed. With --kv-debug the server logs
n_kv for every ubatch, so each condition also reports the span attention actually covered.

Conditions:
  none          B active slots only (reference)
  idle_above    active placed first, then N idle slots above them
  idle_below    N idle slots placed first, then the active slots above them
  freed_above   idle_above, then the idle slots are erased (top of the pool freed)
  holes_below   idle_below, then the idle slots are erased (holes below the active slots)
  relocate      holes_below, then each active slot is saved, erased and restored
                (does restore place it in the low free cells?)

    LLAMA_SERVER=~/llama.cpp/build/bin/llama-server \
      uv run python bench/x1_speed_tax.py [--b 4,8] [--n 0,8,16,32] [--only none,idle_above] [--kv-debug]

Results: results/x1/<tag>/{result.json, server.log, server_cmd.txt}; summary in results/x1/summary.csv.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

PORT = 8093
URL = f"http://127.0.0.1:{PORT}"
CONDITIONS = ["none", "idle_above", "idle_below", "freed_above", "holes_below", "relocate"]
FIND_SLOT = re.compile(r"find_slot: stream\[\d+\], n =\s*(\d+), used =\s*(\d+)")


def ids(n: int, seed: int) -> list[int]:
    """Deterministic token ids well inside Qwen2.5's vocab; different seeds share no prefix."""
    return [1000 + (seed * 7919 + i * 104729) % 90000 for i in range(n)]


class Server:
    """A fresh llama-server per condition, started through scripts/start_server.sh."""

    def __init__(self, out: Path, ctx: int, parallel: int, extra: str, kv_debug: bool) -> None:
        self.out, self.ctx, self.parallel, self.extra, self.kv_debug = out, ctx, parallel, extra, kv_debug

    def __enter__(self) -> "Server":
        self.out.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "CTX": str(self.ctx), "PARALLEL": str(self.parallel), "PORT": str(PORT),
               "EXTRA": self.extra + (" -lv 5" if self.kv_debug else ""),   # 5 = debug
               "CMD_OUT": str(self.out / "server_cmd.txt")}
        if self.kv_debug:
            env["LLAMA_KV_CACHE_DEBUG"] = "1"
        self.log_path = self.out / "server.log"
        self.log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(["bash", "scripts/start_server.sh"], env=env, stdout=self.log,
                                     stderr=subprocess.STDOUT)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 300:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited, see {self.log_path}")
            try:
                if httpx.get(f"{URL}/health", timeout=2).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(1)
        raise RuntimeError("server not healthy after 300 s")

    def log_offset(self) -> int:
        self.log.flush()
        return self.log_path.stat().st_size

    def n_kv_since(self, offset: int) -> list[tuple[int, int]]:
        """(n_kv, used cells) for every ubatch logged after `offset`."""
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            return [(int(m.group(1)), int(m.group(2))) for m in FIND_SLOT.finditer(f.read())]

    def __exit__(self, *exc: Any) -> None:
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


async def complete(c: httpx.AsyncClient, slot: int, prompt: list[int], n_predict: int) -> dict[str, Any]:
    body = {"prompt": prompt, "n_predict": n_predict, "id_slot": slot, "cache_prompt": True,
            "temperature": 0.0, "ignore_eos": True}
    r = await c.post("/completion", json=body)
    if r.status_code != 200:
        raise RuntimeError(f"slot {slot}: HTTP {r.status_code} {r.text[:300]}")
    t = r.json().get("timings", {})
    return {"slot": slot, **{k: t.get(k) for k in ("prompt_n", "cache_n", "predicted_n", "predicted_ms")}}


async def slot_op(c: httpx.AsyncClient, slot: int, action: str, filename: str | None = None) -> dict[str, Any]:
    t0 = time.monotonic()
    r = await c.post(f"/slots/{slot}", params={"action": action}, json={"filename": filename} if filename else {})
    if r.status_code != 200:
        raise RuntimeError(f"slot {slot} {action}: HTTP {r.status_code} {r.text[:300]}")
    return {"op": action, "slot": slot, "wall_ms": round((time.monotonic() - t0) * 1000, 1), "resp": r.json()}


async def place(c: httpx.AsyncClient, slots: list[int], length: int) -> None:
    """Prefill each slot one after another, so cells are placed in this order."""
    for s in slots:
        await complete(c, s, ids(length, seed=s), n_predict=1)


async def run_condition(srv: Server, cond: str, b: int, n: int, length: int, decode: int,
                        reps: int) -> dict[str, Any]:
    active = list(range(b))
    idle = list(range(b, b + n))
    ops: list[dict[str, Any]] = []
    async with httpx.AsyncClient(base_url=URL, timeout=httpx.Timeout(600, connect=10)) as c:
        if cond in ("none",):
            await place(c, active, length)
        elif cond in ("idle_above", "freed_above"):
            await place(c, active, length)
            await place(c, idle, length)
        else:   # idle_below, holes_below, relocate
            await place(c, idle, length)
            await place(c, active, length)
        if cond in ("freed_above", "holes_below", "relocate"):
            for s in idle:
                ops.append(await slot_op(c, s, "erase"))
        if cond == "relocate":
            for s in active:
                ops.append(await slot_op(c, s, "save", f"x1_slot{s}.bin"))
                ops.append(await slot_op(c, s, "erase"))
                ops.append(await slot_op(c, s, "restore", f"x1_slot{s}.bin"))

        rounds = []
        for rep in range(reps):
            off = srv.log_offset()
            t0 = time.monotonic()
            res = await asyncio.gather(*(complete(c, s, ids(length, seed=s), decode) for s in active))
            wall = time.monotonic() - t0
            spans = srv.n_kv_since(off)
            ms_per_tok = [r["predicted_ms"] / r["predicted_n"] for r in res if r["predicted_n"]]
            rounds.append({
                "rep": rep, "wall_s": round(wall, 3),
                "decode_tok_s_total": round(sum(r["predicted_n"] for r in res) / wall, 1),
                "ms_per_token_mean": round(statistics.mean(ms_per_tok), 2),
                "cache_n_min": min(r["cache_n"] for r in res),
                "n_kv_median": statistics.median(s[0] for s in spans) if spans else None,
                "n_kv_max": max(s[0] for s in spans) if spans else None,
                "used_cells_median": statistics.median(s[1] for s in spans) if spans else None,
            })
    return {"cond": cond, "b": b, "n": n, "length": length, "decode": decode, "ops": ops, "rounds": rounds}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b", default="4,8", help="active slots decoding together")
    ap.add_argument("--n", default="0,8,16,32", help="idle slots")
    ap.add_argument("--length", type=int, default=2048, help="tokens per slot")
    ap.add_argument("--decode", type=int, default=128, help="tokens each active slot decodes per round")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--parallel", type=int, default=64)
    ap.add_argument("--only", default=",".join(CONDITIONS))
    ap.add_argument("--kv-debug", action="store_true", help="log n_kv per ubatch (LLAMA_KV_CACHE_DEBUG=1)")
    ap.add_argument("--extra", default="", help="extra llama-server flags, e.g. --no-kv-unified")
    ap.add_argument("--tag", default="", help="suffix for the result directories")
    ap.add_argument("--out", default="results/x1")
    args = ap.parse_args()

    out = Path(args.out)
    rows = []
    for b in map(int, args.b.split(",")):
        for n in map(int, args.n.split(",")):
            for cond in args.only.split(","):
                if (cond == "none") != (n == 0):
                    continue   # "none" is the n=0 reference; every other condition needs idle slots
                if b + n > args.parallel:
                    continue
                tag = f"{cond}_b{b}_n{n}{args.tag}"
                d = out / tag
                if (d / "result.json").exists():
                    print(f"skip {tag}")
                    continue
                with Server(d, args.ctx, args.parallel, args.extra, args.kv_debug) as srv:
                    r = asyncio.run(run_condition(srv, cond, b, n, args.length, args.decode, args.reps))
                (d / "result.json").write_text(json.dumps(r, indent=1), encoding="utf-8", newline="\n")
                last = r["rounds"][1:] or r["rounds"]   # the first round warms up the graph
                row = {"tag": tag, "cond": cond, "b": b, "n": n,
                       "ms_per_token": round(statistics.mean(x["ms_per_token_mean"] for x in last), 2),
                       "decode_tok_s_total": round(statistics.mean(x["decode_tok_s_total"] for x in last), 1),
                       "n_kv_median": last[-1]["n_kv_median"], "used_cells_median": last[-1]["used_cells_median"]}
                rows.append(row)
                print(json.dumps(row))
    if rows:
        summary = out / "summary.csv"
        new = not summary.exists()
        with open(summary, "a", encoding="utf-8", newline="\n") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
