"""E0a: what llama-server actually does with pinned slots and the unified KV pool.

Every probe starts its own server (scripts/start_server.sh), talks to it over HTTP only,
and saves the raw responses, /slots snapshots and the server log under results/e0/, so
each finding in the report has evidence attached.

    LLAMA_SERVER=~/llama.cpp/build/bin/llama-server MODEL=~/models/<7b>.gguf \
      python bench/e0_runtime.py [--only pressure,overflow,prefix,ctx,np,stall,http]

Probes:
  pressure  idle pinned slots whose KV exceeds -c: error, queueing, or eviction of idle slots?
  overflow  concurrent *active* requests whose KV exceeds -c
  prefix    is a shared prompt prefix stored once or once per slot?
  ctx       per-slot context limit under --kv-unified (and --kv-unified-per-slot)
  np        cost of -np 64 vs 4 at the same -c (VRAM, decode speed)
  stall     do slot save/restore block other slots' decoding? restore into another slot?
  http      with 64 idle keep-alive connections, do /health and slot ops still answer quickly?
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

OUT = Path("results/e0")
PORT = 8090
URL = f"http://127.0.0.1:{PORT}"


def ids(n: int, seed: int) -> list[int]:
    """Deterministic token ids well inside Qwen2.5's vocab; different seeds share no prefix."""
    return [1000 + (seed * 7919 + i * 104729) % 90000 for i in range(n)]


def gpu_mem_mib() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


class Server:
    def __init__(self, tag: str, ctx: int, parallel: int, extra: str = "") -> None:
        self.tag, self.ctx, self.parallel, self.extra = tag, ctx, parallel, extra

    def __enter__(self) -> "Server":
        OUT.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "CTX": str(self.ctx), "PARALLEL": str(self.parallel), "PORT": str(PORT),
               "EXTRA": self.extra, "CMD_OUT": str(OUT / f"{self.tag}_server_cmd.txt")}
        self.log = open(OUT / f"{self.tag}_server.log", "w", encoding="utf-8")
        self.proc = subprocess.Popen(["bash", "scripts/start_server.sh"], env=env, stdout=self.log,
                                     stderr=subprocess.STDOUT)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 300:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited, see {OUT / (self.tag + '_server.log')}")
            try:
                if httpx.get(f"{URL}/health", timeout=2).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(1)
        raise RuntimeError("server not healthy after 300 s")

    def __exit__(self, *exc: Any) -> None:
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


async def complete(c: httpx.AsyncClient, slot: int, prompt: list[int], n_predict: int = 8,
                   ignore_eos: bool = False) -> dict[str, Any]:
    body = {"prompt": prompt, "n_predict": n_predict, "id_slot": slot, "cache_prompt": True,
            "temperature": 0.0, "ignore_eos": ignore_eos}
    t0 = time.monotonic()
    r = await c.post("/completion", json=body)
    out: dict[str, Any] = {"slot": slot, "prompt_tokens": len(prompt), "status": r.status_code,
                           "wall_s": round(time.monotonic() - t0, 3)}
    try:
        d = r.json()
    except ValueError:
        return out | {"error": r.text[:300]}
    if r.status_code != 200:
        return out | {"error": json.dumps(d)[:300]}
    t = d.get("timings", {})
    return out | {k: t.get(k) for k in ("prompt_n", "cache_n", "predicted_n", "prompt_ms", "predicted_ms")}


async def stream(c: httpx.AsyncClient, slot: int, prompt: list[int], n_predict: int) -> list[float]:
    """Token arrival times (monotonic) of one streamed completion."""
    body = {"prompt": prompt, "n_predict": n_predict, "id_slot": slot, "cache_prompt": True,
            "temperature": 0.0, "ignore_eos": True, "stream": True}
    times: list[float] = []
    async with c.stream("POST", "/completion", json=body) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and json.loads(line[6:]).get("content"):
                times.append(time.monotonic())
    return times


async def slot_op(c: httpx.AsyncClient, slot: int, action: str, filename: str | None = None) -> dict[str, Any]:
    t0 = time.monotonic()
    r = await c.post(f"/slots/{slot}", params={"action": action}, json={"filename": filename} if filename else {})
    t1 = time.monotonic()
    try:
        d = r.json()
    except ValueError:
        d = {"text": r.text[:300]}
    return {"op": action, "slot": slot, "status": r.status_code, "t0": t0, "t1": t1,
            "wall_ms": round((t1 - t0) * 1000, 1), "resp": d}


async def slots(c: httpx.AsyncClient) -> Any:
    r = await c.get("/slots")
    return r.json() if r.status_code == 200 else {"status": r.status_code, "text": r.text[:300]}


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=URL, timeout=httpx.Timeout(600, connect=10))


def save(name: str, data: Any) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8", newline="\n")


# --- probes ---------------------------------------------------------------------------

async def probe_pressure() -> str:
    """4 pinned slots × 2,500 tokens in an 8,192-token pool, sent one after another."""
    with Server("pressure", 8192, 8):
        async with client() as c:
            first = [await complete(c, i, ids(2500, i)) for i in range(4)]
            snap = await slots(c)
            revisit = [await complete(c, i, ids(2500, i), 1) for i in range(4)]
    save("pressure", {"first": first, "slots_after_first": snap, "revisit": revisit})
    kept = [r.get("cache_n") for r in revisit]
    errs = [r["status"] for r in first if r["status"] != 200]
    return f"errors on first pass: {errs or 'none'}; cache_n on revisit (2500 = kept): {kept}"


async def probe_overflow() -> str:
    """4 concurrent requests × (2,500 prompt + 400 generated) in an 8,192-token pool."""
    with Server("overflow", 8192, 8):
        async with client() as c:
            res = await asyncio.gather(*(complete(c, i, ids(2500, 10 + i), 400, True) for i in range(4)))
    save("overflow", res)
    return "; ".join(f"slot {r['slot']}: HTTP {r['status']} predicted {r.get('predicted_n')} "
                     f"{(r.get('error') or '')[:80]}" for r in res)


async def probe_prefix() -> str:
    """8 slots, each prompt = the same 1,500-token prefix + 300 own tokens."""
    shared = ids(1500, 99)
    lines = []
    for ctx in (32768, 8192):   # no pressure, then pressure (unshared 8 × 1,800 > 8,192)
        with Server(f"prefix_c{ctx}", ctx, 8):
            async with client() as c:
                first = [await complete(c, i, shared + ids(300, 200 + i)) for i in range(8)]
                revisit = [await complete(c, i, shared + ids(300, 200 + i), 1) for i in range(8)]
                snap = await slots(c)
        save(f"prefix_c{ctx}", {"first": first, "revisit": revisit, "slots": snap})
        lines.append(f"-c {ctx}: first-request cache_n {[r.get('cache_n') for r in first]}, "
                     f"revisit cache_n {[r.get('cache_n') for r in revisit]}")
    return " | ".join(lines)


async def probe_ctx() -> str:
    """Can one slot use more than -c / -np under --kv-unified? And with --kv-unified-per-slot?"""
    out = {}
    with Server("ctx", 16384, 8):
        async with client() as c:
            props = (await c.get("/props")).json()
            big = await complete(c, 0, ids(12000, 5), 4)
            out["unified"] = {"props_n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
                              "slots": await slots(c), "prompt_12k": big}
    with Server("ctx_perslot", 16384, 8, "--kv-unified-per-slot 4096"):
        async with client() as c:
            out["per_slot_4096"] = {"prompt_6k": await complete(c, 0, ids(6000, 6), 4),
                                    "prompt_3k": await complete(c, 1, ids(3000, 7), 4)}
    save("ctx", out)
    return (f"props n_ctx={out['unified']['props_n_ctx']}; 12k prompt on one slot of -c 16384 -np 8: "
            f"HTTP {out['unified']['prompt_12k']['status']}; per-slot 4096: 6k → "
            f"HTTP {out['per_slot_4096']['prompt_6k']['status']}, 3k → HTTP {out['per_slot_4096']['prompt_3k']['status']}")


async def probe_np() -> str:
    """Same -c, -np 4 vs 64: VRAM after load and decode speed of 4 concurrent requests."""
    res = {}
    for np_ in (4, 64):
        with Server(f"np{np_}", 32768, np_):
            mem = gpu_mem_mib()
            async with client() as c:
                runs = []
                for rep in range(3):
                    t0 = time.monotonic()
                    rs = await asyncio.gather(*(complete(c, i, ids(1024, 40 + 10 * rep + i), 128, True)
                                                for i in range(4)))
                    runs.append(sum(r.get("predicted_n") or 0 for r in rs) / (time.monotonic() - t0))
            res[np_] = {"gpu_mem_mib": mem, "tok_per_s": runs}
    save("np", res)
    a, b = statistics.median(res[4]["tok_per_s"]), statistics.median(res[64]["tok_per_s"])
    return (f"-np 4: {res[4]['gpu_mem_mib']} MiB, {a:.0f} tok/s; -np 64: {res[64]['gpu_mem_mib']} MiB, "
            f"{b:.0f} tok/s ({(b / a - 1) * 100:+.1f}%)")


async def probe_stall() -> str:
    """8 slots decode while slot 15 (3,000 tokens) is saved, erased and restored; then the
    image is restored into slot 14 to test cross-slot restore."""
    big = ids(3000, 77)
    with Server("stall", 32768, 16):
        async with client() as c, client() as ctl:
            await complete(c, 15, big, 1)
            prompts = [ids(200, 500 + i) for i in range(8)]
            base = await asyncio.gather(*(stream(c, i, prompts[i], 300) for i in range(8)))

            async def ops() -> list[dict[str, Any]]:
                await asyncio.sleep(1.0)
                log = [await slot_op(ctl, 15, "save", "stall.bin")]
                log.append(await slot_op(ctl, 15, "erase"))
                await asyncio.sleep(0.5)
                log.append(await slot_op(ctl, 15, "restore", "stall.bin"))
                return log

            prompts2 = [ids(200, 600 + i) for i in range(8)]
            *busy, oplog = await asyncio.gather(*(stream(c, i, prompts2[i], 300) for i in range(8)), ops())
            after_restore = await complete(c, 15, big, 1)
            cross = await slot_op(ctl, 14, "restore", "stall.bin")
            cross_hit = await complete(c, 14, big, 1)

    def gaps(times: list[float]) -> list[float]:
        return [b - a for a, b in zip(times, times[1:])]

    base_gaps = [g for t in base for g in gaps(t)]
    win = [(o["t0"], o["t1"]) for o in oplog if o["op"] in ("save", "restore")]
    in_win = [b - a for t in busy for a, b in zip(t, t[1:]) if any(a < w1 and b > w0 for w0, w1 in win)]
    res = {"baseline_gap_ms_median": statistics.median(base_gaps) * 1000,
           "baseline_gap_ms_p99": sorted(base_gaps)[int(0.99 * len(base_gaps))] * 1000,
           "gaps_overlapping_ops_ms": [round(g * 1000, 1) for g in in_win],
           "ops": [{k: o[k] for k in ("op", "status", "wall_ms", "resp")} for o in oplog],
           "same_slot_after_restore": after_restore, "cross_slot_restore": cross, "cross_slot_hit": cross_hit}
    save("stall", res)
    worst = max(in_win) * 1000 if in_win else None
    return (f"decode gap median {res['baseline_gap_ms_median']:.1f} ms (p99 {res['baseline_gap_ms_p99']:.1f}); "
            f"worst gap during save/restore {worst and round(worst, 1)} ms; ops "
            f"{[(o['op'], o['wall_ms']) for o in res['ops']]}; same-slot cache_n {after_restore.get('cache_n')}; "
            f"cross-slot restore HTTP {cross['status']}, cache_n {cross_hit.get('cache_n')}")


async def probe_http() -> str:
    """64 agents' keep-alive connections left idle (as when all are inside tool calls)."""
    with Server("http", 65536, 64):
        idle = httpx.AsyncClient(base_url=URL, timeout=600,
                                 limits=httpx.Limits(max_connections=128, max_keepalive_connections=128))
        await asyncio.gather(*(complete(idle, i, ids(50, 300 + i), 1) for i in range(64)))
        async with httpx.AsyncClient(base_url=URL, timeout=60,
                                     limits=httpx.Limits(max_keepalive_connections=0)) as ctl:
            health = []
            for _ in range(5):
                t0 = time.monotonic()
                await ctl.get("/health")
                health.append(round((time.monotonic() - t0) * 1000, 1))
            save_ops = [await slot_op(ctl, 0, "save", f"http{i}.bin") for i in range(3)]
        await idle.aclose()
    res = {"health_ms": health, "save_wall_ms": [o["wall_ms"] for o in save_ops],
           "save_status": [o["status"] for o in save_ops]}
    save("http", res)
    return f"/health {health} ms; slot save {res['save_wall_ms']} ms (HTTP {res['save_status']})"


PROBES = {"pressure": probe_pressure, "overflow": probe_overflow, "prefix": probe_prefix, "ctx": probe_ctx,
          "np": probe_np, "stall": probe_stall, "http": probe_http}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help=f"comma-separated subset of {','.join(PROBES)}")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(PROBES)
    for name in names:
        try:
            print(f"[{name}] {asyncio.run(PROBES[name]())}", flush=True)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
