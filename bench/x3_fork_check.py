"""X3: does the patched fork action really store a shared prefix once?

Capacity test on a pool too small for per-slot copies. A "golden" slot holds a P-token prefix;
each of N slots then holds that prefix plus S tokens of its own:
  - copy mode: every slot prefills the prefix itself (what llama-server does today),
    so N*(P+S) tokens are needed and, above -c, the server has to purge idle slots;
  - fork mode: every slot first forks the prefix from the golden slot, so P + N*S are needed.
Reported per mode: the first request's cache_n per slot (should equal P under fork), server
purges ("purging slot" in the log), and prefill tokens.

    LLAMA_SERVER=~/llama.cpp/build/bin/llama-server MODEL=models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
      uv run python bench/x3_fork_check.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from x1_speed_tax import URL, Server, ids  # noqa: E402


async def run(mode: str, n: int, prefix: int, own: int) -> dict:
    golden = n
    prefix_ids = ids(prefix, seed=999)
    out = {"mode": mode, "slots": []}
    async with httpx.AsyncClient(base_url=URL, timeout=httpx.Timeout(600, connect=10)) as c:
        if mode == "fork":
            r = await c.post("/completion", json={"prompt": prefix_ids, "n_predict": 0, "id_slot": golden,
                                                  "cache_prompt": True})
            r.raise_for_status()
        for s in range(n):
            if mode == "fork":
                r = await c.post(f"/slots/{s}", params={"action": "fork"}, json={"src": golden, "n_tokens": prefix})
                if r.status_code != 200:
                    raise RuntimeError(f"fork failed: {r.status_code} {r.text[:300]}")
                fork = r.json()
            else:
                fork = None
            body = {"prompt": prefix_ids + ids(own, seed=s), "n_predict": 24, "id_slot": s, "cache_prompt": True,
                    "temperature": 0.0}
            r = await c.post("/completion", json=body)
            r.raise_for_status()
            d = r.json()
            t = d["timings"]
            out["slots"].append({"slot": s, "cache_n": t["cache_n"], "prompt_n": t["prompt_n"], "fork": fork,
                                 "text": d["content"]})
        # Touch every slot again: if the server purged any, its prefix is gone and gets re-prefilled.
        again = []
        for s in range(n):
            r = await c.post("/completion", json={"prompt": prefix_ids + ids(own, seed=s), "n_predict": 1,
                                                  "id_slot": s, "cache_prompt": True})
            again.append(r.json()["timings"]["prompt_n"] if r.status_code == 200 else f"HTTP {r.status_code}")
        out["second_pass_prompt_n"] = again
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--prefix", type=int, default=1400)
    ap.add_argument("--own", type=int, default=1000)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--out", default="results/x3")
    args = ap.parse_args()
    texts: dict[str, list[str]] = {}
    for mode in ("copy", "fork"):
        d = Path(args.out) / mode
        with Server(d, args.ctx, args.n + 1, "", False) as srv:
            r = asyncio.run(run(mode, args.n, args.prefix, args.own))
            srv.log.flush()
            r["server_purges"] = srv.log_path.read_text(encoding="utf-8", errors="replace").count("purging slot")
        (d / "result.json").write_text(json.dumps(r, indent=1), encoding="utf-8", newline="\n")
        texts[mode] = [x["text"] for x in r["slots"]]
        print(mode, "first cache_n:", [x["cache_n"] for x in r["slots"]], "| purges:", r["server_purges"],
              "| 2nd-pass prompt_n:", r["second_pass_prompt_n"])

    same = sum(a == b for a, b in zip(texts["copy"], texts["fork"]))
    print(f"identical generations (copy vs fork, single requests at temperature 0): {same}/{len(texts['fork'])}")


if __name__ == "__main__":
    main()
