"""Measure llama-server's real KV offload path (slot save/restore) against
re-prefill, across context lengths. This decides whether retain/offload/drop
is a real trade-off or whether always-offload trivially wins.

Start a server with one slot first:
    llama-server -m MODEL -c 16384 -np 1 -fa on --slot-save-path DIR
DIR controls what "offload" means:
    AWS:   /dev/shm/slots     host RAM (closest to a PCIe offload)
           instance-store NVMe  local SSD tier
    macOS: any local dir      SSD; unified memory has no separate host tier
Then:
    python bench/slot_offload_bench.py --tag a10g-shm
"""

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path


def post(url: str, body: dict) -> tuple[dict, float]:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        data = json.load(r)
    return data, (time.perf_counter() - t0) * 1000


def prompt_tokens(n: int) -> list[int]:
    # Deterministic ids well inside Qwen2.5's 151k vocab; content is irrelevant to timing.
    return [1000 + (i * 7919) % 90000 for i in range(n)]


def run_one(base: str, n_tokens: int) -> dict:
    slot = f"{base}/slots/0"
    fname = f"bench_{n_tokens}.bin"
    prompt = prompt_tokens(n_tokens)
    completion = {"prompt": prompt, "n_predict": 1, "id_slot": 0, "cache_prompt": True}

    post(f"{slot}?action=erase", {})
    cold, _ = post(f"{base}/completion", completion)
    saved, save_wall = post(f"{slot}?action=save", {"filename": fname})
    post(f"{slot}?action=erase", {})
    restored, restore_wall = post(f"{slot}?action=restore", {"filename": fname})
    warm, _ = post(f"{base}/completion", completion)

    if warm["timings"]["prompt_n"] > max(4, 0.05 * n_tokens):
        print(f"  warning: restore at {n_tokens} tokens did not reuse the cache "
              f"(prompt_n={warm['timings']['prompt_n']})")
    return {
        "n_tokens": n_tokens,
        "reprefill_ms": cold["timings"]["prompt_ms"],
        "save_ms": saved["timings"]["save_ms"],
        "save_wall_ms": save_wall,
        "restore_ms": restored["timings"]["restore_ms"],
        "restore_wall_ms": restore_wall,
        "file_bytes": saved["n_written"],
        "resume_prompt_ms": warm["timings"]["prompt_ms"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--lengths", default="256,512,1024,2048,4096,8000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default="local")
    args = ap.parse_args()

    rows = []
    for n in map(int, args.lengths.split(",")):
        for rep in range(args.reps):
            row = run_one(args.url, n) | {"rep": rep}
            rows.append(row)
            print(f"L={n:>5} rep={rep}  re-prefill {row['reprefill_ms']:8.1f} ms | "
                  f"save {row['save_ms']:7.1f} ms  restore {row['restore_ms']:7.1f} ms | "
                  f"{row['file_bytes'] / 1e6:7.1f} MB")

    out = Path("results/microbench") / f"slot_offload_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
