"""Cost model calibrated with the A10G microbenchmarks (M0) and runtime probes (E0a).

cost_model.py holds the predictions frozen before any measurement (spec-sheet numbers).
This module refits the same quantities from results/microbench/ and prints what changed,
plus the numbers Phase 2 depends on: KV capacity in agents, offload vs re-prefill cost,
and the TTL break-even T* relative to tool delays.

    uv run python -m costmodel.calibrated [--microbench results/microbench]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

from costmodel.cost_model import QWEN_7B, TOOL_MEDIAN_S, kv_bytes_per_token

MiB = 2**20
GPU_MIB = 23028          # nvidia-smi memory.total on mem-aws
SERVER_FIXED_MIB = 6433 - 32768 * 56 // 1024   # E0a: 6,433 MiB used at -c 32768, minus that KV
SHARED_PREFIX = 1400     # system prompt + 6 few-shot examples, tokens; E0a: stored once per slot
PROFILES = {             # measured on 7B (CLAUDE.md, profiles_check): (p50, max) context tokens
    "react_short": (1900, 3000),
    "react_long": (3400, 5300),
}


def fit_slot_ops(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    big = [r for r in rows if int(r["n_tokens"]) >= 2048]
    save_bw = statistics.median(float(r["file_bytes"]) / (float(r["save_ms"]) / 1e3) for r in big)
    restore_bw = statistics.median(float(r["file_bytes"]) / (float(r["restore_ms"]) / 1e3) for r in big)
    # Re-prefill time t(S) = a S + b S^2 (attention makes long prompts slightly slower per token).
    by_n: dict[int, list[float]] = {}
    for r in rows:
        by_n.setdefault(int(r["n_tokens"]), []).append(float(r["reprefill_ms"]) / 1e3)
    pts = sorted((n, statistics.median(v)) for n, v in by_n.items())
    (n1, t1), (n2, t2) = pts[-3], pts[-1]
    b = (t2 / n2 - t1 / n1) / (n2 - n1)
    a = t1 / n1 - b * n1
    return {"save_Bps": save_bw, "restore_Bps": restore_bw, "prefill_a": a, "prefill_b": b,
            "bytes_per_token": float(rows[-1]["file_bytes"]) / int(rows[-1]["n_tokens"])}


def fit_decode(path: Path) -> dict[int, float]:
    """Total decode tok/s by batch size from llama-batched-bench (2,048-token prompts)."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) == 10 and cols[0].isdigit():
            out[int(cols[2])] = float(cols[7])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--microbench", default="results/microbench")
    args = ap.parse_args()
    mb = Path(args.microbench)
    ops = fit_slot_ops(sorted(mb.glob("slot_offload_a10g*.csv"))[-1])
    run = sorted(p for p in mb.iterdir() if p.is_dir() and p.name.startswith("a10g"))[-1]
    decode = fit_decode(run / "batched_f16.txt")
    prefill = {r["n_prompt"]: r["avg_ts"] for r in json.loads((run / "prefill_f16.json").read_text())}

    per_tok = kv_bytes_per_token(QWEN_7B, "f16")
    budget_tokens = int((GPU_MIB - SERVER_FIXED_MIB - 512) * MiB / per_tok)   # 512 MiB safety margin
    t_pre = lambda s: ops["prefill_a"] * s + ops["prefill_b"] * s * s  # noqa: E731
    t_off = lambda s: s * ops["bytes_per_token"] * (1 / ops["save_Bps"] + 1 / ops["restore_Bps"])  # noqa: E731

    out = {"fits": ops, "decode_tok_s_by_batch": decode, "prefill_tok_s_by_len": prefill,
           "kv_budget_tokens_max": budget_tokens, "profiles": {}}
    print(f"KV pool if all free VRAM is used: {budget_tokens:,} tokens "
          f"({budget_tokens * per_tok / 2**30:.1f} GiB)")
    print(f"slot save {ops['save_Bps'] / 1e9:.2f} GB/s, restore {ops['restore_Bps'] / 1e9:.2f} GB/s "
          f"(both block every slot while they run, E0a stall probe)")
    print(f"re-prefill t(S) = {ops['prefill_a'] * 1e3:.4f} ms·S + {ops['prefill_b'] * 1e9:.3f} ns·S²")
    print(f"decode tok/s by batch (2k context): {decode}")
    print()
    print("| profile | peak ctx | KV/agent | max agents (prefix per slot) | prefix dup. at K=16 | "
          "offload RT | re-prefill | T* at K=16 | r = T*/median delay |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, (p50, peak) in PROFILES.items():
        s = p50   # an agent entering a tool call holds roughly its median context
        s_active = 15 * p50
        b_off = (s + s_active) * t_off(s)              # its own offload + everyone stalled meanwhile
        b_drop = (s + s_active) * t_pre(s)             # re-prefill slows the batch it joins
        t_star = min(b_off, b_drop) / s
        row = {"p50": p50, "peak": peak, "kv_mb_peak": peak * per_tok / 1e6,
               "max_agents": budget_tokens // peak, "prefix_dup_tokens_k16": 15 * SHARED_PREFIX,
               "offload_rt_s": t_off(s), "reprefill_s": t_pre(s), "t_star_s": t_star,
               "r": t_star / TOOL_MEDIAN_S, "cheaper": "offload" if b_off < b_drop else "drop"}
        out["profiles"][name] = row
        print(f"| {name} | {peak:,} | {row['kv_mb_peak']:.0f} MB | {row['max_agents']} | "
              f"{row['prefix_dup_tokens_k16']:,} tok | {row['offload_rt_s'] * 1e3:.0f} ms | "
              f"{row['reprefill_s'] * 1e3:.0f} ms | {t_star:.2f} s ({row['cheaper']}) | {row['r']:.2f} |")
    Path("predictions").mkdir(exist_ok=True)
    Path("predictions/calibrated_a10g.json").write_text(json.dumps(out, indent=1), encoding="utf-8",
                                                         newline="\n")


if __name__ == "__main__":
    main()
