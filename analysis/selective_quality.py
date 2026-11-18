"""Selective KV, step 2: per configuration, what trimming saves (KV, prefill) and costs (F1).

Reads results/<sweep>/<point>/rep*/ (tasks.jsonl, summary.json) and prints one row per point:
  quality   F1 per repeat, paired F1 change vs `none` (per-question mean over repeats) with a
            bootstrap 95% CI, supporting-fact sentences deleted
  KV        peak context per task (mean, p95: what admission reserves), KV token-seconds per task
  recompute prefill tokens per task
  speed     goodput, task latency p95, TTFT p50/p95, TPOT p50 (ample pool: context only)
  trims     tokens deleted per task, spans left in place (alignment)

    uv run python -m analysis.selective_quality results/p2s_quality [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

from analysis.obs_levels import bootstrap_ci, load_level, pct, per_question_f1


def task_numbers(t: dict[str, Any]) -> dict[str, float]:
    steps = [s for s in t["trace"]["steps"] if s["llm"].get("ok")]
    if not steps:
        return {"peak": 0, "kvs": 0.0, "prefill": 0, "trim_tokens": 0}
    peak = max(s["llm"]["prompt_n"] + s["llm"]["cache_n"] + s["llm"]["predicted_n"] for s in steps)
    kvs = 0.0      # token-seconds: context after each request, held until the next request starts
    for i, s in enumerate(steps):
        held = s["llm"]["prompt_n"] + s["llm"]["cache_n"] + s["llm"]["predicted_n"]
        t1 = steps[i + 1]["llm"]["t_start"] if i + 1 < len(steps) else s["llm"]["t_end"]
        kvs += held * max(0.0, t1 - s["llm"]["t_start"])
    trim = sum((s.get("trim") or {}).get("tokens", 0) for s in t["trace"]["steps"])
    return {"peak": peak, "kvs": kvs, "prefill": sum(s["llm"]["prompt_n"] for s in steps), "trim_tokens": trim}


def summaries(d: Path) -> list[dict[str, Any]]:
    return [json.loads((r / "summary.json").read_text(encoding="utf-8"))
            for r in sorted(d.glob("rep*")) if (r / "summary.json").exists()]


def row(reps: list[list[dict[str, Any]]], sums: list[dict[str, Any]], ref: dict[str, float] | None) -> dict[str, Any]:
    tasks = [t for r in reps for t in r]
    nums = [task_numbers(t) for t in tasks]
    peaks = [n["peak"] for n in nums]
    m = lambda k: round(st.mean(s[k] for s in sums if s.get(k) is not None), 3) if sums else None  # noqa: E731
    out: dict[str, Any] = {
        "reps": len(reps), "tasks": len(tasks),
        "completed": sum(t["status"] in ("ok", "forced") for t in tasks),
        "f1": round(st.mean(t["f1"] for t in tasks), 4),
        "f1_per_rep": [round(st.mean(t["f1"] for t in r), 4) for r in reps],
        "em": round(st.mean(t["em"] for t in tasks), 4),
        "llm_calls": round(st.mean(t["n_llm_calls"] for t in tasks), 2),
        "forced_share": round(sum(t["status"] == "forced" for t in tasks) / len(tasks), 3),
        "peak_mean": round(st.mean(peaks)), "peak_p95": pct(peaks, 0.95),
        "kv_token_s_per_task": round(st.mean(n["kvs"] for n in nums)),
        "prefill_per_task": round(st.mean(n["prefill"] for n in nums)),
        "trim_tokens_per_task": round(st.mean(n["trim_tokens"] for n in nums)),
        "trim_sf_sentences": sum(s.get("trim_sf_sentences", 0) for s in sums),
        "trim_spans_skipped": sum(s.get("trim_spans_skipped", 0) for s in sums),
        "trim_failures": sum(s.get("trim_failures", 0) for s in sums),
        "goodput": m("steady_goodput_per_min"), "throughput": m("steady_throughput_per_min"),
        "latency_p95_s": m("latency_p95_s"), "ttft_p50_s": m("ttft_p50_s"), "ttft_p95_s": m("ttft_p95_s"),
        "tpot_p50_ms": m("tpot_p50_ms"),
    }
    if ref is not None:
        mine = per_question_f1(reps)
        common = sorted(set(mine) & set(ref))
        diffs = [mine[q] - ref[q] for q in common]
        lo, hi = bootstrap_ci(diffs)
        out.update({"df1": round(sum(diffs) / len(diffs), 4), "df1_ci95": [round(lo, 4), round(hi, 4)]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir")
    ap.add_argument("--ref", default="none", help="point the F1 change is measured against")
    ap.add_argument("--json")
    args = ap.parse_args()
    root = Path(args.sweep_dir)
    order = [p.name for p in sorted(root.iterdir()) if p.is_dir()]
    order = ([args.ref] if args.ref in order else []) + [p for p in order if p != args.ref]
    data = {p: load_level(root / p) for p in order}
    data = {p: r for p, r in data.items() if r}
    ref = per_question_f1(data[args.ref]) if args.ref in data else None
    rows = {p: row(r, summaries(root / p), None if p == args.ref else ref) for p, r in data.items()}

    print(f"{'point':14} {'reps':>4} {'F1':>6} {'dF1 [95% CI]':>24} {'SF del':>6} {'peak mean/p95':>14} "
          f"{'KV tok·s':>9} {'prefill':>8} {'trimmed':>8} {'calls':>5} {'cap%':>4} {'goodput':>7} {'lat p95':>7} "
          f"{'TTFT p50/p95':>12} {'TPOT':>5}")
    for p, s in rows.items():
        d = f"{s['df1']:+.3f} [{s['df1_ci95'][0]:+.3f},{s['df1_ci95'][1]:+.3f}]" if "df1" in s else "-"
        print(f"{p:14} {s['reps']:>4} {s['f1']:6.3f} {d:>24} {s['trim_sf_sentences']:>6} "
              f"{s['peak_mean']:>6}/{s['peak_p95']:<7} {s['kv_token_s_per_task']:>9} {s['prefill_per_task']:>8} "
              f"{s['trim_tokens_per_task']:>8} {s['llm_calls']:>5} {100 * s['forced_share']:>4.0f} "
              f"{s['goodput'] or 0:>7.1f} {s['latency_p95_s'] or 0:>7.1f} "
              f"{s['ttft_p50_s'] or 0:>5.2f}/{s['ttft_p95_s'] or 0:<6.2f} {s['tpot_p50_ms'] or 0:>5.1f}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
