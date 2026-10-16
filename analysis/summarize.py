"""Collect per-point summaries of an experiment into one table.

    uv run python analysis/summarize.py results/phase1_sweep
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentserve.metrics.system import summarize_run  # noqa: E402


def collect(exp_dir: Path | str, recompute: bool = False) -> pd.DataFrame:
    rows = []
    for d in sorted(Path(exp_dir).iterdir()):
        if not (d / "tasks.jsonl").exists():
            continue
        s_path = d / "summary.json"
        summary = summarize_run(d) if recompute or not s_path.exists() else json.loads(s_path.read_text())
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        row = {"run": d.name, **{f"p.{k}": v for k, v in (meta.get("point") or {}).items()}}
        row.update({k: v for k, v in summary.items() if not isinstance(v, dict)})
        for g, gs in (summary.get("by_mix") or {}).items():
            row.update({f"mix.{g}.{k}": v for k, v in gs.items()})
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    exp = Path(sys.argv[1])
    df = collect(exp, recompute="--recompute" in sys.argv)
    df.to_csv(exp / "summary.csv", index=False)
    cols = [c for c in df.columns if c.startswith("p.")] + [
        c for c in ("n_finished", "n_failed", "throughput_tasks_per_min", "task_latency_p50", "task_latency_p95",
                    "task_latency_p99", "req_queue_p95", "prefix_cache_hit_ratio", "peak_phys_footprint_mb",
                    "peak_gpu_mem_used_mb", "em", "f1") if c in df]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df[cols].round(3).to_string(index=False))
