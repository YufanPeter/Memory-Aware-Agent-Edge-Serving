"""Run an experiment (all sweep points x seeds).

    uv run python experiments/run.py configs/experiments/smoke.yaml
    uv run python experiments/run.py configs/experiments/phase1_sweep.yaml --set workload.n_tasks=20 --limit 2

Results go to results/<name>/<point_id>/; finished points (summary.json present)
are skipped unless --force, so an interrupted sweep can be resumed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentserve.core.config import expand_sweep, load_config, resolve, set_dotted  # noqa: E402
from agentserve.runner import run_point  # noqa: E402

HEADLINE = ["n_finished", "n_failed", "throughput_tasks_per_min", "task_latency_p50", "task_latency_p95",
            "req_queue_p95", "prefix_cache_hit_ratio", "peak_phys_footprint_mb", "peak_gpu_mem_used_mb", "em", "f1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a dotted config key (value parsed as YAML)")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--limit", type=int, help="run only the first N points")
    ap.add_argument("--force", action="store_true", help="re-run points that already have results")
    args = ap.parse_args()
    args.config = Path(args.config).resolve()
    args.out = Path(args.out).resolve()
    os.chdir(ROOT)  # relative model/data paths in configs are relative to the project root

    cfg = load_config(Path(args.config).resolve())
    for kv in args.set:
        k, _, v = kv.partition("=")
        set_dotted(cfg, k, resolve(yaml.safe_load(v), args.config.parent))

    points = list(expand_sweep(cfg))[: args.limit]
    root = Path(args.out) / cfg["name"]
    print(f"[{cfg['name']}] {len(points)} point(s) -> {root}")
    for i, (pid, point) in enumerate(points, 1):
        out = root / pid
        if (out / "summary.json").exists() and not args.force:
            print(f"  ({i}/{len(points)}) {pid}: done, skipping")
            continue
        print(f"  ({i}/{len(points)}) {pid} ...", flush=True)
        summary = asyncio.run(run_point(point, out))
        print("     " + json.dumps({k: round(summary[k], 3) if isinstance(summary.get(k), float) else summary.get(k)
                                    for k in HEADLINE if k in summary}))

    from analysis.summarize import collect
    df = collect(root)
    if not df.empty:
        df.to_csv(root / "summary.csv", index=False)
        print(f"wrote {root / 'summary.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
