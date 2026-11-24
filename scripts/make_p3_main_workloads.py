"""Build the 16k main-experiment workloads (docs/phase2_ideas.md 4.6-4.7) from the base's measured capacity.

    uv run python -m scripts.make_p3_main_workloads --capacity 53.5
    uv run python -m scripts.make_p3_main_workloads --from-runs results/p3_cap16k/base_r1.75

Test split, 500 tasks (70% react_long, 30% react_short), bursty arrivals with fixed 60 s busy / 60 s quiet
periods, quiet rate = 20% of busy rate, mean rate = 50 / 80 / 110% of the capacity; seed r-1 for round r.
Writes configs/workloads/p3_main16k_l<load>_s<seed>.yaml and workloads/p3_main16k_l<load>_s<seed>.json and
prints the realized load of each (mean rate over whole busy + quiet cycles).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from edgeserve.workload import make_workload, save_workload

LOADS = (50, 80, 110)
SEEDS = (0, 1, 2)


def capacity_from_runs(point_dir: Path) -> float:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(point_dir.glob("rep*/summary.json"))]
    if not runs:
        raise SystemExit(f"no finished runs under {point_dir}")
    return round(statistics.mean(r["steady_throughput_per_min"] for r in runs), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--capacity", type=float, help="base capacity, tasks/min")
    g.add_argument("--from-runs", type=Path, help="sweep point dir whose runs' mean steady throughput is the capacity")
    args = ap.parse_args()
    cap = args.capacity if args.capacity is not None else capacity_from_runs(args.from_runs)
    print(f"capacity {cap} tasks/min")
    for load in LOADS:
        on = cap * load / 100 / 60 / 0.6
        for seed in SEEDS:
            name = f"p3_main16k_l{load}_s{seed}"
            spec = {"name": name, "split": "test", "n_tasks": 500, "seed": seed,
                    "mix": {"react_long": 0.7, "react_short": 0.3},
                    "arrival": {"type": "bursty", "rate_on": round(on, 4), "rate_off": round(0.2 * on, 4),
                                "mean_on_s": 60, "mean_off_s": 60, "fixed_periods": True}}
            Path(f"configs/workloads/{name}.yaml").write_text(
                f"# Selective KV step 3, main experiment at 16k (docs/phase2_ideas.md 4.6), written by\n"
                f"# scripts/make_p3_main_workloads.py: {load}% of the base's capacity ({cap} tasks/min), seed {seed}"
                f" = round {seed + 1}.\n" + json.dumps(spec) + "\n", encoding="utf-8", newline="\n")
            wl = make_workload(spec)
            digest = save_workload(wl, f"workloads/{name}.json")
            ts = sorted(t["arrival_s"] for t in wl["tasks"])
            full = int(ts[-1] // 120) * 120
            real = f"{sum(t < full for t in ts) / full * 60 / cap:.1%}" if full else "n/a (under one cycle)"
            print(f"  {name}: realized load {real} (plan {load}%), sha {digest}")
    Path("results/p3_main16k_capacity.txt").write_text(f"{cap}\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
