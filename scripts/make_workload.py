"""Generate and freeze a workload.

    uv run python -m scripts.make_workload configs/workloads/smoke.yaml [--out workloads/smoke.json]
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from edgeserve.workload import load_spec, make_workload, save_workload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--out")
    args = ap.parse_args()
    spec = load_spec(args.spec)
    out = Path(args.out or f"workloads/{spec['name']}.json")
    wl = make_workload(spec)
    digest = save_workload(wl, out)
    mix = Counter(t["profile"] for t in wl["tasks"])
    print(f"{out}: {len(wl['tasks'])} tasks, last arrival {wl['tasks'][-1]['arrival_s']}s, "
          f"mix {dict(mix)}, sha {digest}")


if __name__ == "__main__":
    main()
