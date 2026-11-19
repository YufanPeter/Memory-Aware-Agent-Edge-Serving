"""Pick B1's fixed K per (budget point, trim mode) from the p3_tune runs with the Phase 2 tuning
rule, and write the dev sweep for those picks (docs/phase3_spec.md §4 step 2).

Rule (analysis/phase2_report.pick): the K with the highest overall goodput among those completing
>= 95% of tasks; if none does, the K with the highest completion (goodput breaks ties). Points are
named c<ctx>_d<delay>_b1_<mode>_k<K>; every point's rep0 must be finished.

    uv run --no-sync python -m scripts.pick_b1 [--root results] [--sweep p3_tune] \
        [--spec configs/sweeps/p3_tune.yaml] [--out configs/sweeps/p3_dev_b1.yaml]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

COMPLETION_MIN = 0.95
NAME = re.compile(r"^(?P<group>c\d+k_d[\d.]+_b1_(?P<mode>\w+?))_k(?P<k>\d+)$")


def pick(cands: list[dict]) -> dict:
    ok = [c for c in cands if c["completion"] >= COMPLETION_MIN]
    if ok:
        return max(ok, key=lambda c: c["goodput"])
    return max(cands, key=lambda c: (c["completion"], c["goodput"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results")
    ap.add_argument("--sweep", default="p3_tune")
    ap.add_argument("--spec", default="configs/sweeps/p3_tune.yaml")
    ap.add_argument("--out", default="configs/sweeps/p3_dev_b1.yaml")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workload", default="workloads/dev_long_200.json")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    points = {p["name"]: {**spec.get("defaults", {}), **p} for p in spec["points"]}
    groups: dict[str, list[dict]] = {}
    for name, p in points.items():
        m = NAME.match(name)
        if not m:
            continue
        s = json.loads((Path(args.root) / args.sweep / name / "rep0" / "summary.json").read_text(encoding="utf-8"))
        groups.setdefault(m["group"], []).append({
            "name": name, "k": int(m["k"]), "point": p,
            "completion": s["completed"] / s["n_tasks"], "goodput": s["goodput_tasks_per_min"] or 0.0})
    out_points, notes = [], []
    for g, cands in sorted(groups.items()):
        c = pick(cands)
        floor = "meets 95%" if c["completion"] >= COMPLETION_MIN else "fallback: no K reached 95%"
        notes.append(f"# {g}: " + ", ".join(f"K={x['k']} {x['goodput']:.2f}/min {100 * x['completion']:.1f}%"
                                         for x in sorted(cands, key=lambda x: x["k"])) + f" -> K={c['k']} ({floor})")
        p = c["point"]
        keep = {k: p[k] for k in ("ctx", "delay_scale", "k", "trim", "trim_eager") if k in p}
        out_points.append({"name": c["name"], **keep})
    head = ["# Written by scripts/pick_b1.py from the p3_tune runs (Phase 2 tuning rule). B1 at its retuned K",
            "# on dev, same sweep name as p3_dev.yaml.", *notes]
    body = {"name": "p3_dev",
            "defaults": {"workload": args.workload, "parallel": 64, "reps": args.reps, "policy": "observe"},
            "points": out_points}
    Path(args.out).write_text("\n".join(head) + "\n" + yaml.safe_dump(body, sort_keys=False, default_flow_style=None,
                                                                     width=120), encoding="utf-8")
    print("\n".join(notes))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
