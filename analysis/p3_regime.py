"""Selective KV step 3, regime map: pool size x react_long share, base vs base + selective KV.

    uv run python -m analysis.p3_regime results/p3_regime [--json out.json]

Reads results/p3_regime/c{8k,12k,16k}_rl{30,70,100}_{base,kv}/rep* and ceiling_rl{30,70,100}/rep*
(numbers are means over repeats; F1 is paired on per-question means) and prints 3x3 grids (rows:
pool, columns: react_long share):
  goodput gain   steady goodput of base + kv relative to the base
  base/ceiling   steady throughput of the base relative to the 131k fixed-K=8 ceiling (how
                 memory-bound the base is: low = memory is the bottleneck)
  peak p95       per-task peak KV p95, base -> kv
  failures       failed tasks, base / kv
  dF1            paired F1 change of kv vs base on the same questions, with a bootstrap 95% CI
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analysis.obs_levels import bootstrap_ci, pct

POOLS = ("8k", "12k", "16k")
SHARES = ("30", "70", "100")


def load_point(point: Path) -> dict[str, Any] | None:
    """All finished repeats of a point, averaged; per-repeat values kept under "reps"."""
    reps = [r for r in (load(d) for d in sorted(point.glob("rep*"))) if r]
    if not reps:
        return None
    mean = lambda k: sum(r[k] for r in reps if r[k] is not None) / max(1, sum(r[k] is not None for r in reps))  # noqa: E731
    f1: dict[str, list[float]] = {}
    for r in reps:
        for q, v in r["f1"].items():
            f1.setdefault(q, []).append(v)
    return {"n": len(reps), "goodput": mean("goodput"), "throughput": mean("throughput"),
            "failed": mean("failed"), "peak_p95": mean("peak_p95"), "tpot": mean("tpot"),
            "refused": mean("refused"), "drops": mean("drops"),
            "f1": {q: sum(v) / len(v) for q, v in f1.items()},
            "reps": [{k: r[k] for k in ("goodput", "throughput", "drops", "failed")} for r in reps]}


def load(run: Path) -> dict[str, Any] | None:
    if not (run / "summary.json").exists():
        return None
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    tasks = [json.loads(x) for x in (run / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if x]
    peaks = []
    for t in tasks:
        ok = [x["llm"] for x in t["trace"]["steps"] if x["llm"].get("ok")]
        if ok:
            peaks.append(max(q["prompt_n"] + q["cache_n"] + q["predicted_n"] for q in ok))
    km = s.get("kvm") or {}
    return {"summary": s, "f1": {t["task_id"]: t["f1"] for t in tasks},
            "goodput": s.get("steady_goodput_per_min"), "throughput": s.get("steady_throughput_per_min"),
            "failed": s["failed"], "peak_p95": pct(peaks, 0.95) if peaks else None,
            "tpot": s.get("tpot_p50_ms"), "refused": km.get("refused", 0),
            "drops": sum(v for k, v in km.get("give_ups", {}).items() if k.endswith(":drop"))}


def grid(title: str, cell) -> None:
    print(f"\n{title}")
    print(f"{'pool':>6} " + " ".join(f"{'react_long ' + s + '%':>22}" for s in SHARES))
    for p in POOLS:
        print(f"{p:>6} " + " ".join(f"{cell(p, s):>22}" for s in SHARES))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir")
    ap.add_argument("--json")
    args = ap.parse_args()
    root = Path(args.sweep_dir)
    runs = {p.name: load_point(p) for p in root.iterdir() if p.is_dir()}
    get = lambda name: runs.get(name)  # noqa: E731
    out: dict[str, Any] = {}

    def cell_data(p: str, s: str) -> dict[str, Any] | None:
        b, k, c = get(f"c{p}_rl{s}_base"), get(f"c{p}_rl{s}_kv"), get(f"ceiling_rl{s}")
        if not b or not k:
            return None
        common = sorted(set(b["f1"]) & set(k["f1"]))
        d = [k["f1"][q] - b["f1"][q] for q in common]
        lo, hi = bootstrap_ci(d) if d else (float("nan"), float("nan"))
        r = {"base_goodput": b["goodput"], "kv_goodput": k["goodput"],
             "goodput_gain": (k["goodput"] / b["goodput"] - 1) if b["goodput"] else None,
             "throughput_gain": (k["throughput"] / b["throughput"] - 1) if b["throughput"] else None,
             "base_over_ceiling": (b["throughput"] / c["throughput"]) if c and c["throughput"] else None,
             "peak_p95": [b["peak_p95"], k["peak_p95"]], "failed": [b["failed"], k["failed"]],
             "refused": [b["refused"], k["refused"]], "drops": [b["drops"], k["drops"]],
             "tpot": [b["tpot"], k["tpot"]],
             "df1": sum(d) / len(d) if d else None, "df1_ci95": [lo, hi]}
        out[f"{p}_rl{s}"] = r
        return r

    data = {(p, s): cell_data(p, s) for p in POOLS for s in SHARES}
    fmt = lambda f: (lambda p, s: f(data[(p, s)]) if data[(p, s)] else "-")  # noqa: E731
    grid("goodput gain of selective KV (base -> base + kv, correct tasks/min)",
         fmt(lambda r: f"{r['base_goodput']:.1f}->{r['kv_goodput']:.1f} ({100 * r['goodput_gain']:+.0f}%)"))
    grid("base / ceiling throughput (low = memory-bound)",
         fmt(lambda r: f"{100 * r['base_over_ceiling']:.0f}%" if r["base_over_ceiling"] else "-"))
    grid("throughput gain of selective KV", fmt(lambda r: f"{100 * r['throughput_gain']:+.0f}%"))
    grid("peak KV p95 per task (base -> kv)", fmt(lambda r: f"{r['peak_p95'][0]:.0f} -> {r['peak_p95'][1]:.0f}"))
    grid("failed tasks (base / kv, mean over repeats)", fmt(lambda r: f"{r['failed'][0]:.1f} / {r['failed'][1]:.1f}"))
    grid("TPOT p50 ms (base / kv)", fmt(lambda r: f"{r['tpot'][0]:.1f} / {r['tpot'][1]:.1f}"))
    grid("paired F1 change of kv vs base [95% CI]",
         fmt(lambda r: f"{r['df1']:+.3f} [{r['df1_ci95'][0]:+.2f},{r['df1_ci95'][1]:+.2f}]"))
    print("\nrepeats (goodput per repeat, gate drops in brackets):")
    for p in POOLS:
        for sh in SHARES:
            for cfg in ("base", "kv"):
                r = get(f"c{p}_rl{sh}_{cfg}")
                if r and r["n"] > 1:
                    reps = "  ".join(f"{x['goodput']:5.1f} ({x['drops']:4})" for x in r["reps"])
                    gs = [x["goodput"] for x in r["reps"]]
                    sd = (sum((g - sum(gs) / len(gs)) ** 2 for g in gs) / (len(gs) - 1)) ** 0.5
                    print(f"  {p:>4} rl{sh:<3} {cfg:4}  mean {r['goodput']:5.1f} sd {sd:4.1f}  |  {reps}")
    ceil = {s: get(f"ceiling_rl{s}") for s in SHARES}
    print("\nceilings (131k, fixed K=8) throughput / goodput: "
          + ", ".join(f"rl{s}: {c['throughput']}/{c['goodput']}" for s, c in ceil.items() if c))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
