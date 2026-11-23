"""Selective KV step 3, tuning report: capacity per configuration, the best parameter of each, and
paired per-profile F1 of the selective configurations against the base at the same rho.

    uv run python -m analysis.p3_tune results/p3_tune_dev [--json out.json]

Points are named <family>_<param> (fixed_k2, base_r1.5, kv_r2.0, text_r2.5, kvp_r1.5). Best per
family = highest steady goodput among points with no failed task. In this closed-loop tuning run
every task arrives at t=0, so task latency is mostly queueing; service time (admission to done)
is the per-task cost.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

from analysis.obs_levels import bootstrap_ci, pct

PROFILES = ("react_short", "react_long", "react_explain", "oneshot_long")


def load(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    if not (run / "summary.json").exists():
        return None
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    tasks = [json.loads(x) for x in (run / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if x]
    return s, tasks


def row(s: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    km = s.get("kvm") or {}
    gu = km.get("give_ups", {})
    service = [t["service_s"] for t in tasks]
    out = {
        "completed": s["completed"], "failed": s["failed"],
        "goodput": s.get("steady_goodput_per_min"), "throughput": s.get("steady_throughput_per_min"),
        "f1": round(s["f1"], 4), "service_p50": round(pct(service, 0.5), 1), "service_p95": round(pct(service, 0.95), 1),
        "ttft_p95": s.get("ttft_p95_s"), "tpot_p50": s.get("tpot_p50_ms"),
        "drops": sum(v for k, v in gu.items() if k.endswith(":drop")),
        "peak_resident": km.get("peak_resident_tokens"),
        "trims": s.get("trims", 0), "trim_deferred": s.get("trim_deferred", 0),
        "trim_tokens": s.get("trim_tokens_total", 0), "prefill": s.get("prefill_tokens_total"),
    }
    for p in PROFILES:
        f = [t["f1"] for t in tasks if t["profile"] == p]
        out[f"f1_{p}"] = round(st.mean(f), 3) if f else None
    return out


def paired(a: list[dict[str, Any]], b: list[dict[str, Any]], profile: str | None) -> dict[str, Any]:
    """F1 of b minus F1 of a on the same tasks (same workload), optionally one profile."""
    fa = {t["task_id"]: t["f1"] for t in a if profile in (None, t["profile"])}
    fb = {t["task_id"]: t["f1"] for t in b if profile in (None, t["profile"])}
    common = sorted(set(fa) & set(fb))
    d = [fb[k] - fa[k] for k in common]
    lo, hi = bootstrap_ci(d) if d else (float("nan"), float("nan"))
    return {"n": len(d), "df1": round(sum(d) / len(d), 3) if d else None, "ci95": [round(lo, 3), round(hi, 3)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir")
    ap.add_argument("--json")
    args = ap.parse_args()
    root = Path(args.sweep_dir)
    runs = {p.name: load(p / "rep0") for p in sorted(root.iterdir()) if p.is_dir()}
    runs = {k: v for k, v in runs.items() if v}
    rows = {k: row(*v) for k, v in runs.items()}

    print(f"{'point':10} {'done':>5} {'fail':>4} {'goodput':>7} {'thru':>6} {'svc p50/p95':>11} {'TTFT95':>6} "
          f"{'TPOT50':>6} {'drops':>5} {'trims':>5} {'defer':>5} {'trimmed':>7} {'peak':>6} {'F1':>6} "
          + " ".join(f"{p.split('_')[1][:5]:>6}" for p in PROFILES))
    for k, r in rows.items():
        print(f"{k:10} {r['completed']:>5} {r['failed']:>4} {r['goodput'] or 0:>7.1f} {r['throughput'] or 0:>6.1f} "
              f"{r['service_p50']:>5}/{r['service_p95']:<5} {r['ttft_p95'] or 0:>6.2f} {r['tpot_p50'] or 0:>6.1f} "
              f"{r['drops']:>5} {r['trims']:>5} {r['trim_deferred']:>5} {r['trim_tokens']:>7} "
              f"{r['peak_resident'] or 0:>6} {r['f1']:>6.3f} "
              + " ".join(f"{r['f1_' + p] if r['f1_' + p] is not None else float('nan'):>6.3f}" for p in PROFILES))

    best: dict[str, str] = {}
    for k, r in rows.items():
        fam = k.rsplit("_", 1)[0]
        if r["failed"] == 0 and (fam not in best or (r["goodput"] or 0) > (rows[best[fam]]["goodput"] or 0)):
            best[fam] = k
    print("\nbest per configuration (highest goodput, no failed task):")
    for fam, k in best.items():
        r = rows[k]
        print(f"  {fam:8} -> {k:10} goodput {r['goodput']}, throughput {r['throughput']}, F1 {r['f1']}")

    print("\npaired F1 change vs base at the same rho (same 200 tasks), overall and per profile:")
    pairs = {}
    for k in rows:
        fam, param = k.rsplit("_", 1)
        ref = f"base_{param}"
        if fam in ("kv", "text", "kvp") and ref in runs:
            pairs[k] = {p or "all": paired(runs[ref][1], runs[k][1], p) for p in (None, *PROFILES)}
            line = "  ".join(f"{p[:11]} {v['df1']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]"
                             for p, v in pairs[k].items() if v["n"])
            print(f"  {k:10} vs {ref:9} {line}")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "best": best, "paired": pairs}, indent=1),
                                   encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
