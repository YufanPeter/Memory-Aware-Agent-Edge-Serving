"""Selective KV step 3, main experiment report (16k pool, bursty test workloads, docs/phase2_ideas.md 4.6).

Per load level and configuration: mean +- sd over rounds of goodput and throughput (steady window), task
latency p50/p95/p99 over all 500 tasks (queueing included), TTFT and TPOT, completions and failures, F1,
and what explains them (KV given up, gate waits, idle-KV share, peak books vs the pool, realized load).
Then the rule fixed in advance (docs/phase2_ideas.md 4.5): goodput is compared at 110% only; latency
p95 / p99 at 80% and 110%; a gap is clear when |mean difference| > 2 x the larger sd of the two
configurations. F1: paired per task within each round (same workload), pooled over rounds.

    uv run python -m analysis.p3_main results/p3_main16k --capacity 53.5 [--json out.json]

Point names: r<round>_l<load>_<config>, e.g. r1_l110_base, r2_l80_kv, r1_l50_fixedk3.
Idle-KV share = pool token-seconds held by agents inside a tool call / all resident token-seconds. For the
managed configurations it is rebuilt from kv_state events (they know about drops, trims and the shared
prefix); fixed K has no KV manager and uses the Phase 1 reconstruction (analysis/kv_timeline.py).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path
from typing import Any

from analysis import kv_timeline
from analysis.check_kvm import check
from analysis.obs_levels import bootstrap_ci

NAME = re.compile(r"r(\d+)_l(\d+)_(\w+)$")
CYCLE_S = 120.0   # one busy + one quiet period


def idle_share_from_kv_state(events: Path) -> float | None:
    """Share of resident pool token-seconds held by idle agents (owner in a tool call)."""
    slots: dict[int, tuple[str, int, int]] = {}   # slot -> (state, tokens, shared)
    tok_s = {"idle": 0.0, "all": 0.0}
    last_t = None
    with open(events, encoding="utf-8") as f:
        for line in f:
            if '"kv_state"' not in line:
                continue
            e = json.loads(line)
            if last_t is not None:
                dt = e["t"] - last_t
                for state, tokens, shared in slots.values():
                    own = tokens - min(shared, tokens) if state != "golden" else tokens
                    tok_s["all"] += own * dt
                    if state == "idle":
                        tok_s["idle"] += own * dt
            last_t = e["t"]
            slots[e["slot"]] = (e["to"], e["tokens"], e.get("shared", 0))
    return round(tok_s["idle"] / tok_s["all"], 4) if tok_s["all"] else None


def realized_load(tasks: list[dict[str, Any]], capacity: float) -> float | None:
    """Mean arrival rate over whole busy + quiet cycles, as a share of the base's capacity."""
    ts = sorted(t["arrival_s"] for t in tasks)
    full = int(ts[-1] // CYCLE_S) * CYCLE_S
    if not full:
        return None
    return round(sum(t < full for t in ts) / full * 60 / capacity, 3)


def one_run(run: Path, capacity: float) -> dict[str, Any]:
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    tasks = [json.loads(x) for x in (run / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if x]
    km = s.get("kvm") or {}
    gu = km.get("give_ups", {})
    safety = check(run)
    if km:
        idle = idle_share_from_kv_state(run / "events.jsonl")
    else:
        idle = kv_timeline.analyze(run)["share"]["idle_tool"]
    return {
        "goodput": s["steady_goodput_per_min"], "throughput": s["steady_throughput_per_min"],
        "lat_p50": s["latency_p50_s"], "lat_p95": s["latency_p95_s"], "lat_p99": s["latency_p99_s"],
        "ttft_p50": s["ttft_p50_s"], "ttft_p95": s["ttft_p95_s"],
        "tpot_p50": s["tpot_p50_ms"], "tpot_p95": s["tpot_p95_ms"],
        "completed": s["completed"], "failed": s["failed"], "refused": km.get("refused", 0),
        "f1": s["f1"], "drops": sum(v for k, v in gu.items() if k.endswith(":drop")),
        "gate_wait_s": km.get("gate_wait_s", 0.0), "idle_share": idle,
        "peak_books": km.get("peak_resident_tokens"), "pool": km.get("pool_tokens"),
        "server_overflows": safety["server_overflows"], "server_purges": safety["server_purges"],
        "load": realized_load(tasks, capacity), "trims": s.get("trims", 0),
        "_f1": {t["task_id"]: (t["profile"], t["f1"]) for t in tasks},
    }


def agg(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"runs": len(runs)}
    for k in runs[0]:
        if k.startswith("_"):
            continue
        vals = [r[k] for r in runs if isinstance(r[k], (int, float))]
        out[k] = round(st.mean(vals), 3) if vals else None
        out[f"{k}_sd"] = round(st.stdev(vals), 3) if len(vals) > 1 else None
        out[f"{k}_all"] = [r[k] for r in runs]
    return out


def paired_f1(pairs: list[tuple[dict[str, Any], dict[str, Any]]], profile: str | None) -> dict[str, Any]:
    """F1 of b minus F1 of a, per task, within each round (same workload), pooled over rounds."""
    d = []
    for a, b in pairs:
        for tid, (prof, fa) in a.items():
            if tid in b and profile in (None, prof):
                d.append(b[tid][1] - fa)
    if not d:
        return {"n": 0, "df1": None, "ci95": None}
    lo, hi = bootstrap_ci(d)
    return {"n": len(d), "df1": round(sum(d) / len(d), 3), "ci95": [round(lo, 3), round(hi, 3)]}


def verdict(a: dict[str, Any], b: dict[str, Any], key: str) -> dict[str, Any]:
    if a[key] is None or b[key] is None:
        return {"gap": None, "rel": None, "bar": None, "verdict": "n/a (missing)"}
    gap = b[key] - a[key]
    sds = [x for x in (a[f"{key}_sd"], b[f"{key}_sd"]) if x is not None]
    bar = 2 * max(sds) if sds else None
    v = "n/a (single runs)" if bar is None else ("clear" if abs(gap) > bar else "not clear")
    return {"gap": round(gap, 3), "rel": round(gap / a[key], 3) if a[key] else None,
            "bar": round(bar, 3) if bar is not None else None, "verdict": v}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir")
    ap.add_argument("--capacity", type=float, required=True,
                    help="base capacity the loads were set from, tasks/min (results/p3_main16k_capacity.txt)")
    ap.add_argument("--json")
    args = ap.parse_args()

    runs: dict[tuple[int, str], dict[int, dict[str, Any]]] = {}
    for p in sorted(Path(args.sweep_dir).iterdir()):
        m = NAME.match(p.name)
        if not m or not (p / "rep0" / "summary.json").exists():
            continue
        rnd, load, cfg = int(m.group(1)), int(m.group(2)), m.group(3)
        runs.setdefault((load, cfg), {})[rnd] = one_run(p / "rep0", args.capacity)

    rows = {f"l{load}_{cfg}": agg(list(r.values())) for (load, cfg), r in sorted(runs.items())}
    hdr = (f"{'load':5} {'config':8} {'n':>2} {'goodput':>12} {'thru':>6} {'lat p50/p95/p99 (s)':>20} "
           f"{'TTFT50/95':>10} {'TPOT50/95':>11} {'done':>5} {'fail':>4} {'F1':>5} {'drops':>6} {'gwait':>6} "
           f"{'idle':>5} {'books':>6} {'ovf':>3} {'load':>5}")
    print(hdr)
    for (load, cfg), _ in sorted(runs.items()):
        r = rows[f"l{load}_{cfg}"]
        if r["goodput"] is None or r["throughput"] is None:
            print(f"{load:>4}% {cfg:8} {r['runs']:>2}  no steady window (fewer than 10 completions)")
            continue
        gp = f"{r['goodput']:.1f}" + (f"+-{r['goodput_sd']:.1f}" if r["goodput_sd"] is not None else "")
        print(f"{load:>4}% {cfg:8} {r['runs']:>2} {gp:>12} {r['throughput']:>6.1f} "
              f"{r['lat_p50']:>6.1f}/{r['lat_p95']:>6.1f}/{r['lat_p99']:>6.1f} "
              f"{r['ttft_p50']:>4.2f}/{r['ttft_p95']:<5.2f} {r['tpot_p50']:>5.1f}/{r['tpot_p95']:<5.1f} "
              f"{r['completed']:>5.0f} {r['failed']:>4.0f} {r['f1']:>5.3f} {r['drops']:>6.0f} {r['gate_wait_s']:>6.0f} "
              f"{(r['idle_share'] or 0):>5.2f} {(r['peak_books'] or 0):>6.0f} {r['server_overflows']:>3.0f} "
              f"{(r['load'] or 0):>5.0%}")

    print("\nrule fixed in advance (|gap| > 2 x larger sd): kv vs base")
    decision: dict[str, Any] = {}
    for load, keys in ((110, ("goodput", "lat_p95", "lat_p99")), (80, ("lat_p95", "lat_p99")), (50, ())):
        a, b = rows.get(f"l{load}_base"), rows.get(f"l{load}_kv")
        if not a or not b:
            continue
        for key in keys:
            v = verdict(a, b, key)
            decision[f"l{load}_{key}"] = v
            rel = f"{v['rel']:+.1%}" if v["rel"] is not None else "n/a"
            gap = f"{v['gap']:+8.2f}" if v["gap"] is not None else "n/a"
            print(f"  {load:>3}% {key:8} base {a[key]}  kv {b[key]}  gap {gap} ({rel})  bar {v['bar']}  -> {v['verdict']}")
        pairs = [(runs[(load, "base")][k]["_f1"], runs[(load, "kv")][k]["_f1"])
                 for k in runs[(load, "base")] if k in runs[(load, "kv")]]
        f1 = {p or "all": paired_f1(pairs, p) for p in (None, "react_long", "react_short")}
        decision[f"l{load}_f1"] = f1
        print(f"  {load:>3}% paired F1 kv - base: " + "  ".join(f"{k} {v['df1']:+.3f} {v['ci95']}" for k, v in f1.items() if v["n"]))

    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "decision": decision}, indent=1),
                                   encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
