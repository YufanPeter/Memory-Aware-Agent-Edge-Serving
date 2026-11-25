"""Selective KV step 3, tuning report: capacity per configuration, the best parameter of each, and
paired per-profile F1 of the selective configurations against the base at the same rho.

    uv run python -m analysis.p3_tune results/p3_tune_dev [--json out.json]

Points are named <family>_<param> (fixed_k2, base_r1.5, kv_r2.0, text_r2.5, kvp_r1.5). A point may have
several runs (rep0, rep1, ...): numbers are means over its runs, and goodput also shows each run and the
standard deviation. Best per family = highest mean steady goodput among points whose only failures are
budget refusals (tasks whose context can never fit in the pool, refused by the KV manager before they
could overflow). The selective configurations' best is then compared with the base's best by the rule
fixed in advance (docs/phase2_ideas.md section 4.5): the gap is clear when it exceeds twice the larger
standard deviation of the two. In this closed-loop tuning run
every task arrives at t=0, so task latency is mostly queueing; service time (admission to done)
is the per-task cost. A second table gives the topic metrics of analysis/run_metrics.py (memory, recompute,
agent, edge), averaged over each point's runs.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

from analysis.obs_levels import bootstrap_ci, pct
from analysis.run_metrics import run_metrics

PROFILES = ("react_short", "react_long", "react_explain", "oneshot_long")


def load(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path] | None:
    if not (run / "summary.json").exists():
        return None
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    tasks = [json.loads(x) for x in (run / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if x]
    return s, tasks, run


def row(s: dict[str, Any], tasks: list[dict[str, Any]], run: Path) -> dict[str, Any]:
    km = s.get("kvm") or {}
    gu = km.get("give_ups", {})
    service = [t["service_s"] for t in tasks]
    out = {
        "completed": s["completed"], "failed": s["failed"], "refused": km.get("refused", 0),
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
    out.update({f"m_{k}": v for k, v in run_metrics(run).items()})
    return out


def mean_row(rs: list[dict[str, Any]]) -> dict[str, Any]:
    """One point's row: the mean of its runs' rows, plus goodput and throughput per run."""
    out: dict[str, Any] = {}
    for k in rs[0]:
        vals = [r[k] for r in rs if r[k] is not None]
        out[k] = round(st.mean(vals), 3) if vals and all(isinstance(v, (int, float)) for v in vals) else rs[0][k]
    for k in ("goodput", "throughput"):
        out[f"{k}_runs"] = [r[k] for r in rs]
        vals = [v for v in out[f"{k}_runs"] if v is not None]
        out[f"{k}_sd"] = round(st.stdev(vals), 2) if len(vals) > 1 else None
    out["runs"] = len(rs)
    out["only_refusals"] = all(r["failed"] == r["refused"] for r in rs)   # per run, not on the means
    return out


def task_f1(runs: list[list[dict[str, Any]]]) -> dict[str, tuple[str, float]]:
    """task id -> (profile, F1 averaged over the point's runs)."""
    acc: dict[str, list[float]] = {}
    prof: dict[str, str] = {}
    for tasks in runs:
        for t in tasks:
            acc.setdefault(t["task_id"], []).append(t["f1"])
            prof[t["task_id"]] = t["profile"]
    return {k: (prof[k], st.mean(v)) for k, v in acc.items()}


def paired(a: dict[str, tuple[str, float]], b: dict[str, tuple[str, float]], profile: str | None) -> dict[str, Any]:
    """F1 of b minus F1 of a on the same tasks (same workload, run-averaged), optionally one profile."""
    fa = {k: f for k, (p, f) in a.items() if profile in (None, p)}
    fb = {k: f for k, (p, f) in b.items() if profile in (None, p)}
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
    runs = {p.name: [x for r in sorted(p.glob("rep*")) if (x := load(r))]
            for p in sorted(root.iterdir()) if p.is_dir()}
    runs = {k: v for k, v in runs.items() if v}
    rows = {k: mean_row([row(*x) for x in v]) for k, v in runs.items()}
    f1s = {k: task_f1([x[1] for x in v]) for k, v in runs.items()}

    print(f"{'point':10} {'done':>5} {'fail':>4} {'ref':>3} {'goodput':>7} {'thru':>6} {'svc p50/p95':>11} {'TTFT95':>6} "
          f"{'TPOT50':>6} {'drops':>5} {'trims':>5} {'defer':>5} {'trimmed':>7} {'peak':>6} {'F1':>6} "
          + " ".join(f"{p.split('_')[1][:5]:>6}" for p in PROFILES))
    for k, r in rows.items():
        print(f"{k:10} {r['completed']:>5} {r['failed']:>4} {r['refused']:>3} {r['goodput'] or 0:>7.1f} {r['throughput'] or 0:>6.1f} "
              f"{r['service_p50']:>5}/{r['service_p95']:<5} {r['ttft_p95'] or 0:>6.2f} {r['tpot_p50'] or 0:>6.1f} "
              f"{r['drops']:>5} {r['trims']:>5} {r['trim_deferred']:>5} {r['trim_tokens']:>7} "
              f"{r['peak_resident'] or 0:>6} {r['f1']:>6.3f} "
              + " ".join(f"{r['f1_' + p] if r['f1_' + p] is not None else float('nan'):>6.3f}" for p in PROFILES))

    print("\nmemory / recompute / agent / edge (analysis/run_metrics.py; means over runs)")
    print(f"{'point':10} {'KV mean/peak':>13} {'pool%':>5} {'agents':>6} {'KV/agent':>8} {'idle':>5} {'drops':>5} "
          f"{'gwait s':>7} {'trimKV':>7} {'recomp':>6} {'prefill':>7} {'hit':>5} {'admit p95':>9} {'calls':>5} "
          f"{'forced':>6} {'EM':>5} {'out':>4} {'GPU MiB':>7} {'util':>4} {'RSS':>5}")
    for k, r in rows.items():
        g = lambda x: r.get(f"m_{x}") or 0  # noqa: E731
        print(f"{k:10} {g('kv_mean'):>6.0f}/{g('kv_peak'):<6.0f} {g('kv_mean_share'):>5.0%} {g('agents_mean'):>6.2f} "
              f"{g('kv_per_agent'):>8.0f} {g('idle_share'):>5.2f} {g('drops'):>5.0f} {g('gate_wait_s'):>7.1f} "
              f"{g('trim_kv_tokens'):>7.0f} {g('recompute_per_task'):>6.0f} {g('prefill_per_task'):>7.0f} "
              f"{g('cache_hit_rate'):>5.2f} {g('admit_wait_p95'):>9.1f} {g('llm_calls_per_task'):>5.2f} "
              f"{g('forced_share'):>6.1%} {g('em'):>5.3f} {g('output_tokens_per_request'):>4.0f} "
              f"{g('gpu_mem_peak_mib'):>7.0f} {g('gpu_util_mean'):>4.0f} {g('host_rss_peak_mib'):>5.0f}")

    best: dict[str, str] = {}
    for k, r in rows.items():
        fam = k.rsplit("_", 1)[0]
        if r["only_refusals"] and r["goodput"] and (fam not in best or r["goodput"] > (rows[best[fam]]["goodput"] or 0)):
            best[fam] = k
    print("\ngoodput per run (mean +- sd):")
    for k, r in rows.items():
        sd = f" +- {r['goodput_sd']:.2f}" if r["goodput_sd"] is not None else ""
        print(f"  {k:10} {r['goodput'] or 0:6.2f}{sd:9} runs {r['goodput_runs']}   throughput runs {r['throughput_runs']}")

    print("\nbest per configuration (highest mean goodput, no failure other than budget refusals):")
    for fam, k in best.items():
        r = rows[k]
        params = sorted(float(p.rsplit("_", 1)[1].lstrip("rk")) for p in rows if p.rsplit("_", 1)[0] == fam)
        here = float(k.rsplit("_", 1)[1].lstrip("rk"))
        edge = "  (at the edge of the grid: test one more point outside)" if here in (params[0], params[-1]) and len(params) > 1 else ""
        print(f"  {fam:8} -> {k:10} goodput {r['goodput']} (runs {r['goodput_runs']}), "
              f"throughput {r['throughput']}, F1 {r['f1']}{edge}")

    decision = {}
    if "base" in best:
        b = rows[best["base"]]
        print("\nrule fixed in advance: a clear gap exceeds 2 x the larger sd of the two best points")
        for fam in ("kv", "text", "kvp"):
            if fam not in best:
                continue
            r = rows[best[fam]]
            gap = r["goodput"] - b["goodput"]
            sds = [x for x in (r["goodput_sd"], b["goodput_sd"]) if x is not None]
            bar = 2 * max(sds) if sds else None
            verdict = "n/a (single runs)" if bar is None else ("clear" if abs(gap) > bar else "not clear")
            decision[fam] = {"gap": round(gap, 2), "rel": round(gap / b["goodput"], 3), "bar": bar, "verdict": verdict,
                             "f1": paired(f1s[best["base"]], f1s[best[fam]], None)}
            print(f"  {best[fam]} vs {best['base']}: goodput {gap:+.2f} ({gap / b['goodput']:+.1%}), bar "
                  f"{bar if bar is None else round(bar, 2)} -> {verdict}; throughput "
                  f"{r['throughput'] - b['throughput']:+.1f} ({(r['throughput'] - b['throughput']) / b['throughput']:+.1%}); "
                  f"paired F1 {decision[fam]['f1']['df1']:+.3f} {decision[fam]['f1']['ci95']}")

    print("\npaired F1 change vs base at the same rho (same 200 tasks, run-averaged), overall and per profile:")
    pairs = {}
    for k in rows:
        fam, param = k.rsplit("_", 1)
        ref = f"base_{param}"
        if fam in ("kv", "text", "kvp") and ref in runs:
            pairs[k] = {p or "all": paired(f1s[ref], f1s[k], p) for p in (None, *PROFILES)}
            line = "  ".join(f"{p[:11]} {v['df1']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]"
                             for p, v in pairs[k].items() if v["n"])
            print(f"  {k:10} vs {ref:9} {line}")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "best": best, "decision": decision, "paired": pairs}, indent=1),
                                   encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
