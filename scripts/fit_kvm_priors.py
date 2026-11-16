"""Fit the KV admission priors (docs/phase2_spec.md §2.3) from Phase 1 perf-split traces.

KV admission reserves R̂_i = max(s_i + Ĝ_q(p, n_i), P̂0_p + Ĝ_q(p, n_i)) per admitted agent,
where Ĝ_q(p, n) is the q-quantile of (final peak − size after step n) over tasks of profile
p still running after n requests. Sizes are exact slot cell counts:
size after request j = prompt_n + cache_n + predicted_n − 1 (the last sampled token is never
decoded), and size after step 0 is the first prompt's token count P0.

Only tasks with status ok/forced count, and only from runs with no overflow failure: in a run
where agents died of "Context size has been exceeded" the survivors stop before their true
peak, which would bias Ĝ low. The default sources are the spec's (E1 at 131k, E2 c64k,
E2b c32k K=8); any other run passed with --runs goes through the same filter.

    uv run --no-sync python -m scripts.fit_kvm_priors [--runs 'results/e1_concurrency/*/rep*' ...]
    uv run --no-sync python -m scripts.fit_kvm_priors --check 'results/g0_shakedown/*/rep*'

Writes configs/kv_priors.json: {profile: {P0, G: {q: [per step n = 0, 1, ...]}, steps_mean,
duty, ...}} plus a "_meta" entry (sources, exclusions, settings); readers skip keys that
start with "_".
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path
from typing import Any

DEFAULT_RUNS = ["results/e1_concurrency/*/rep*", "results/e2_budget/c64k/rep*", "results/e2_budget_k/c32k_k8/rep*"]
QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
KEEP = ("ok", "forced")
# Statuses that mean the run's pool overflowed (or the gate refused a context): the whole
# run is excluded, not just those tasks.
OVERFLOW = ("ctx_overflow", "kv_infeasible")
MIN_N = 10   # below this many tasks at a step, the previous step's value is carried forward


def quantile(xs: list[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default)."""
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def task_sizes(row: dict[str, Any]) -> tuple[int, list[int]] | None:
    """(P0, [size after request 1, 2, ...]) from one tasks.jsonl row, or None if a step failed."""
    steps = row["trace"]["steps"]
    if not steps or not all(s["llm"].get("ok") for s in steps):
        return None
    first = steps[0]["llm"]
    p0 = int(first["prompt_n"]) + int(first["cache_n"])
    sizes = [int(s["llm"]["prompt_n"]) + int(s["llm"]["cache_n"]) + int(s["llm"]["predicted_n"]) - 1 for s in steps]
    return p0, sizes


def growth(p0: int, sizes: list[int]) -> list[int]:
    """Remaining growth peak − size after step n, for n = 0 .. len(sizes) − 1 (the agent is
    still running after n requests only while n < len(sizes))."""
    peak = max([p0, *sizes])
    return [peak - s for s in [p0, *sizes[:-1]]]


def run_ok(run: Path) -> tuple[bool, str]:
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    bad = {k: v for k, v in s.get("statuses", {}).items() if k in OVERFLOW}
    return (not bad, f"overflow statuses {bad}" if bad else "")


def load_tasks(runs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rows of the accepted runs (status ok/forced) and a list of exclusions with reasons."""
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for run in runs:
        if not (run / "tasks.jsonl").exists() or not (run / "summary.json").exists():
            excluded.append({"run": str(run), "reason": "incomplete run"})
            continue
        ok, why = run_ok(run)
        if not ok:
            excluded.append({"run": str(run), "reason": why})
            continue
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        with open(run / "tasks.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["status"] in KEEP:
                    r["_run"], r["_delay_scale"] = str(run), float(meta.get("delay_scale", 1.0))
                    rows.append(r)
    return rows, excluded


def fit_profile(rows: list[dict[str, Any]], quantiles: tuple[float, ...] = QUANTILES,
                min_n: int = MIN_N) -> dict[str, Any]:
    p0s: list[int] = []
    peaks: list[int] = []
    by_step: dict[int, list[int]] = {}
    n_req: list[int] = []
    duty: list[float] = []
    llm_s: list[float] = []
    tool_s: list[float] = []
    for r in rows:
        sz = task_sizes(r)
        if sz is None:
            continue
        p0, sizes = sz
        p0s.append(p0)
        peaks.append(max([p0, *sizes]))
        n_req.append(len(sizes))
        for n, g in enumerate(growth(p0, sizes)):
            by_step.setdefault(n, []).append(g)
        steps = r["trace"]["steps"]
        busy = sum(float(s["llm"].get("latency_s") or 0.0) for s in steps)
        llm_s.append(busy)
        scale = r["_delay_scale"]
        tool_s.append(sum(float(s.get("tool_delay_s") or 0.0) for s in steps) / scale if scale > 0 else 0.0)
        if r.get("service_s"):
            duty.append(min(1.0, busy / float(r["service_s"])))
    if not p0s:
        return {}
    G: dict[str, list[int]] = {}
    for q in quantiles:
        vals: list[int] = []
        for n in range(max(by_step) + 1):
            xs = by_step.get(n, [])
            if len(xs) >= min_n or not vals:
                vals.append(math.ceil(quantile(xs, q)) if xs else 0)
            else:
                vals.append(vals[-1])
        G[str(q)] = vals
    return {
        "P0": round(statistics.median(p0s)), "P0_max": max(p0s),
        "G": G,
        "n_at_step": [len(by_step.get(n, [])) for n in range(max(by_step) + 1)],
        "peak": {str(q): math.ceil(quantile(peaks, q)) for q in quantiles},
        "steps_mean": round(statistics.mean(n_req), 3), "steps_max": max(n_req),
        "duty": round(statistics.mean(duty), 4) if duty else None,
        # duty at another delay scale x: llm_s / (llm_s + x * tool_s_unit) (tool time at scale 1)
        "llm_s_mean": round(statistics.mean(llm_s), 3),
        "tool_s_unit_mean": round(statistics.mean(tool_s), 3),
        "n_tasks": len(p0s),
    }


def fit(rows: list[dict[str, Any]], quantiles: tuple[float, ...] = QUANTILES,
        min_n: int = MIN_N) -> dict[str, dict[str, Any]]:
    profiles = sorted({r["profile"] for r in rows})
    out = {p: fit_profile([r for r in rows if r["profile"] == p], quantiles, min_n) for p in profiles}
    return {p: v for p, v in out.items() if v}


def g_at(prior: dict[str, Any], q: float, n: int) -> int:
    """Ĝ_q(p, n); steps past the fitted range use the last value."""
    vals = prior["G"][str(q)]
    return vals[min(n, len(vals) - 1)]


def coverage(priors: dict[str, Any], rows: list[dict[str, Any]], q: float = 0.9) -> dict[str, Any]:
    """Share of (task, step) pairs whose true peak is within size_n + Ĝ_q(p, n): should be ≈ q
    if the fit transfers (spec §2.3: checked against the G0 dev traces)."""
    out: dict[str, Any] = {}
    for p in sorted({r["profile"] for r in rows}):
        if p not in priors:
            continue
        hit = tot = hit0 = tot0 = 0
        for r in rows:
            if r["profile"] != p or (sz := task_sizes(r)) is None:
                continue
            p0, sizes = sz
            peak = max([p0, *sizes])
            for n, size in enumerate([p0, *sizes[:-1]]):
                ok = peak <= size + g_at(priors[p], q, n)
                hit, tot = hit + ok, tot + 1
                if n == 0:
                    hit0, tot0 = hit0 + ok, tot0 + 1
        out[p] = {"pairs": tot, "covered": round(hit / tot, 4) if tot else None,
                  "covered_step0": round(hit0 / tot0, 4) if tot0 else None}
    return out


def expand(patterns: list[str]) -> list[Path]:
    return sorted({Path(p) for pat in patterns for p in glob.glob(pat)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="run directory globs")
    ap.add_argument("--out", default="configs/kv_priors.json")
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--check", nargs="+", help="run globs to check the existing priors against (no fit)")
    ap.add_argument("--q", type=float, default=0.9, help="quantile reported by --check")
    args = ap.parse_args()

    if args.check:
        priors = json.loads(Path(args.out).read_text(encoding="utf-8"))
        rows, excluded = load_tasks(expand(args.check))
        print(json.dumps({"coverage": coverage(priors, rows, args.q), "excluded": excluded}, indent=1))
        return

    runs = expand(args.runs)
    rows, excluded = load_tasks(runs)
    if not rows:
        raise SystemExit(f"no usable tasks in {args.runs}")
    priors: dict[str, Any] = fit(rows, min_n=args.min_n)
    scales = sorted({r["_delay_scale"] for r in rows})
    priors["_meta"] = {
        "generated_by": "scripts/fit_kvm_priors.py", "sources": sorted({r["_run"] for r in rows}),
        "excluded": excluded, "statuses_kept": list(KEEP), "quantiles": list(QUANTILES), "min_n": args.min_n,
        "delay_scales": scales,
        "size_rule": "after request j: prompt_n + cache_n + predicted_n - 1; step 0: first prompt tokens",
        "G_rule": "q-quantile of (peak - size after step n) over tasks with more than n requests",
    }
    priors["_meta"]["coverage_in_sample_q0.9"] = coverage(priors, rows, 0.9)
    Path(args.out).write_text(json.dumps(priors, indent=1) + "\n", encoding="utf-8", newline="\n")
    for p, v in priors.items():
        if not p.startswith("_"):
            print(f"{p}: n={v['n_tasks']} P0={v['P0']} G0.9={v['G']['0.9']} steps={v['steps_mean']} "
                  f"duty={v['duty']} peak0.9={v['peak']['0.9']}")
    print(f"excluded: {excluded or 'none'} -> {args.out}")


if __name__ == "__main__":
    main()
