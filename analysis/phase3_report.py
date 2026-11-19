"""Tables for the Phase 3 (TRIM) report (P1-P5), from results/runs.csv (run analysis.aggregate
first) and the run directories. Selection, deltas, tests and verdicts are analysis.phase2_report's,
unchanged, so a Phase 3 number and a Phase 2 number are computed the same way.

    uv run --no-sync python -m analysis.phase3_report [--root results] [--out docs/phase3/tables.md]

Sweeps (docs/phase3_spec.md §4):
- p3_select: selector choice on perf (p1_long_200), 64k ×1, B1 K=8, 2 reps, every selector.
- p3_quality: quality at ample memory on dev (dev_long_200), 64k ×1, B1 K=8, 2 reps.
- p3_tune: B1's K retuned per trim mode at 16k ×4 and 8k ×1 (perf, 1 rep per K).
- p3_dev: dev, 3 reps: B1 at its retuned K and ours (Phase 2 knobs, refit priors), × {oracle,
  relevance:2}, at 16k ×4 and 8k ×1. The untrimmed cells are Phase 2's p2_main runs (same split
  and knobs). Ours' points without `_sf` ran before the shrink-fast fix (KVConfig.shrink_fast,
  2026-11-19); the `_sf` points rerun them with it. The fix was made after seeing those runs, so
  every row that uses it is post hoc.

Metric notes:
- Recompute = re-prefilled tokens per task from the traces, leaving out request pairs whose second
  request carries a compacted prompt (analysis.aggregate.recompute_split): KV a compaction removes
  is not lost KV. Trim-lost = cells removed or re-prefilled at those pairs: the removed text, plus
  the whole compacted prompt when the slot was given up before the compaction ran. The manager's
  own recompute_tokens counts both, so it is not used for TRIM runs.
- Peak context = per task, the largest slot content (prompt + cached + generated) over its
  requests. KV held per tool wait = the slot's cells when a search or lookup starts, after the
  eager compaction if one ran; only measured at ample memory (64k), where no slot is evicted.
"""
from __future__ import annotations

import argparse
import statistics
from functools import cache
from pathlib import Path
from typing import Any

from analysis import phase2_report as p2
from analysis import stats
from analysis.phase2_report import Cell, Run, budget, ci_cell, delta, f1_tost, fmt, pval, sig, steady_ci, table, verdict

SELECT, QUALITY, TUNE, DEV = "p3_select", "p3_quality", "p3_tune", "p3_dev"
P3 = (SELECT, QUALITY, TUNE, DEV)
POINTS = ((16384, 4.0), (8192, 1.0))
SEL_ORDER = ("none", "oracle", "referenced", "relevance:2", "relevance:3", "drop")
REALISTIC = ("referenced", "relevance:2", "relevance:3")
SELECT_F1_WITHIN = 0.03   # spec §4 step 1, fixed before any run
DEV_MODES = ("oracle", "relevance:2")
SHORT = {"none": "", "oracle": "oracle", "relevance:2": "rel2", "relevance:3": "rel3", "referenced": "ref",
         "drop": "drop"}
TOOLS = ("search", "lookup")
EXTRA = ("trim_req", "trim_prefill", "trim_lost", "prefill", "kept", "sf_kept", "restore_bound", "max_agents",
         "gate_wait", "peak", "peak_long", "held", "forced", "calls", "sf_recall", "trim_lat")


# --- loading ---------------------------------------------------------------------------

def label(r: Run) -> str:
    """B1 / ours, + trim mode, (naive) for ours' TRIM runs before the shrink-fast fix."""
    s = r["cfg"]
    if r["trim"] != "none":
        s += "+" + SHORT.get(r["trim"], r["trim"])
        if r["cfg"] != "B1" and not r["fixed"]:
            s += " (naive)"
    return s


@cache
def trace_metrics(run: str) -> dict[str, float | None]:
    """Per-run means from the traces: peak context per task (all, react_long), KV held when a tool
    call starts, forced-finish share, agent LLM calls per task, supporting-fact recall of what the
    agent saw, and the eager compaction requests' latency."""
    path = Path(run)
    if not (path / "tasks.jsonl").exists():
        return {}
    peaks, peaks_long, held, calls, sfr, lat = [], [], [], [], [], []
    forced = n = 0
    for t in stats.load_tasks(path):
        n += 1
        forced += t["status"] == "forced"
        if t.get("sf_recall") is not None:
            sfr.append(float(t["sf_recall"]))
        steps = t["trace"]["steps"]
        calls.append(sum(1 for s in steps if s.get("kind") != "trim" and s.get("llm")))
        pk = 0
        for i, s in enumerate(steps):
            llm = s.get("llm") or {}
            if not llm.get("ok"):
                continue
            cells = llm["prompt_n"] + llm["cache_n"] + llm["predicted_n"]
            pk = max(pk, cells)
            if s.get("kind") == "trim":
                lat.append(float(llm["latency_s"]))
                continue
            if s.get("action") in TOOLS:
                nxt = steps[i + 1] if i + 1 < len(steps) else None
                last = llm
                if nxt and nxt.get("kind") == "trim" and nxt.get("step") == s.get("step") and (nxt.get("llm") or {}).get("ok"):
                    last = nxt["llm"]
                held.append(last["prompt_n"] + last["cache_n"] + last["predicted_n"] - 1)
        peaks.append(pk)
        if t.get("profile") == "react_long":
            peaks_long.append(pk)

    def mean(xs: list[float]) -> float | None:
        return statistics.mean(xs) if xs else None

    return {"peak": mean(peaks), "peak_long": mean(peaks_long), "held": mean(held),
            "forced": forced / n if n else None, "calls": mean(calls), "sf_recall": mean(sfr),
            "trim_lat": statistics.median(lat) if lat else None}


def extend(r: Run, traces: bool) -> Run:
    """Phase 3 fields on top of phase2_report.derive's."""
    n = p2.num(r, "n_tasks") or 0
    r["trim"] = r.get("trim") or "none"
    r["fixed"] = r["point"].endswith("_sf")
    r["label"] = label(r)
    r["trim_req"] = p2.num(r, "trim_requests_per_task")
    r["trim_prefill"] = p2.num(r, "trim_prefill_per_task")
    tl = p2.num(r, "trim_lost_tokens")
    r["trim_lost"] = tl / n if tl is not None and n else None
    r["prefill"] = p2.num(r, "prefill_per_task")
    r["kept"] = p2.num(r, "trim_kept_char_share")
    r["sf_kept"] = p2.num(r, "trim_sf_kept_share")
    has_mgr = p2.num(r, "mgr_give_ups") is not None
    rb = p2.num(r, "mgr_violations_restore_bound")
    r["restore_bound"] = rb if rb is not None else (0.0 if has_mgr else None)
    r["max_agents"] = p2.num(r, "mgr_max_agents") if has_mgr else p2.num(r, "kvx_agents_max")
    r["gate_wait"] = p2.num(r, "kvx_gate_wait_mean_s") if has_mgr else None
    if traces:
        r.update(trace_metrics(str(r["dir"])))
    return r


def load(root: Path) -> list[Run]:
    """Phase 3 runs, Phase 2's dev main runs (the untrimmed cells) and Phase 2's perf tuning runs
    (for B1's untrimmed pick). Trace metrics only where a table uses them."""
    runs, _ = p2.load_runs(root, sweeps=(*p2.TUNING, p2.MAIN, *P3))
    return [extend(r, traces=r["sweep"] in (SELECT, QUALITY, DEV, p2.MAIN)) for r in runs]


def cell(runs: list[Run]) -> Cell:
    c = p2.make_cell(runs)
    for m in EXTRA:
        vals = [r[m] for r in c["runs"] if r.get(m) is not None]
        c[m] = statistics.mean(vals) if vals else None
        c[m + "_lo"], c[m + "_hi"] = (min(vals), max(vals)) if vals else (None, None)
    for k in ("trim", "fixed", "label"):
        c[k] = c["runs"][0][k]
    return c


def cells_by(runs: list[Run], sweep: str) -> dict[str, Cell]:
    """One cell per point of a sweep, keyed by point name."""
    by: dict[str, list[Run]] = {}
    for r in runs:
        if r["sweep"] == sweep:
            by.setdefault(r["point"], []).append(r)
    return {pt: cell(rs) for pt, rs in by.items()}


def by_trim(cs: dict[str, Cell]) -> dict[str, Cell]:
    return {c["trim"]: c for c in cs.values()}


# --- formatting ------------------------------------------------------------------------

def pct(c: Cell | None, m: str, digits: int = 1) -> str:
    return fmt(c, m, digits, 100)


def rel_to(c: Cell, base: Cell | None, m: str) -> str:
    """The cell's mean, with its change vs base in %."""
    if c.get(m) is None:
        return "–"
    s = f"{c[m]:,.0f}"
    if base is not None and base is not c and base.get(m):
        s += f" ({100 * (c[m] - base[m]) / base[m]:+.0f}%)"
    return s


def tost_dir(q: dict[str, Any] | None) -> str:
    """phase2_report.tost_cell, plus the direction when the 90% CI excludes 0 (↑ better, ↓ worse),
    since TRIM can move F1 out of the margin upwards."""
    s = p2.tost_cell(q)
    if q and q.get("equivalent") is not None:
        lo, hi = q["ci90"]
        if lo > 0:
            s += " ↑"
        elif hi < 0:
            s += " ↓"
    return s


def dirs(c: Cell) -> list[Path]:
    return [r["dir"] for r in c["runs"]]


# --- P1: selector choice (perf) --------------------------------------------------------

def chosen(cs: dict[str, Cell]) -> str | None:
    """Spec §4 step 1: among the realistic selectors whose mean F1 is within 0.03 of none's, the
    smallest mean peak context; if none qualifies, the realistic selector with the highest F1."""
    none = cs.get("none")
    real = [cs[m] for m in REALISTIC if m in cs and cs[m].get("peak") is not None]
    if none is None or not real:
        return None
    ok = [c for c in real if c["f1"] >= none["f1"] - SELECT_F1_WITHIN]
    return (min(ok, key=lambda c: c["peak"]) if ok else max(real, key=lambda c: c["f1"]))["trim"]


SEL_HEAD = ["selector", "reps", "F1", "EM", "ΔF1 vs none [90% CI] (n q)", "forced %", "LLM calls/task",
            "peak ctx/task (Δ)", "react_long peak (Δ)", "KV held per tool wait (Δ)", "obs chars kept %",
            "SF kept %", "trim req/task", "trim prefill tok/task", "trim req p50 s"]


def selector_rows(cs: dict[str, Cell], pick: str | None = None) -> list[list[Any]]:
    none = cs.get("none")
    rows = []
    for m in SEL_ORDER:
        c = cs.get(m)
        if c is None:
            continue
        q = f1_tost(dirs(none), dirs(c)) if none is not None and c is not none else None
        rows.append([m + (" ★" if m == pick else ""), c["n"], fmt(c, "f1", 3), fmt(c, "em", 3), tost_dir(q),
                     pct(c, "forced"), fmt(c, "calls", 2), rel_to(c, none, "peak"), rel_to(c, none, "peak_long"),
                     rel_to(c, none, "held"), "–" if m == "none" else pct(c, "kept"),
                     "–" if m == "none" else pct(c, "sf_kept", 0), fmt(c, "trim_req", 2), fmt(c, "trim_prefill", 0),
                     fmt(c, "trim_lat", 2)])
    return rows


def null_row(c: Cell | None) -> list[list[Any]]:
    """none rep1 vs rep0, paired the same way: the run-to-run F1 noise floor."""
    if c is None or c["n"] < 2:
        return []
    a, b = [c["runs"][0]["dir"]], [c["runs"][1]["dir"]]
    return [["none rep1 vs rep0 (null)", "", "", "", tost_dir(f1_tost(a, b))] + [""] * (len(SEL_HEAD) - 5)]


def p1(runs: list[Run]) -> str:
    cs = by_trim(cells_by(runs, SELECT))
    if not cs:
        return "_P1: no p3_select runs yet._"
    pick = chosen(cs)
    cap = ("**P1. Selector choice (perf split, p1_long_200; 64k ×1, B1 K=8, so nothing is evicted; 2 reps).** "
           "Every TRIM run is eager: after the Thought/Action that follows an observation, one prefill-only "
           "request rewrites that observation to the kept sentences, before the tool call starts. "
           f"★ = the pre-registered choice: among referenced, relevance:2 and relevance:3, those with mean F1 "
           f"within {SELECT_F1_WITHIN} of none's, the smallest mean peak context. ΔF1 = paired per-question F1 "
           "vs none over the questions answered in every run (TOST margin ±0.05; ↑/↓ = the 90% CI excludes 0). "
           "Peak ctx = per task, the largest slot content over its requests; react_long = the 70% of tasks with "
           "10 paragraphs per search. KV held per tool wait = the slot's cells when a search or lookup starts "
           "(after the compaction). Δ = change vs none. Obs chars / SF kept = share of observation characters "
           "and of supporting-fact sentences the selector kept. Trim requests are the compaction requests; "
           "p50 s = their median latency.")
    return "\n\n".join([cap, table(SEL_HEAD, selector_rows(cs, pick) + null_row(cs.get("none")))])


# --- P2: quality on dev ----------------------------------------------------------------

def p2_quality(runs: list[Run]) -> str:
    cs = by_trim(cells_by(runs, QUALITY))
    if not cs:
        return "_P2: no p3_quality runs yet._"
    cap = ("**P2. Answer quality at ample memory (dev split, dev_long_200; 64k ×1, B1 K=8, 2 reps).** Same "
           "columns as P1. This is Q2 of the spec: what trimming costs in quality, on the split the "
           "throughput comparison uses, with the selector fixed on perf. Quality verdict: paired F1 TOST vs "
           "none, margin ±0.05 (↑ = significantly higher).")
    return "\n\n".join([cap, table(SEL_HEAD, selector_rows(cs) + null_row(cs.get("none")))])


# --- P3: B1 retune ---------------------------------------------------------------------

def p3_tune(runs: list[Run]) -> str:
    tune = [r for r in runs if r["sweep"] == TUNE]
    if not tune:
        return "_P3: no p3_tune runs yet._"
    untrimmed = p2.picks(p2.tuning_cells(runs), "B1")
    cap = ("**P3. B1's fixed K retuned under TRIM (perf split, 1 rep per K; selection runs, not results).** "
           "The Phase 2 rule: the K with the highest overall goodput among those completing ≥ 95% (★), else "
           "the highest completion (★⁻). Each grid has one K more than Phase 2's, added before any run, "
           "because TRIM lowers per-agent KV. Untrimmed = Phase 2's perf pick at the same point (T1).")
    rows = []
    for b in POINTS:
        pu, _ = untrimmed.get(b, (None, None))
        if pu:
            rows.append([budget(*b), "none (Phase 2)", f"K={pu['k']} ★", pct(pu, "completion"), fmt(pu, "goodput", 2),
                         f"{fmt(pu, 'purges', 0)} / {fmt(pu, 'overflows', 0)}", fmt(pu, "f1", 3)])
        for m in DEV_MODES:
            cs = sorted((cell([r]) for r in tune if (r["ctx"], r["delay"]) == b and r["trim"] == m),
                        key=lambda c: c["k"])
            pk = p2.pick(cs)
            for c in cs:
                mark = (" " + p2.star(pk)) if pk is not None and c["k"] == pk["k"] else ""
                rows.append([budget(*b), m, f"K={c['k']}{mark}", pct(c, "completion"), fmt(c, "goodput", 2),
                             f"{fmt(c, 'purges', 0)} / {fmt(c, 'overflows', 0)}", fmt(c, "f1", 3)])
    return "\n\n".join([cap, table(["point", "trim", "K", "completion %", "goodput", "purges / overflows", "F1"],
                                   rows)])


# --- P4: main dev comparison -----------------------------------------------------------

def main_cells(runs: list[Run]) -> dict[tuple[int, float], dict[str, Cell]]:
    """Per point, the dev cells by label: Phase 2's B1 pick, B1 at the K TRIM's tuning picked (when
    it differs), and ours, untrimmed (p2_main); then p3_dev's cells."""
    untrimmed = p2.picks(p2.tuning_cells(runs), "B1")
    tuned = {}
    for r in runs:
        if r["sweep"] == DEV and r["cfg"] == "B1":
            tuned.setdefault((r["ctx"], r["delay"]), set()).add(r["k"])
    out: dict[tuple[int, float], dict[str, Cell]] = {}
    for b in POINTS:
        pu, _ = untrimmed.get(b, (None, None))
        ks = {pu["k"]} if pu else set()
        ks |= tuned.get(b, set())
        sel: dict[str, list[Run]] = {}
        for r in runs:
            if (r["ctx"], r["delay"]) != b or r["split"] != "dev":
                continue
            if r["sweep"] == p2.MAIN and (r["cfg"] == "ours" or (r["cfg"] == "B1" and r["k"] in ks)):
                key = r["cfg"] if r["cfg"] == "ours" else f"B1 K={r['k']}"
                sel.setdefault(key, []).append(r)
            elif r["sweep"] == DEV:
                sel.setdefault(r["label"], []).append(r)
        cs = {k: cell(v) for k, v in sel.items()}
        if pu:
            for k, c in cs.items():
                c["star"] = c["cfg"] == "B1" and c["trim"] == "none" and c["k"] == pu["k"]
        out[b] = cs
    return out


ROW_ORDER = ("B1", "ours", "B1+oracle", "B1+rel2", "ours+oracle (naive)", "ours+rel2 (naive)", "ours+oracle",
             "ours+rel2")
MAIN_HEAD = ["config", "knob", "reps", "completion %", "failures", "goodput", "F1", "agents (max)",
             "give-ups (off/drop/erase)", "recompute tok/task", "trim-lost tok/task", "prefill tok/task",
             "trim req/task", "gate wait s", "restore-bound viol.", "purges / overflows"]


def row_key(name: str) -> tuple[int, str]:
    base = "B1" if name.startswith("B1 K=") else name
    return (ROW_ORDER.index(base) if base in ROW_ORDER else 99, name)


def main_row(name: str, c: Cell) -> list[Any]:
    label = name + (" ★" if c.get("star") else "")
    return [label, p2.knob(c), c["n"], pct(c, "completion"), p2.fails(c), fmt(c, "goodput", 2), fmt(c, "f1", 3),
            f"{fmt(c, 'agents')} ({fmt(c, 'max_agents', 0)})", p2.gives(c), fmt(c, "recompute", 0),
            fmt(c, "trim_lost", 0), fmt(c, "prefill", 0), fmt(c, "trim_req", 2), fmt(c, "gate_wait", 2),
            fmt(c, "restore_bound", 0), f"{fmt(c, 'purges', 0)} / {fmt(c, 'overflows', 0)}"]


def p4(runs: list[Run]) -> str:
    cells = main_cells(runs)
    if not any(c["trim"] != "none" for cs in cells.values() for c in cs.values()):
        return "_P4: no p3_dev runs yet._"
    cap = ("**P4. Main comparison with TRIM (dev split, dev_long_200, 3 reps).** Untrimmed rows are Phase 2's "
           "dev runs (p2_main): B1 at Phase 2's perf pick (★) and at the K TRIM's retune picked, and ours. B1+X = "
           "B1 at the K P3 picked for trim mode X; ours+X = ours with its Phase 2 knobs (ω = 1, k = K_max) and "
           "priors refit on trimmed runs. (naive) = ours' first TRIM runs, before the shrink-fast fix: there the "
           "compaction request waited in the gate queue like any other, and its own slot could be given up "
           "while it waited. The fixed rows (post hoc) dispatch a request that only shrinks its own resident "
           "slot at once. Agents = time-mean agents in flight (max = the manager's peak count). "
           + p2.GIVE_NOTE + " Recompute = re-prefilled tokens per task from the traces, without the "
           "compaction pairs; trim-lost = tokens removed or re-prefilled at a compaction (module docstring). "
           "Prefill tok/task = every prompt token the server computed. Restore-bound viol. = restores whose "
           "image exceeded the gate's reservation (the image is dropped and re-prefilled instead).")
    parts = [cap]
    for b in POINTS:
        cs = cells.get(b) or {}
        rows = [main_row(n, cs[n]) for n in sorted(cs, key=row_key)]
        parts.append(f"*{budget(*b)}*\n\n" + table(MAIN_HEAD, rows))
    return "\n\n".join(parts)


# --- P5: effects -----------------------------------------------------------------------

def effects(cs: dict[str, Cell]) -> list[tuple[str, str, str]]:
    """(effect, base label, new label) for one point, in report order."""
    b1_star = next((n for n, c in cs.items() if c.get("star")), None)
    out = []
    for m in ("oracle", "rel2"):
        b1t = f"B1+{m}"
        k_t = cs[b1t]["k"] if b1t in cs else None
        same_k = f"B1 K={k_t}" if k_t is not None else None
        if same_k in cs and same_k != b1_star:
            out.append((f"TRIM for B1, same K ({m})", same_k, b1t))
        if b1_star:
            out.append((f"TRIM for B1, K retuned ({m})", b1_star, b1t))
        out.append((f"TRIM alone vs manager alone ({m})", "ours", b1t))
        out.append((f"TRIM for ours, naive ({m})", "ours", f"ours+{m} (naive)"))
        out.append((f"TRIM for ours, fixed ({m})", "ours", f"ours+{m}"))
        out.append((f"shrink-fast fix ({m})", f"ours+{m} (naive)", f"ours+{m}"))
        out.append((f"ours vs B1, both with TRIM ({m})", b1t, f"ours+{m}"))
    if b1_star:
        out.append(("ours vs B1, untrimmed (Phase 2, T3)", b1_star, "ours"))
    return out


def p5(runs: list[Run], n_boot: int) -> str:
    cells = main_cells(runs)
    if not any(c["trim"] != "none" for cs in cells.values() for c in cs.values()):
        return "_P5: no p3_dev runs yet._"
    cap = ("**P5. Effects with TRIM (dev, 3 reps a side).** new − base: Δ overall goodput (tasks/min and %), "
           "Δ completion (pp), Welch p on per-run overall goodput (n runs per side is small, so p is only "
           "indicative), steady Δ goodput with its hierarchical bootstrap 95% CI, and paired per-question ΔF1 "
           "over the questions answered in every run on both sides (TOST ±0.05; ↑/↓ = the 90% CI excludes 0). "
           "Verdict: S only if Welch p < 0.05 and the steady CI excludes 0, else n.e. Rows with the shrink-fast "
           "fix are post hoc (the fix was made after the naive runs).")
    head = ["effect", "cells", "Δ goodput", "Δ %", "Δ completion pp", "Welch p (n)", "steady Δ goodput [95% CI]",
            "ΔF1 answered [90% CI] (n q)", "verdict"]
    parts = [cap]
    for b in POINTS:
        cs = cells.get(b) or {}
        rows = []
        for name, a, n in effects(cs):
            ca, cb = cs.get(a), cs.get(n)
            if ca is None or cb is None:
                continue
            d = delta(ca, cb)
            ci = steady_ci(ca, cb, n_boot)
            q = f1_tost(dirs(ca), dirs(cb))
            rows.append([name, f"{n} ({p2.knob(cb)}) − {a} ({p2.knob(ca)})", sig(d["goodput"], 2),
                         sig(d["goodput_pct"]), sig(d["completion_pp"]), pval(d), ci_cell(ci), tost_dir(q),
                         verdict(d, ci)])
        parts.append(f"*{budget(*b)}*\n\n" + table(head, rows))
    return "\n\n".join(parts)


# --- main ------------------------------------------------------------------------------

def notes(runs: list[Run]) -> list[str]:
    out = []
    sel = by_trim(cells_by(runs, SELECT))
    pick = chosen(sel)
    if pick and pick != "relevance:2":
        out.append(f"The selection rule now picks {pick}, not relevance:2 (the selector the dev runs used).")
    dev = [r for r in runs if r["sweep"] == DEV]
    naive_viol = [r for r in dev if r["cfg"] != "B1" and not r["fixed"] and (r["restore_bound"] or 0) > 0]
    fixed_viol = [r for r in dev if r["cfg"] != "B1" and r["fixed"] and (r["restore_bound"] or 0) > 0]
    if naive_viol:
        out.append(f"Restore-bound violations in {len(naive_viol)} naive ours+TRIM runs (a slot saved before "
                   "its compaction ran restores an uncompacted image larger than the reservation; the manager "
                   "drops the image and the prompt is re-prefilled). No task failed from it.")
    out.append(f"Restore-bound violations in fixed runs: {len(fixed_viol)}.")
    for r in dev:
        if r["completion"] is not None and r["completion"] < 1:
            out.append(f"{r['point']} {r['rep']}: completion {100 * r['completion']:.1f}%.")
    out.append("No Phase 3 run uses the held-out test split; every TRIM number is dev (or perf for selection).")
    return out


def report(root: Path, n_boot: int = 2000) -> str:
    runs = load(root)
    intro = ("Generated by `analysis/phase3_report.py` from `results/runs.csv` and the run directories; do not "
             "edit by hand. Goodput = EM-correct tasks per minute over the whole run (failed tasks count as "
             "wrong); completion = answered / n. Rules and tests are Phase 2's (`analysis/phase2_report.py`). "
             "Spec: `docs/phase3_spec.md`.")
    parts = ["# Phase 3 tables (TRIM)", intro, "## P1", p1(runs), "## P2", p2_quality(runs), "## P3", p3_tune(runs),
             "## P4", p4(runs), "## P5", p5(runs, n_boot),
             "## Data notes", "\n".join(f"- {n}" for n in notes(runs))]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="docs/phase3/tables.md")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replicates for the steady CI (0 = off)")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report(Path(args.root), args.boot), encoding="utf-8", newline="\n")
    print(f"tables -> {out}")


if __name__ == "__main__":
    main()
