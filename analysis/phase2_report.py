"""Tables for the Phase 2 report (T1-T6) and a data check, from results/runs.csv (run
analysis.aggregate first) and the run directories. Every table is built from whatever runs exist:
a table whose sweep has no runs yet becomes a one-line note, so the script can run at any point of
the schedule and simply picks up new runs.

    uv run --no-sync python -m analysis.phase2_report [--root results] [--out docs/phase2/tables.md]

Metric rules (fixed before the main runs, docs/phase2_spec.md §8.2):
- Headline: overall goodput (EM-correct tasks per minute over the whole run; failed and
  kv_infeasible tasks count as wrong) together with completion (answered / n). Steady rates are
  secondary and never the headline: a run that fails tasks frees KV, so its steady window looks
  faster than a run that finishes everything.
- Knob choice on the perf split (`pick`): the knob with the highest overall goodput among those
  completing >= 95%; if none does, the one with the highest completion. B1 also has `top`, the
  knob with the highest goodput whatever it completes, since the main table shows both.
- Means over reps with (min-max). Welch t-test on per-run overall goodput (1-3 runs per side,
  so p is a rough guide), the hierarchical bootstrap CI of steady goodput from analysis.stats
  when both sides have >= 2 runs with a steady window, and paired per-question F1 TOST vs B1
  (margin 0.05) over the questions answered in every run on both sides.
- Tuning runs (perf split) only justify the picks: no delta is computed from them. The dev 2x2
  uses the knob the perf rule chose, never a pick re-made on dev.
- One estimator per metric across all runs: recompute tokens from the traces, give-ups from the
  manager's per-task counters (not its eviction-op counts).

Sweeps: perf tuning (e2_budget, e2_budget_k, p0_baselines, p0x4_tune, p2_tune), dev main
(p2_main, plus c8k_d1_b2_k2 from p2_ablate), ablations and the 131k check (p2_ablate), dev 2x2
(p2_2x2). g0_shakedown is validation, never a result. The first adm_retain points of p2_tune
(c16k_d4_admret_w*, before the active-gate fix) are excluded; their fix is c16k_d4_admret_v2_w*.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import Counter
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

from analysis import aggregate, stats

TUNING = ("e2_budget", "e2_budget_k", "p0_baselines", "p0x4_tune", "p2_tune")
MAIN, ABLATE, TWO = "p2_main", "p2_ablate", "p2_2x2"
REPORTED = (MAIN, ABLATE, TWO)
# Main-table cells that ran with the ablations: B2 at 8k x1 and B1's perf top-goodput K at 16k x1.
MAIN_EXTRA = {(ABLATE, "c8k_d1_b2_k2"), (ABLATE, "c16k_d1_b1_k5")}
EXCLUDE = (("p2_tune", re.compile(r"_admret_w")),)  # adm_retain before the active-gate fix
SPLIT = {"p1_long_200": "perf", "dev_long_200": "dev", "test_long_500": "test"}
MAIN_CELLS = ((16384, 4.0), (8192, 1.0), (16384, 1.0), (32768, 4.0))
HEAD = (16384, 4.0)
FIXED_K = ("B1", "B2", "B3", "unpinned", "gated_k", "managed_k")
CFG_ORDER = ("B1", "B2", "B3", "gated_k", "managed_k", "adm_retain", "ours", "ours_h0")
COMPLETION_MIN = 0.95
F1_MARGIN = 0.05
HOST_BYTES_PER_TOKEN = 57_360   # B2's matched host tier h·c with h = 1 (configs/sweeps/p0x4_tune.yaml)
METRICS = ("completion", "goodput", "throughput", "steady_thr", "steady_good", "em", "f1", "agents",
           "give_ups", "give_offload", "give_drop", "give_erase", "ops_offload", "ops_drop", "ops_erase",
           "recompute", "op_share", "purges", "overflows", "violations")
EFFECTS = (("lifecycle at fixed K", "gated_k", "managed_k"),
           ("lifecycle at KV admission", "adm_retain", "ours"),
           ("admission, llama.cpp lifecycle", "gated_k", "adm_retain"),
           ("admission, managed lifecycle", "managed_k", "ours"))

GIVE_NOTE = ("Give-ups = give-up events (one per task and idle period, so a drop then an erase is one), "
             "with the victims by action off/drop/erase summed from the per-task manager counters "
             "(tasks.jsonl); – = no manager in the run.")
RECOMPUTE_NOTE = ("Recompute = re-prefilled tokens per task: for each pair of consecutive requests of a task, "
                  "the cells held after the first minus the next one's cache_n, from the traces. One "
                  "estimator for every run (it equals the manager's own count wherever there is one).")
SMALL_N = "n runs per side is small, so p is only indicative"

Run = dict[str, Any]
Cell = dict[str, Any]


# --- loading ---------------------------------------------------------------------------

def num(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def excluded(row: dict[str, Any]) -> bool:
    return any(row["sweep"] == s and rx.search(row["point"]) for s, rx in EXCLUDE)


def config(row: dict[str, Any]) -> str:
    """The configuration a run belongs to, from its flags rather than its name, so a B1 point of
    a Phase 1 sweep and an observe run of p2_main land in the same class."""
    pol = row.get("policy") or "observe"
    if pol != "observe":
        return pol
    if row.get("unified") == "False":
        return "B3"
    if row.get("pin") == "False":
        return "B2" if (num(row, "cache_ram") or 0) > 0 else "unpinned"
    return "B1"


def failures(row: dict[str, Any]) -> dict[str, int]:
    out = {}
    for k in row:
        if k.startswith("status_") and k[7:] not in stats.ANSWERED and (n := num(row, k)):
            out[k[7:]] = int(n)
    return out


def mean_agents(run: Path, makespan: float | None) -> float | None:
    """Time-mean agents in flight, Σ service time / makespan: Phase 1 runs have no kv_tick, and
    on Phase 2 runs this matches the ticks' mean to ±0.02."""
    path = run / "tasks.jsonl"
    if not makespan or not path.exists():
        return None
    return sum(float(t["service_s"]) for t in stats.load_tasks(run)) / makespan


def task_counts(row: dict[str, Any], run: Path) -> dict[str, float | None]:
    """Recompute tokens and give-ups by action (analysis.aggregate's recompute_tokens and
    gave_up_* columns). A runs.csv written before those columns existed has no such header,
    so they are computed from the run's tasks.jsonl instead of silently reading as missing."""
    cols = ("recompute_tokens", "gave_up_offloaded", "gave_up_dropped", "gave_up_erased")
    if "recompute_tokens" in row:
        return {c: num(row, c) for c in cols}
    if not (run / "tasks.jsonl").exists():
        return {}
    tasks = stats.load_tasks(run)
    return {"recompute_tokens": aggregate.recompute_tokens(tasks), **aggregate.give_up_row(tasks)}


def derive(row: dict[str, Any], root: Path) -> Run:
    """One run: the raw row plus the report's metrics (None where the run cannot say)."""
    r: Run = dict(row)
    n = num(row, "n_tasks") or 0
    mk = num(row, "makespan_s")
    r["cfg"] = cfg = config(row)
    r["split"] = SPLIT.get(row.get("workload", ""), row.get("workload"))
    r["ctx"] = int(num(row, "ctx") or 0)
    r["delay"] = num(row, "delay_scale")
    r["k"] = int(num(row, "k") or 0)
    r["omega"] = None if cfg in FIXED_K or row.get("admission") in ("fixed", "") else num(row, "omega")
    r["host_mib"] = int(num(row, "cache_ram") or 0)
    r["dir"] = root / row["sweep"] / row["point"] / row["rep"]
    r["completion"] = (num(row, "completed") or 0) / n if n else None
    r["goodput"] = num(row, "goodput_tasks_per_min")
    r["throughput"] = num(row, "throughput_tasks_per_min")
    r["steady_thr"] = num(row, "steady_throughput_per_min")
    r["steady_good"] = num(row, "steady_goodput_per_min")
    r["em"], r["f1"] = num(row, "em"), num(row, "f1")
    r["agents"] = num(row, "kvx_agents_mean")
    if r["agents"] is None:
        r["agents"] = mean_agents(r["dir"], mk)
    tk = task_counts(row, r["dir"])
    has_mgr = num(row, "mgr_give_ups") is not None
    r["give_ups"] = num(row, "mgr_give_ups")
    for a, col in (("offload", "offloaded"), ("drop", "dropped"), ("erase", "erased")):
        r[f"give_{a}"] = tk.get(f"gave_up_{col}")
        v = num(row, f"mgr_evictions_{a}")
        r[f"ops_{a}"] = v if v is not None else (0.0 if has_mgr else None)
    rec = tk.get("recompute_tokens")
    rec = rec if rec is not None else num(row, "mgr_recompute_tokens")   # the same definition
    r["recompute"] = rec / n if rec is not None and n else None
    op = num(row, "mgr_op_server_s")
    r["op_share"] = op / mk if op is not None and mk else None
    r["purges"] = num(row, "slog_purges") if num(row, "slog_purges") is not None else num(row, "mgr_server_purges")
    r["overflows"] = (num(row, "slog_overflow_events") if num(row, "slog_overflow_events") is not None
                      else num(row, "mgr_overflows"))
    r["violations"] = sum(num(row, k) or 0 for k in ("kvx_n_violations", "slog_purge_violations",
                                                      "slog_overflow_violations"))
    r["violation_run"] = row.get("violation_run") == "True"
    r["failures"] = failures(row)
    return r


def load_runs(root: Path, sweeps: tuple[str, ...] = (*TUNING, *REPORTED)) -> tuple[list[Run], list[dict[str, Any]]]:
    """Runs of the report's sweeps, and the rows dropped by EXCLUDE (listed in the data notes)."""
    path = root / "runs.csv"
    if not path.exists():
        return [], []
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["sweep"] in sweeps]
    return [derive(r, root) for r in rows if not excluded(r)], [r for r in rows if excluded(r)]


# --- cells, the tuning rule and deltas -------------------------------------------------

def make_cell(runs: list[Run]) -> Cell:
    """Mean, min and max over reps of every metric; the runs stay attached for the tests."""
    runs = sorted(runs, key=lambda r: r["rep"])
    c: Cell = {k: runs[0].get(k) for k in ("cfg", "split", "ctx", "delay", "k", "omega", "host_mib", "sweep",
                                           "point")}
    c["runs"], c["n"] = runs, len(runs)
    for m in METRICS:
        vals = [r[m] for r in runs if r.get(m) is not None]
        c[m] = statistics.mean(vals) if vals else None
        c[m + "_lo"], c[m + "_hi"] = (min(vals), max(vals)) if vals else (None, None)
    fails: Counter[str] = Counter()
    for r in runs:
        fails.update(r["failures"])
    c["failures"] = {s: v / len(runs) for s, v in sorted(fails.items())}
    return c


def group(runs: list[Run], key: Callable[[Run], Any]) -> dict[Any, Cell]:
    by: dict[Any, list[Run]] = {}
    for r in runs:
        by.setdefault(key(r), []).append(r)
    return {k: make_cell(v) for k, v in by.items()}


def knob_key(r: Run) -> tuple[Any, ...]:
    """A tuning knob across sweeps: the same configuration in two sweeps is one cell with reps."""
    return (r["cfg"], r["ctx"], r["delay"], r["k"] if r["cfg"] in FIXED_K else None, r["omega"], r["host_mib"])


def pick(cands: list[Cell], floor: float = COMPLETION_MIN) -> Cell | None:
    """The tuning rule: highest overall goodput among knobs completing >= floor; if none does,
    the highest completion (goodput breaks ties)."""
    ok = [c for c in cands if (c["completion"] or 0) >= floor]
    if ok:
        return max(ok, key=lambda c: c["goodput"] or 0)
    return max(cands, key=lambda c: (c["completion"] or 0, c["goodput"] or 0), default=None)


def meets_floor(c: Cell | None, floor: float = COMPLETION_MIN) -> bool:
    return c is not None and (c["completion"] or 0) >= floor


def star(c: Cell | None) -> str:
    """★ for a pick that meets the completion floor, ★⁻ for the fallback (highest completion when
    no knob reaches it), so a reader never takes a fallback pick as meeting the floor."""
    return "★" if meets_floor(c) else "★⁻"


def top(cands: list[Cell]) -> Cell | None:
    """The knob with the highest overall goodput, however many tasks it drops."""
    return max(cands, key=lambda c: c["goodput"] or 0, default=None)


def rel(base: float | None, new: float | None) -> float | None:
    """Relative change of new over base, in %."""
    return 100 * (new - base) / base if base and new is not None else None


def delta(base: Cell, new: Cell) -> dict[str, Any]:
    """new vs base: Δ overall goodput (abs, %), Δ completion (pp), Welch p on per-run goodput."""
    ga = [r["goodput"] for r in base["runs"] if r["goodput"] is not None]
    gb = [r["goodput"] for r in new["runs"] if r["goodput"] is not None]
    w = stats.welch(ga, gb) if ga and gb else {"p": None}
    return {"goodput": (new["goodput"] - base["goodput"]) if None not in (new["goodput"], base["goodput"]) else None,
            "goodput_pct": rel(base["goodput"], new["goodput"]),
            "completion_pp": (100 * (new["completion"] - base["completion"])
                              if None not in (new["completion"], base["completion"]) else None),
            "p": w.get("p"), "n": (len(ga), len(gb))}


@cache
def _window(run: str) -> tuple[list[float], float] | None:
    return stats.steady_window(stats.load_tasks(Path(run)))


def steady_ci(base: Cell, new: Cell, n_boot: int) -> dict[str, Any] | None:
    """Bootstrap CI of the steady-goodput Δ, when both sides have >= 2 runs with a steady window.
    With one run a side, resampling runs always returns that run, so the CI would hold only
    task-level noise and look tighter than the evidence is."""
    wa = [w for r in base["runs"] if (r["dir"] / "tasks.jsonl").exists() and (w := _window(str(r["dir"])))]
    wb = [w for r in new["runs"] if (r["dir"] / "tasks.jsonl").exists() and (w := _window(str(r["dir"])))]
    if len(wa) < 2 or len(wb) < 2 or n_boot <= 0:
        return None
    return stats.bootstrap_delta(wa, wb, "goodput", n_boot)


def f1_tost(base_dirs: list[Path], new_dirs: list[Path], margin: float = F1_MARGIN,
            answered: bool = True) -> dict[str, Any] | None:
    """Paired per-question F1 TOST, new − base. By default over the questions answered in every
    run on both sides: the test is meant to show the manager leaves answer quality alone, and
    with failures scored 0 a baseline that fails tasks (or a B1 null whose reps complete
    differently) would come out "not equivalent" for a completion gap that T3 already reports.
    answered=False is that failures-as-0 version, kept as a separate column."""
    base_dirs = [d for d in base_dirs if (d / "tasks.jsonl").exists()]
    new_dirs = [d for d in new_dirs if (d / "tasks.jsonl").exists()]
    if not base_dirs or not new_dirs:
        return None
    only = stats.answered_in_all(base_dirs + new_dirs) if answered else None
    return stats.tost_paired(stats.paired_diffs(stats.per_question(base_dirs, only=only),
                                                stats.per_question(new_dirs, only=only)), margin)


# --- formatting ------------------------------------------------------------------------

def fmt(c: Cell | None, m: str, digits: int = 1, scale: float = 1.0) -> str:
    """Mean, with (min–max) when the cell has reps that differ."""
    if c is None or c.get(m) is None:
        return "–"
    s = f"{c[m] * scale:.{digits}f}"
    lo, hi = c.get(m + "_lo"), c.get(m + "_hi")
    if c["n"] > 1 and lo is not None and hi is not None and f"{lo * scale:.{digits}f}" != f"{hi * scale:.{digits}f}":
        s += f" ({lo * scale:.{digits}f}–{hi * scale:.{digits}f})"
    return s


def sig(v: float | None, digits: int = 1, plus: bool = True) -> str:
    if v is None:
        return "–"
    return f"{v:+.{digits}f}" if plus else f"{v:.{digits}f}"


def pval(d: dict[str, Any]) -> str:
    if d.get("p") is None:
        return f"– (n {d['n'][0]}/{d['n'][1]})"
    return f"{d['p']:.2f} (n {d['n'][0]}/{d['n'][1]})"


def by_action(c: Cell, prefix: str) -> str | None:
    if all(c.get(f"{prefix}_{a}") is None for a in ("offload", "drop", "erase")):
        return None
    return "/".join(f"{c.get(f'{prefix}_{a}') or 0:.0f}" for a in ("offload", "drop", "erase"))


def gives(c: Cell) -> str:
    """Give-ups as "total (offload/drop/erase)": the total is one event per task and idle period
    (a drop then an erase is one), so it can be below the sum by action."""
    acts = by_action(c, "give")
    if acts is None:
        return "–"
    return f"{c['give_ups']:.0f} ({acts})" if c.get("give_ups") is not None else f"– ({acts})"


def ops(c: Cell) -> str:
    """Eviction ops the manager started by action: wasted saves and stale clean-ups included."""
    return by_action(c, "ops") or "–"


def fails(c: Cell) -> str:
    return ", ".join(f"{s} {v:g}" for s, v in c["failures"].items()) or "0"


def knob(c: Cell) -> str:
    if c["cfg"] == "B2":
        return f"K={c['k']}, {c['host_mib']} MiB"
    if c["cfg"] in FIXED_K:
        return f"K={c['k']}"
    return f"ω={c['omega']:g}" if c.get("omega") is not None else ""


def cell_name(c: Cell) -> str:
    """The point name without its c<ctx>k_d<delay>_ prefix (ablation presets, 2x2 cells)."""
    return re.sub(r"^c\d+k_d[\d.]+_", "", c["point"] or "")


def budget(ctx: int, delay: float | None) -> str:
    """The project's names: 8k = 8,192 tokens, ..., but 131k = 131,072 (not 128k)."""
    name = "131k" if ctx == 131072 else f"{ctx // 1024}k"
    return f"{name} ×{delay:g}"


def table(head: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def order(c: Cell) -> tuple[int, int, float]:
    i = CFG_ORDER.index(c["cfg"]) if c["cfg"] in CFG_ORDER else len(CFG_ORDER)
    return (i, c["k"] if c["cfg"] in FIXED_K else 0, c.get("omega") or 0)


# --- tables ----------------------------------------------------------------------------

def tuning_cells(runs: list[Run]) -> dict[tuple[Any, ...], Cell]:
    return group([r for r in runs if r["sweep"] in TUNING and r["split"] == "perf"], knob_key)


def by_budget(cells: dict[Any, Cell], cfg: str) -> dict[tuple[int, float], list[Cell]]:
    out: dict[tuple[int, float], list[Cell]] = {}
    for c in cells.values():
        if c["cfg"] == cfg:
            out.setdefault((c["ctx"], c["delay"]), []).append(c)
    return out


def picks(cells: dict[Any, Cell], cfg: str) -> dict[tuple[int, float], tuple[Cell | None, Cell | None]]:
    """Per budget × delay: (pick, top) of one configuration's tuning knobs."""
    return {b: (pick(cs), top(cs)) for b, cs in by_budget(cells, cfg).items()}


def t1(cells: dict[Any, Cell]) -> str:
    rows = []
    for b in sorted({(c["ctx"], c["delay"]) for c in cells.values() if c["cfg"] in ("B1", "B2", "B3")},
                    key=lambda x: (x[1], x[0])):
        for cfg in ("B1", "B2", "B3"):
            cs = sorted(by_budget(cells, cfg).get(b, []), key=lambda c: (c["host_mib"], c["k"]))
            p, t = pick(cs), top(cs)
            for c in cs:
                mark = (star(c) if c is p else "") + ("†" if c is t and cfg == "B1" else "")
                rows.append([budget(*b), cfg, knob(c), c["n"], fmt(c, "completion", 1, 100), fmt(c, "goodput", 2),
                             fmt(c, "throughput", 1), mark])
    if not rows:
        return "_No perf-split baseline runs yet._"
    return ("**T1. Baseline tuning (perf split, p1_long_200; selection runs, not results).** Overall "
            "goodput and throughput are over the whole run. ★ = the tuning rule's pick (highest goodput "
            "among K completing ≥ 95%); ★⁻ = no K reaches 95%, so the pick is the highest-completion K; "
            "† = B1's highest-goodput K whatever it completes. B2 lists its host "
            "tier (`--cache-ram`); matched h·c is 448/896/1792 MiB at 8k/16k/32k.\n\n"
            + table(["budget", "config", "knob", "reps", "completion %", "goodput", "throughput", "pick"], rows))


def t2(cells: dict[Any, Cell]) -> str:
    ours = by_budget(cells, "ours")
    b1, b2 = picks(cells, "B1"), picks(cells, "B2")
    head = ["budget", "config", "knob", "completion %", "goodput", "throughput", "steady thr.", "agents",
            "give-ups (off/drop/erase)", "recompute tok/task", "op time %"]
    rows = []
    for b in sorted(ours, key=lambda x: (x[1], x[0])):
        cs = sorted(ours[b], key=lambda c: c["omega"] or 0)
        p = pick(cs)
        ref, b1_top = b1.get(b, (None, None))
        b2_pick = b2.get(b, (None, None))[0]
        shown = [(c, star(c) if c is p else "") for c in cs]
        shown += [(c, m) for c, m in ((ref, star(ref) + ("†" if ref is b1_top else "")), (b1_top, "†"),
                                      (b2_pick, star(b2_pick))) if c is not None]
        seen = set()
        for c, mark in shown:
            if id(c) in seen:
                continue
            seen.add(id(c))
            rows.append([budget(*b), f"{c['cfg']}{mark}", knob(c), fmt(c, "completion", 1, 100), fmt(c, "goodput", 2),
                         fmt(c, "throughput"), fmt(c, "steady_thr"), fmt(c, "agents"), gives(c),
                         fmt(c, "recompute", 0), fmt(c, "op_share", 1, 100)])
    if not rows:
        return "_No ours tuning runs yet._"
    return ("**T2. Ours ω tuning (perf split, 1 rep per knob) next to the tuned baselines at the same "
            "point.** Selection runs, not results: the knobs were chosen on these runs, so no deltas are "
            "drawn from them (the dev reps in T3 are the comparison). ★ = the tuning rule's pick per "
            "configuration (★⁻ = fallback, no knob completes ≥ 95%), † = B1's highest-goodput K. "
            "Agents = time-mean agents in flight. " + GIVE_NOTE + " " + RECOMPUTE_NOTE + " Op time = slot-op "
            "server time / makespan.\n\n"
            + table(head, rows))


def main_cells(runs: list[Run]) -> dict[tuple[int, float], list[Cell]]:
    """The dev main table's cells per budget × delay: p2_main plus MAIN_EXTRA."""
    rs = [r for r in runs if r["sweep"] == MAIN or (r["sweep"], r["point"]) in MAIN_EXTRA]
    out: dict[tuple[int, float], list[Cell]] = {}
    for c in group(rs, lambda r: (r["ctx"], r["delay"], r["point"])).values():
        out.setdefault((c["ctx"], c["delay"]), []).append(c)
    return {b: sorted(cs, key=order) for b, cs in out.items()}


METRIC_HEAD = ["config", "knob", "reps", "completion %", "failures", "goodput", "throughput", "steady thr.", "EM",
               "F1", "agents", "give-ups (off/drop/erase)", "recompute tok/task", "op time %", "purges / overflows"]


def metric_row(c: Cell, label: str | None = None) -> list[Any]:
    return [label or c["cfg"], knob(c), c["n"], fmt(c, "completion", 1, 100), fails(c), fmt(c, "goodput", 2),
            fmt(c, "throughput"), fmt(c, "steady_thr"), fmt(c, "em", 3), fmt(c, "f1", 3), fmt(c, "agents"), gives(c),
            fmt(c, "recompute", 0), fmt(c, "op_share", 1, 100), f"{fmt(c, 'purges', 0)} / {fmt(c, 'overflows', 0)}"]


def b1_mark(c: Cell, tuned: dict[tuple[int, float], tuple[Cell | None, Cell | None]]) -> str:
    """★/† when a B1 K of the main table is the perf tuning's pick / top K at that point."""
    p, t = tuned.get((c["ctx"], c["delay"]), (None, None))
    return (star(p) if p and p["k"] == c["k"] else "") + ("†" if t and t["k"] == c["k"] else "")


def comparisons(cs: list[Cell], n_boot: int) -> list[list[Any]]:
    """Ours and ours_h0 vs every baseline of one point, then the B1 rep-vs-rep F1 null."""
    rows = []
    bases = [c for c in cs if c["cfg"] in ("B1", "B2")]
    for new in (c for c in cs if c["cfg"] in ("ours", "ours_h0")):
        for base in bases:
            d = delta(base, new)
            ci = steady_ci(base, new, n_boot)
            a, b = [r["dir"] for r in base["runs"]], [r["dir"] for r in new["runs"]]
            q, q0 = (f1_tost(a, b), f1_tost(a, b, answered=False)) if base["cfg"] == "B1" else (None, None)
            rows.append([new["cfg"], f"{base['cfg']} {knob(base)}", sig(d["goodput_pct"]), sig(d["completion_pp"]),
                         pval(d), f"{ci['delta']:+.2f} [{ci['ci'][0]:+.2f}, {ci['ci'][1]:+.2f}]" if ci else "–",
                         tost_cell(q), tost_cell(q0, verdict=False)])
    for base in (c for c in bases if c["cfg"] == "B1" and c["n"] >= 2):
        a, b = [base["runs"][0]["dir"]], [base["runs"][1]["dir"]]
        rows.append([f"{base['cfg']} {knob(base)} {base['runs'][1]['rep']}", base["runs"][0]["rep"] + " (null)",
                     "", "", "", "", tost_cell(f1_tost(a, b)), tost_cell(f1_tost(a, b, answered=False), verdict=False)])
    return rows


def tost_cell(q: dict[str, Any] | None, verdict: bool = True) -> str:
    """Mean ΔF1 and its 90% CI, with the number of paired questions; the verdict only where the
    test is the pre-registered one (answered in every run)."""
    if not q or q.get("equivalent") is None:
        return "–"
    s = f"{q['mean']:+.3f} [{q['ci90'][0]:+.3f}, {q['ci90'][1]:+.3f}]"
    if verdict:
        s += f" {'eq.' if q['equivalent'] else 'not eq.'}"
    return s + f" (n {q['n']})"


def t3(runs: list[Run], tuned_b1: dict[tuple[int, float], tuple[Cell | None, Cell | None]], n_boot: int) -> str:
    cells = main_cells(runs)
    if not cells:
        return "_T3: no p2_main runs yet._"
    parts = [("**T3. Main comparison (dev split, dev_long_200).** Per point, mean over reps (min–max). "
             "B1 ★ = the perf tuning's pick (best K completing ≥ 95%; ★⁻ = no K reached 95% on perf, so it is "
             "the highest-completion K), † = its highest-goodput K. Failures are per run by status "
             "(kv_infeasible counts as a failure). Purges / overflows are from server.log. " + GIVE_NOTE + " "
             + RECOMPUTE_NOTE + " Deltas: Δ overall goodput % and Δ completion (pp) of ours vs each baseline; "
             "Welch p on per-run overall goodput, with n runs per side (" + SMALL_N + "). Steady Δ goodput "
             "(tasks/min) with its hierarchical bootstrap 95% CI, secondary, only when both sides have ≥ 2 runs. "
             "ΔF1 = paired per-question F1, ours − B1, mean and 90% CI over the n questions answered in every "
             f"run on both sides; equivalent if inside ±{F1_MARGIN} (TOST), and this is the quality verdict. "
             "ΔF1 failures = 0 pairs all questions with failed tasks scored 0, so it mixes in the completion gap "
             "(no verdict). The null row is B1 rep1 vs rep0, paired the same way.")]
    for b in sorted(cells, key=lambda x: MAIN_CELLS.index(x) if x in MAIN_CELLS else 99):
        cs = cells[b]
        rows = [metric_row(c, c["cfg"] + (b1_mark(c, tuned_b1) if c["cfg"] == "B1" else "")) for c in cs]
        parts.append(f"*{budget(*b)}*\n\n" + table(METRIC_HEAD, rows))
        comp = comparisons(cs, n_boot)
        if comp:
            parts.append(table(["config", "vs", "Δ goodput %", "Δ completion pp", "Welch p (n)",
                                "steady Δ goodput [95% CI]", "ΔF1 answered [90% CI] (n q)",
                                "ΔF1 failures = 0 (n q)"], comp))
    return "\n\n".join(parts)


def two_by_two(cells: dict[str, Cell | None]) -> list[list[Any]]:
    rows = []
    for name, a, b in EFFECTS:
        ca, cb = cells.get(a), cells.get(b)
        if ca is None or cb is None:
            rows.append([name, f"{b} − {a}", "–", "–", "–", "–", f"missing: {', '.join(x for x, c in ((a, ca), (b, cb)) if c is None)}"])
            continue
        d = delta(ca, cb)
        rows.append([name, f"{b} ({knob(cb)}) − {a} ({knob(ca)})", sig(d["goodput"], 2), sig(d["goodput_pct"]),
                     sig(d["completion_pp"]), pval(d), ""])
    return rows


TWO_CELLS = ("gated_k", "managed_k", "adm_retain")


def perf_2x2(cells: dict[Any, Cell]) -> dict[str, list[Cell]]:
    """The perf tuning's knobs at the headline point, per 2×2 cell (B1 and ours for reference)."""
    head = [c for c in cells.values() if (c["ctx"], c["delay"]) == HEAD]
    return {cfg: sorted((c for c in head if c["cfg"] == cfg), key=order) for cfg in ("B1", *TWO_CELLS, "ours")}


def knob_value(c: Cell) -> Any:
    """The tuned knob of a cell: K for the fixed-K cells, ω for KV admission."""
    return c["k"] if c["cfg"] in FIXED_K else c.get("omega")


# Knobs fixed from the perf tuning before the dev 2x2 ran, where the goodput rule's pick is a tie.
DEV_KNOB = {"managed_k": 8}
DEV_KNOB_WHY = {"managed_k": "on perf K=8 and K=10 tie on goodput (11.68 vs 11.69), and K=8 had the higher "
                             "throughput (26.3 vs 25.1 tasks/min), so K=8 was chosen before the dev runs."}


def dev_2x2(runs: list[Run], cells: dict[Any, Cell]) -> tuple[dict[str, Cell | None], list[str]]:
    """The dev 2×2 cells at the knob the perf tuning chose, never re-picked on dev: choosing on
    the evaluation split would bias the decomposition deltas. Notes for knobs that are missing
    on dev or that dev ran but the perf rule did not choose."""
    dev = [r for r in runs if r["sweep"] == TWO and (r["ctx"], r["delay"]) == HEAD]
    if not dev:
        return {}, []
    two = list(group(dev, lambda r: r["point"]).values())
    perf = perf_2x2(cells)
    chosen: dict[str, Cell | None] = {}
    notes = []
    for cfg in TWO_CELLS:
        p = pick(perf[cfg])
        if cfg in DEV_KNOB and p is not None:   # a tie the rule breaks by noise; decided on perf, before dev
            p = next((c for c in perf[cfg] if knob_value(c) == DEV_KNOB[cfg]), p)
            notes.append(f"Dev 2×2 {cfg}: {knob(p)} is used; {DEV_KNOB_WHY[cfg]}")
        have = [c for c in two if c["cfg"] == cfg]
        chosen[cfg] = next((c for c in have if p and knob_value(c) == knob_value(p)), None)
        if p is None and have:
            notes.append(f"Dev 2×2 {cfg} ran ({', '.join(knob(c) for c in have)}) but the perf tuning has no "
                         f"{cfg} pick, so it is not used.")
        elif p is not None and chosen[cfg] is None:
            notes.append(f"Dev 2×2 {cfg}: the perf tuning picks {knob(p)}, which p2_2x2 has not run"
                         + (f" (it has {', '.join(knob(c) for c in have)})" if have else "") + ".")
        elif p is not None:
            extra = [knob(c) for c in have if c is not chosen[cfg]]
            if extra:
                notes.append(f"Dev 2×2 {cfg}: also ran {', '.join(extra)}; only the perf pick {knob(p)} is used.")
    return chosen, notes


def t4(runs: list[Run], cells: dict[Any, Cell]) -> str:
    parts = [("**T4. 2×2 decomposition at 16k ×4.** Lifecycle effect = managed − llama.cpp lifecycle, at "
             "fixed K and at KV admission; admission effect = KV admission − fixed K, under each lifecycle. "
             "Δ goodput in correct tasks/min and %, Δ completion in pp. B1 (no gate) is shown for reference. "
             "Effects are reported only from the dev reps, at the knob the perf tuning chose (spec §8.1).")]
    cand = perf_2x2(cells)
    if any(cand[x] for x in TWO_CELLS):
        rows = []
        for cfg, cs in cand.items():
            p = pick(cs)
            for c in cs:
                rows.append([cfg + (star(c) if c is p else ""), knob(c), fmt(c, "completion", 1, 100),
                             fmt(c, "goodput", 2), fmt(c, "throughput"), fmt(c, "agents"), fmt(c, "purges", 0),
                             fmt(c, "recompute", 0), fmt(c, "op_share", 1, 100)])
        parts.append("*Selection runs (perf, 1 rep per knob), not results: ★ = the tuning rule's pick per cell "
                     "(★⁻ = fallback, nothing completes ≥ 95%). No effects are computed from these runs, since "
                     "the picks were made on them.*\n\n"
                     + table(["cell", "knob", "completion %", "goodput", "throughput", "agents", "purges",
                              "recompute tok/task", "op time %"], rows))
    else:
        parts.append("_No perf 2×2 tuning runs yet._")
    chosen, _ = dev_2x2(runs, cells)
    if chosen:
        main = main_cells(runs).get(HEAD, [])
        chosen["ours"] = next((c for c in main if c["cfg"] == "ours"), None)
        rows = [metric_row(c) for c in [*[c for c in main if c["cfg"] == "B1"],
                                        *(chosen[x] for x in (*TWO_CELLS, "ours"))] if c]
        parts.append("*Dev reps (p2_2x2 at the perf-chosen knobs; ours and B1 from p2_main). Welch p: "
                     + SMALL_N + ".*\n\n" + table(METRIC_HEAD, rows))
        parts.append(table(["effect", "cells", "Δ goodput", "Δ %", "Δ completion pp", "Welch p (n)", "note"],
                           two_by_two(chosen)))
    else:
        parts.append("_Dev 2×2 reps (p2_2x2): not run yet._")
    return "\n\n".join(parts)


def ablation_cells(runs: list[Run]) -> list[Cell]:
    """Ablation presets at 16k x4: p2_ablate's single runs plus the follow-up reps p2_2x2 ran for
    some of them (same point name, same dev workload), merged per point."""
    rs = [r for r in runs if (r["ctx"], r["delay"]) == HEAD and r["cfg"] not in ("B1", "B2")
          and ((r["sweep"] == ABLATE and (r["sweep"], r["point"]) not in MAIN_EXTRA)
               or (r["sweep"] == TWO and r["cfg"] not in TWO_CELLS))]
    return sorted(group(rs, lambda r: r["point"]).values(), key=cell_name)


def t5(runs: list[Run]) -> str:
    abl = ablation_cells(runs)
    if not abl:
        return "_T5: no ablation runs (p2_ablate at 16k ×4) yet._"
    main = main_cells(runs).get(HEAD, [])
    ours = next((c for c in main if c["cfg"] == "ours"), None)
    refs = [c for c in main if c["cfg"] in ("ours", "ours_h0")]
    rows = []
    for c in [*refs, *abl]:
        d = delta(ours, c) if ours else None
        rows.append([c["cfg"] + " (p2_main)" if c in refs else cell_name(c), c["cfg"], c["n"],
                     fmt(c, "completion", 1, 100), fmt(c, "goodput", 2),
                     sig(d["goodput_pct"]) if d and c is not ours else "", pval(d) if d and c is not ours else "",
                     gives(c), ops(c), fmt(c, "recompute", 0), fmt(c, "op_share", 1, 100), fmt(c, "violations", 0)])
    note = "" if ours else " Ours' dev mean (p2_main) is not in yet, so no deltas."
    return ("**T5. Ablations at 16k ×4 (dev, 1 rep each unless noted).** Each preset changes one rule of ours "
            "(same ω, k = K_max); Δ = overall goodput vs ours' dev mean, Welch p where both sides have ≥ 2 runs ("
            + SMALL_N + "). " + GIVE_NOTE + " Eviction ops = slot ops the manager started by action, including "
            "wasted saves and clean-ups of stale slots, so they exceed the give-ups. " + RECOMPUTE_NOTE
            + note + "\n\n" + table(["preset", "policy", "reps", "completion %", "goodput", "Δ vs ours %",
                                     "Welch p (n)", "give-ups (off/drop/erase)", "eviction ops off/drop/erase",
                                     "recompute tok/task", "op time %", "violations"], rows))


def t6(runs: list[Run]) -> str:
    regimes = [("32k ×4 (p2_main)", main_cells(runs).get((32768, 4.0), [])),
               ("131k ×1 (p2_ablate)", sorted(group([r for r in runs if r["sweep"] == ABLATE and r["ctx"] == 131072],
                                                    lambda r: r["point"]).values(), key=order))]
    parts = [("**T6. No-benefit regimes.** Pre-registered as small or no gain: the pool is big enough that "
              "fixed K rarely runs out. Δ vs the regime's B1 (lowest K shown); Welch p on per-run overall "
              "goodput (" + SMALL_N + "). " + GIVE_NOTE)]
    for name, cs in regimes:
        if not cs:
            parts.append(f"_{name}: not run yet._")
            continue
        base = next((c for c in cs if c["cfg"] == "B1"), None)
        rows = []
        for c in cs:
            d = delta(base, c) if base and c is not base else None
            rows.append([c["cfg"], knob(c), c["n"], fmt(c, "completion", 1, 100), fmt(c, "goodput", 2),
                         fmt(c, "throughput"), fmt(c, "agents"), gives(c), fmt(c, "op_share", 1, 100),
                         sig(d["goodput_pct"]) if d else "", pval(d) if d else ""])
        parts.append(f"*{name}*\n\n" + table(["config", "knob", "reps", "completion %", "goodput", "throughput",
                                               "agents", "give-ups (off/drop/erase)", "op time %", "Δ goodput vs B1 %",
                                               "Welch p (n)"], rows))
    return "\n\n".join(parts)


# --- data check ------------------------------------------------------------------------

def expected_reps(sweeps: tuple[str, ...] = (MAIN, ABLATE, TWO, "p2_tune")) -> dict[tuple[str, str], int]:
    """Points and reps each sweep config asks for. Several files may share a sweep name
    (p2_tune_v2 adds to p2_tune), so they are merged."""
    import yaml

    from scripts.sweep import points as sweep_points
    out = {}
    for path in sorted(Path("configs/sweeps").glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(spec, dict) and spec.get("name") in sweeps:
            for name, p in sweep_points(spec):
                out[(spec["name"], name)] = int(p["reps"])
    return out


def data_notes(runs: list[Run], dropped: list[dict[str, Any]], cells: dict[Any, Cell]) -> list[str]:
    notes = []
    if dropped:
        notes.append(f"Excluded {len(dropped)} run(s) by rule (adm_retain before the active-gate fix): "
                     + ", ".join(f"{r['sweep']}/{r['point']}" for r in dropped) + ".")
    for r in runs:
        if r["violation_run"] or r["violations"]:
            notes.append(f"Violation run: {r['sweep']}/{r['point']}/{r['rep']} ({r['violations']:.0f} violations).")
    for r in runs:   # B1/B2 failures are the expected cost of a fixed K and are in T3; flag the rest
        managed = r["cfg"] not in ("B1", "B2", "B3", "unpinned")
        ids = r.get("kv_infeasible_ids") or ""
        if (r["sweep"] in REPORTED or r["sweep"] == "p2_tune") and ((managed and r["failures"]) or ids):
            notes.append(f"Failures in {r['sweep']}/{r['point']}/{r['rep']} ({r['cfg']}): "
                         + ", ".join(f"{s} {n}" for s, n in r["failures"].items())
                         + (f" (kv_infeasible task ids: {ids})" if ids else "") + ".")
    have = Counter((r["sweep"], r["point"]) for r in runs)
    try:
        want = expected_reps()
    except Exception as e:  # noqa: BLE001 - the notes must not block the tables
        want = {}
        notes.append(f"Could not read the sweep configs ({e}).")
    pending = sorted(k for k, n in want.items() if have[k] == 0 and not excluded({"sweep": k[0], "point": k[1]}))
    short = sorted((k, have[k], n) for k, n in want.items() if 0 < have[k] < n)
    for (s, p), got, n in short:
        notes.append(f"Missing reps: {s}/{p} has {got} of {n}.")
    by_sweep = Counter(s for s, _ in pending)
    for s, n in sorted(by_sweep.items()):
        notes.append(f"Pending: {n} point(s) of {s} have no runs ("
                     + ", ".join(p for ss, p in pending if ss == s) + ").")
    for cfg in ("B1", "B2", "ours", *TWO_CELLS):
        for b, (p, _) in sorted(picks(cells, cfg).items(), key=lambda x: (x[0][1], x[0][0])):
            if p is not None and not meets_floor(p):
                notes.append(f"No {cfg} knob reaches {COMPLETION_MIN:.0%} completion at {budget(*b)} (perf); the pick "
                             f"({star(p)}) is the highest-completion knob, {knob(p)} at {100 * p['completion']:.1f}%.")
    notes += dev_2x2(runs, cells)[1]
    tuned = picks(cells, "B1")
    for b, cs in main_cells(runs).items():
        p, t = tuned.get(b, (None, None))
        ks = sorted(c["k"] for c in cs if c["cfg"] == "B1")
        want_k = sorted({c["k"] for c in (p, t) if c})
        if ks and want_k and ks != want_k:
            notes.append(f"B1 K in the main table at {budget(*b)} is {ks}; the perf tuning rule gives "
                         f"pick K={p['k'] if p else '–'} and top K={t['k'] if t else '–'}.")
        po = pick(by_budget(cells, "ours").get(b, []))
        for c in cs:
            if c["cfg"] == "ours" and po and c["omega"] != po["omega"]:
                notes.append(f"Ours ω in the main table at {budget(*b)} is {c['omega']:g}; perf tuning picks {po['omega']:g}.")
    for c in cells.values():
        if c["cfg"] == "B2" and c["host_mib"] != round(c["ctx"] * HOST_BYTES_PER_TOKEN / 2 ** 20):
            notes.append(f"B2 tuning at {budget(c['ctx'], c['delay'])} K={c['k']} used {c['host_mib']} MiB of host "
                         f"cache, not the matched h·c ({round(c['ctx'] * HOST_BYTES_PER_TOKEN / 2 ** 20)} MiB).")
    return notes


# --- main ------------------------------------------------------------------------------

CAPTIONS = {
    "f1_completion_goodput.png":
        "F1. Completion vs overall goodput per point (dev main, p2_main): small markers are reps, large "
        "markers the mean per configuration. B1 is coloured by role and labelled with its K: ★ = the perf "
        "tuning's pick (★⁻ = fallback, no K completes ≥ 95%), † = the perf top-goodput K, plain = neither. "
        "B1's K grid from the perf tuning is omitted: perf and dev are different question sets, so a perf "
        "line next to dev points would mislead.",
    "f2a_tuning.png":
        "F2a. Perf tuning runs at 16k ×4, 1 rep per knob (selection data, not results): overall goodput vs "
        "time-mean agents in flight for every knob of B1, gated_k, managed_k, adm_retain and ours. Circles = "
        "the knob run on dev (managed_k: K=8 by tie-break); bold numbers = llama.cpp purges (server.log); "
        "grey % = B1 completion.",
    "f2b_dev.png":
        "F2b. 2×2 at 16k ×4 on dev (3 reps per cell at the perf-chosen knobs; B1 and ours from p2_main): "
        "mean, min–max and reps. Grey = no gate, orange = llama.cpp lifecycle, green = managed; arrows = "
        "lifecycle effects with Welch p and the S / n.e. verdict.",
    "f2_decomposition.png": "F2a and F2b side by side (for the notes).",
    "f3_timeline_16k_x4.png":
        "F3. GPU KV over time at 16k ×4 (dev rep0): B1 at the perf-tuned K vs ours, stacked by state "
        "against -c (dashed), with llama.cpp purges (▼) and failed tasks (×) marked. Active KV is valued at "
        "each request's final token count (an upper bound while it runs), so B1's stack can exceed -c just "
        "before its overflows. Offloaded KV is off-GPU and drawn as a line, not stacked.",
    "f4_ablations.png":
        "F4. Ablations at 16k ×4 (dev): overall goodput per preset (dot = mean, bar = min–max, grey = reps, "
        "hollow = 1 run, n in the label) against ours' mean (line) and rep range (band) and B1 K=4's mean "
        "(dashed); text = Δ vs ours' mean.",
    "kv_states.png": "KV token·s by state per configuration at 16k ×4 (dev rep0); B2's breakdown is "
                     "approximate (hatched).",
    "timeline_<point>.png": "GPU KV over time by state for every configuration at 16k ×4 (dev rep0).",
}


def figure_captions() -> str:
    return "\n".join(f"- `{name}`: {text}" for name, text in CAPTIONS.items())


def report(root: Path, n_boot: int = 2000) -> str:
    runs, dropped = load_runs(root)
    cells = tuning_cells(runs)
    tuned_b1 = picks(cells, "B1")
    intro = ("Generated by `analysis/phase2_report.py` from `results/runs.csv` and the run directories; "
             "do not edit by hand. Goodput = EM-correct tasks per minute over the whole run (failed and "
             "kv_infeasible tasks count as wrong); completion = answered / n. Steady rates (10–90% "
             "completion window) are secondary, since failing tasks frees KV and inflates them.")
    parts = ["# Phase 2 tables", intro,
             "## T1", t1(cells), "## T2", t2(cells), "## T3", t3(runs, tuned_b1, n_boot), "## T4", t4(runs, cells),
             "## T5", t5(runs), "## T6", t6(runs), "## Figures", figure_captions(),
             "## Data notes", "\n".join(f"- {n}" for n in data_notes(runs, dropped, cells)) or "- none"]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="docs/phase2/tables.md")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replicates for the steady CI (0 = off)")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report(Path(args.root), args.boot), encoding="utf-8", newline="\n")
    print(f"tables -> {out}")


if __name__ == "__main__":
    main()
