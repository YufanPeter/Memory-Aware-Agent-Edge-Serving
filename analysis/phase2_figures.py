"""Figures for the Phase 2 report, from results/runs.csv (run analysis.aggregate first) and the run
directories. Each figure is drawn only when its sweep has runs, so the script can run at any
point of the schedule; captions live in analysis.phase2_report.CAPTIONS (written to tables.md).
Run selection, the tuning rule and the metrics are analysis.phase2_report's, so a figure and its
table never disagree.

    uv run --no-sync --with matplotlib python -m analysis.phase2_figures [--root results] [--out docs/figures/phase2]

  f1_completion_goodput.png  dev main: completion vs overall goodput, per point (reps + means)
  f2_decomposition.png       16k x4 2x2, two panels: (a) perf selection runs, goodput vs time-mean agents
                             in flight for every knob (tuning data; circles = knob run on dev); (b) dev
                             reps at those knobs (min-max), with the lifecycle effects (Welch p, verdict)
  f2a_tuning.png, f2b_dev.png  the two F2 panels alone, drawn at the size the report prints them
                             (F2b one column wide, F2a on the one-column appendix page)
  f3_timeline_16k_x4.png     dev rep0 at 16k x4: GPU KV by state over time, B1 (tuned K) vs ours, with
                             llama.cpp purges and failed tasks marked
  f4_ablations.png           dev ablations at 16k x4: goodput per preset (dots, min-max, n reps) vs ours'
                             rep range and B1
  kv_states.png              KV token·s by state per configuration at 16k x4 (dev rep0)
  timeline_<pt>.png          GPU KV over time by state, every configuration at 16k x4 (dev rep0)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analysis import phase2_report as R
from analysis.kv_timeline import (
    EXACT_STATES,
    GPU_STATES,
    STATES,
    approximate,
    exact_intervals,
    has_exact,
    intervals,
    load_events,
    timeline,
)

plt.rcParams.update({"figure.dpi": 150, "font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
W = 3.5   # single column of the two-column report
HOST_TIERS = ("offloaded", "host")   # not in the GPU pool: never stacked against -c
COLOR = {"active": "#2b6cb0", "idle_tool": "#dd6b20", "ready": "#d69e2e", "stale": "#a0aec0", "op": "#805ad5",
         "other": "#718096", "offloaded": "#38a169", "host": "#319795"}
CFG_STYLE = {   # (colour, marker, legend label); B1 by its role in the perf tuning, not by K order
    "B1_pick": ("#2b6cb0", "o", "B1★ (perf pick)"), "B1_top": ("#90cdf4", "o", "B1† (perf top goodput)"),
    "B1_other": ("#a0aec0", "o", "B1, other K"),
    "B2": ("#dd6b20", "D", "B2"), "ours": ("#2f855a", "s", "ours"), "ours_h0": ("#81e6d9", "s", "ours_h0"),
    "other": ("#718096", "^", "other")}
LLAMA, MANAGED, NOGATE = "#c05621", "#2f855a", "#718096"


def style_key(c: R.Cell, mark: str) -> str:
    """B1's colour comes from its role (R.b1_mark), so it means the same thing in every panel."""
    if c["cfg"] == "B1":
        return "B1_pick" if mark.startswith("★") else "B1_top" if "†" in mark else "B1_other"
    return c["cfg"] if c["cfg"] in CFG_STYLE else "other"


def fig_f1(runs: list[R.Run], out: Path) -> None:
    cells = R.main_cells(runs)
    pts = [b for b in R.MAIN_CELLS if b in cells] + sorted(b for b in cells if b not in R.MAIN_CELLS)
    if not pts:
        return
    tuned = R.picks(R.tuning_cells(runs), "B1")
    cols = 2 if len(pts) > 1 else 1
    rows = math.ceil(len(pts) / cols)
    # Constrained layout with the legend "outside": the legend gets its own band above the grid,
    # so it can never cover a panel title, whatever the number of entries.
    fig, axes = plt.subplots(rows, cols, figsize=(W, 0.75 + 1.45 * rows), squeeze=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)
    handles = {}
    for a, b in zip(axes.flat, pts):
        for c in cells[b]:
            mark = R.b1_mark(c, tuned) if c["cfg"] == "B1" else ""
            color, marker, label = CFG_STYLE[style_key(c, mark)]
            xs = [r["goodput"] for r in c["runs"] if r["goodput"] is not None]
            ys = [100 * r["completion"] for r in c["runs"] if r["completion"] is not None]
            a.scatter(xs, ys, s=9, color=color, marker=marker, alpha=0.55, lw=0)
            if c["goodput"] is not None:
                h = a.scatter([c["goodput"]], [100 * c["completion"]], s=30, color=color, marker=marker,
                              edgecolor="black", lw=0.5, zorder=3)
                handles.setdefault(label, h)
                if c["cfg"] == "B1":   # below the marker: the means near 100% sit on a row
                    a.annotate(f"K={c['k']}{mark}", (c["goodput"], 100 * c["completion"]), textcoords="offset points",
                               xytext=(0, -9), fontsize=5.5, color="#2d3748", ha="center", va="center")
        a.set_title(R.budget(*b), fontsize=7.5, pad=2)
        a.tick_params(labelsize=6.5)
        a.margins(x=0.12)
    lo = min((100 * r["completion"] for cs in cells.values() for c in cs for r in c["runs"]
              if r["completion"] is not None), default=0)
    for i, a in enumerate(axes.flat):   # one completion scale, so panels compare at a glance
        a.set_ylim(max(0, lo - 7), 104)
        if i >= len(pts):
            a.set_visible(False)
    fig.supxlabel("overall goodput (correct tasks/min); each panel has its own x scale", fontsize=6.5)
    fig.supylabel("completion (% of tasks answered)", fontsize=6.5)
    order = [s[2] for s in CFG_STYLE.values() if s[2] in handles]
    fig.legend([handles[x] for x in order], order, loc="outside upper center", ncol=3, frameon=False,
               fontsize=6, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
    fig.savefig(out / "f1_completion_goodput.png")
    plt.close(fig)


F2_LINES = (   # cfg, colour, line style, marker, filled, legend label
    ("B1", NOGATE, "--", "o", True, "B1: no gate, fixed $K$"),
    ("gated_k", LLAMA, "-", "o", True, "gated_k: llama.cpp lifecycle, fixed $K$"),
    ("managed_k", MANAGED, "-", "o", True, "managed_k: managed lifecycle, fixed $K$"),
    ("adm_retain", LLAMA, ":", "s", False, "adm_retain: llama.cpp lifecycle, KV admission"),
    ("ours", MANAGED, ":", "s", False, "ours: managed lifecycle, KV admission"))
PURGE_CFGS = ("gated_k", "adm_retain")   # llama.cpp's lifecycle under a gate: its purges are the story
F2_NOTE = 6.5   # pt: F2a/F2b are drawn at the size they are printed, so nothing goes below this


def f2_dev(runs: list[R.Run], tuning: dict) -> dict[str, R.Cell]:
    """F2's dev panel: the dev reps at the knobs fixed from the perf tuning (p2_2x2; B1 and ours
    from p2_main), never re-picked on dev. Empty until p2_2x2 has runs."""
    chosen, _ = R.dev_2x2(runs, tuning)
    if not any(chosen.values()):
        return {}
    main = R.main_cells(runs).get(R.HEAD, [])
    out = {cfg: c for cfg, c in chosen.items() if c}
    out.update({f"B1 K={c['k']}": c for c in sorted((c for c in main if c["cfg"] == "B1"), key=R.order)})
    if ours := next((c for c in main if c["cfg"] == "ours"), None):
        out["ours"] = ours
    return out


def verdict(base: R.Cell, new: R.Cell) -> str:
    """The report's decision rule: S iff Welch p < 0.05 and the steady bootstrap CI excludes 0."""
    p = R.delta(base, new)["p"]
    ci = R.steady_ci(base, new, 2000)
    if p is None or ci is None:
        return "n.e."
    lo, hi = ci["ci"]
    return "S" if p < 0.05 and (lo > 0 or hi < 0) else "n.e."


def knob_label(c: R.Cell) -> str:
    return f"$K$={c['k']}" if c["cfg"] in R.FIXED_K else f"$\\omega$={c['omega']:g}"


def f2_tuning_panel(a, perf: dict, dev: dict[str, R.Cell], fs: float) -> list:
    """Perf selection runs, every knob: goodput vs time-mean agents in flight. Circles mark the
    knob each cell ran on dev in the dev panel (B1: the rule's pick); for managed_k that is K=8,
    a tie-break outside the goodput rule (R.DEV_KNOB), which the panel says next to the circle."""
    a_lo = 6.6
    handles, off = [], []
    for cfg, color, ls, marker, filled, label in F2_LINES:
        cs = sorted((c for c in perf.get(cfg, []) if c["agents"] is not None and c["goodput"] is not None),
                    key=lambda c: c["agents"])
        if not cs:
            continue
        p = R.pick(cs)
        if cfg in dev:   # the knob the dev reps ran
            p = next((c for c in cs if R.knob_value(c) == R.knob_value(dev[cfg])), p)
        shown = [c for c in cs if c["goodput"] >= a_lo]
        (h,) = a.plot([c["agents"] for c in shown], [c["goodput"] for c in shown], ls=ls, marker=marker,
                      color=color, ms=4, lw=1.1, mfc=color if filled else "white", label=label, zorder=3)
        handles.append(h)
        if p is not None:
            a.plot([p["agents"]], [p["goodput"]], marker="o", mfc="none", mec="black", mew=0.8, ms=9, lw=0,
                   zorder=4)
            rule = R.pick(cs)
            if rule is not None and rule is not p:   # dev knob differs from the goodput rule's pick
                a.annotate(f"{knob_label(p)} by tie-break\n({p['goodput']:.2f} vs {rule['goodput']:.2f} at "
                           f"{knob_label(rule)})", (p["agents"], p["goodput"]), textcoords="offset points",
                           xytext=(0, 7), ha="center", va="bottom", fontsize=fs, color=color)
        for i, c in enumerate(shown):
            if cfg in PURGE_CFGS and c["purges"] is not None:
                # placed off the neighbouring lines: gated_k above (left of the first point, where
                # managed_k sits), adm_retain below, its last point to the right
                if cfg == "gated_k":
                    xy, ha, va = ((-6, 2), "right", "bottom") if i == 0 else ((6, 3), "left", "bottom")
                elif i == len(shown) - 1:
                    xy, ha, va = (6, 0), "left", "center"
                else:
                    xy, ha, va = ((5, -3), "left", "top") if i == 0 else ((-5, -3), "right", "top")
                a.annotate(f"{c['purges']:.0f}", (c["agents"], c["goodput"]), textcoords="offset points",
                           xytext=xy, ha=ha, va=va, fontsize=fs, color=color, fontweight="bold")
            elif cfg == "B1":
                a.annotate(f"{100 * c['completion']:.0f}%", (c["agents"], c["goodput"]), textcoords="offset points",
                           xytext=(0, -6), ha="center", va="top", fontsize=fs, color=color)
        for c in (c for c in cs if c["goodput"] < a_lo):   # a collapsed knob: on the edge, value in the note
            last = shown[-1] if shown else c
            a.plot([last["agents"], c["agents"]], [last["goodput"], a_lo], ls=ls, color=color, lw=1.1)
            a.plot([c["agents"]], [a_lo], marker="v", color=color, ms=4, clip_on=False, zorder=3)
            off.append(f"{cfg} $K$={c['k']}: {c['goodput']:.2f}, {100 * c['completion']:.0f}% done (off scale, ▼)")
        if cfg == "ours":
            for i, c in enumerate(cs):   # first one to the left (crowded at K≈5), the rest below
                a.annotate(f"$\\omega$={c['omega']:g}", (c["agents"], c["goodput"]), textcoords="offset points",
                           xytext=(-6, 0) if i == 0 else (0, -6), ha="right" if i == 0 else "center",
                           va="center" if i == 0 else "top", fontsize=fs, color=color)
    (pick_h,) = a.plot([], [], marker="o", mfc="none", mec="black", mew=0.8, ms=8, lw=0, label="knob run on dev")
    g_hi = max((c["goodput"] for cs in perf.values() for c in cs if c["goodput"] is not None), default=a_lo)
    a.set_ylim(a_lo, g_hi + 0.85)   # headroom for the tie-break note inside the axes
    a.set_xlim(3.5, 10.3)
    a.set_xlabel("agents in flight (time-mean; ≈ $K$ for fixed $K$ = 4, 6, 8, 10)", fontsize=7)
    a.set_ylabel("overall goodput (correct tasks/min)", fontsize=7)
    a.tick_params(labelsize=7)
    a.text(0.01, 0.02, "\n".join(["bold: llama.cpp purges (server.log); ours, managed_k: 0",
                                    "grey %: B1 completion (every gated configuration: 100%)", *off]),
           transform=a.transAxes, ha="left", va="bottom", fontsize=fs, color="#4a5568")
    return [*handles, pick_h]


def f2_dev_panel(d, dev: dict[str, R.Cell], fs: float) -> list:
    """Dev reps at the chosen knobs, laid out as the 2x2: mean, min-max bar, each rep; arrows are
    the two lifecycle effects with Welch p and the report's verdict."""
    # x spacing follows the tick-label widths ("managed_k", "adm_retain" are the long ones), so
    # neighbouring labels never touch at one column wide
    slots = [(n, i * 1.0) for i, n in enumerate(k for k in dev if k.startswith("B1"))]
    x0 = len(slots) * 1.0 + 0.35
    step = 1.5
    slots += [(n, x0 + j * step) for j, n in enumerate(("gated_k", "managed_k")) if n in dev]
    x1 = x0 + 2 * step + 0.2
    slots += [(n, x1 + j * 1.3) for j, n in enumerate(("adm_retain", "ours")) if n in dev]
    xpos = dict(slots)
    ticks = []
    for name, x in slots:
        c = dev[name]
        cfg = c["cfg"]
        color, _, marker, filled, _ = next(s[1:] for s in F2_LINES if s[0] == cfg)
        reps = [r["goodput"] for r in c["runs"] if r["goodput"] is not None]
        d.scatter([x + 0.22] * len(reps), reps, s=7, color=color, alpha=0.6, lw=0, zorder=2)
        d.errorbar([x], [c["goodput"]], yerr=[[c["goodput"] - c["goodput_lo"]], [c["goodput_hi"] - c["goodput"]]],
                   fmt=marker, color=color, ms=4.5, mfc=color if filled else "white", capsize=2, lw=0.9, zorder=3)
        ticks.append(f"{cfg}\n{knob_label(c)}")
        below = []
        if c["completion"] is not None and c["completion"] < 0.995:
            below.append(f"{100 * c['completion']:.1f}%\ndone")
        if cfg in PURGE_CFGS and c["purges"] is not None:
            below.append(f"{c['purges']:.0f}\npurges")
        if below:
            d.annotate("\n".join(below), (x, c["goodput_lo"]), textcoords="offset points", xytext=(0, -3),
                       ha="center", va="top", fontsize=fs, color=color, fontweight="bold" if cfg != "B1" else None)
    top = max(c["goodput_hi"] for c in dev.values())
    for _, base, new in R.EFFECTS[:2]:   # lifecycle effects: llama.cpp -> managed, same admission
        cb, cn = dev.get(base), dev.get(new)
        if cb and cn:
            dl = R.delta(cb, cn)
            y = top + 0.35
            d.annotate("", (xpos[new], y), (xpos[base], y),
                       arrowprops={"arrowstyle": "->", "color": "#2d3748", "lw": 0.7, "shrinkA": 0, "shrinkB": 0})
            d.text((xpos[base] + xpos[new]) / 2, y + 0.08,
                   f"{dl['goodput_pct']:+.1f}%, $p$={dl['p']:.2f}, {verdict(cb, cn)}",
                   ha="center", va="bottom", fontsize=fs, color="#2d3748")
    d.set_xticks([x for _, x in slots], ticks, fontsize=fs)
    d.tick_params(axis="x", length=0, pad=2)
    d.tick_params(axis="y", labelsize=7)
    d.grid(axis="x", visible=False)
    lo = min(c["goodput_lo"] for c in dev.values())
    d.set_ylim(lo - 1.6, top + 1.7)
    d.set_xlim(-0.5, slots[-1][1] + 0.55)
    d.set_ylabel("overall goodput (correct/min)", fontsize=7)
    for xa, xb, label in ((x0 - 0.6, x0 + step + 0.6, "fixed $K$"),
                          (x1 - 0.65, x1 + 1.3 + 0.5, "KV admission")):
        d.axvspan(xa, xb, color="#edf2f7", zorder=0, lw=0)
        d.text((xa + xb) / 2, top + 1.65, label, ha="center", va="top", fontsize=fs, color="#4a5568")
    return [Line2D([], [], color=c, marker="o", lw=0, ms=4, label=lab)
            for c, lab in ((NOGATE, "no gate"), (LLAMA, "llama.cpp lifecycle"), (MANAGED, "managed lifecycle"))]


def fig_f2(runs: list[R.Run], out: Path) -> None:
    """F2 as two figures drawn at their printed size (report: F2b in a column, F2a on the
    appendix page), plus the combined two-panel figure for the technical notes."""
    tuning = R.tuning_cells(runs)
    perf = R.perf_2x2(tuning)
    if not (perf["gated_k"] or perf["managed_k"]):
        return
    dev = f2_dev(runs, tuning)

    fig, a = plt.subplots(figsize=(6.2, 2.9), layout="constrained")
    hs = f2_tuning_panel(a, perf, dev, F2_NOTE)
    fig.legend(hs, [h.get_label() for h in hs], loc="outside upper center", ncol=2, frameon=False,
               fontsize=7, handlelength=2.2, columnspacing=1.5, borderaxespad=0.1)
    fig.savefig(out / "f2a_tuning.png", dpi=300)
    plt.close(fig)

    if dev:
        fig, d = plt.subplots(figsize=(W, 1.95), layout="constrained")   # colours in the report caption
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
        f2_dev_panel(d, dev, F2_NOTE)
        fig.savefig(out / "f2b_dev.png", dpi=300)
        plt.close(fig)

    fig, (a, d) = plt.subplots(1, 2, figsize=(10.5, 3.4), layout="constrained",
                               gridspec_kw={"width_ratios": (1.35, 1)})
    hs = f2_tuning_panel(a, perf, dev, F2_NOTE)
    a.set_title("(a) perf split: selection runs, 1 rep per knob (tuning data, not results)", fontsize=8, pad=3)
    if dev:
        f2_dev_panel(d, dev, F2_NOTE)
        ns = sorted({c["n"] for c in dev.values()})
        d.set_title(f"(b) dev split: {'/'.join(map(str, ns))} reps per cell at the knobs run on dev",
                    fontsize=8, pad=3)
    else:
        d.set_axis_off()
        d.text(0.5, 0.5, "dev 2×2 reps (p2_2x2): not run yet", ha="center", va="center", fontsize=7)
    fig.legend(hs, [h.get_label() for h in hs], loc="outside upper center", ncol=3, frameon=False,
               fontsize=7, handlelength=2.2, columnspacing=1.2, borderaxespad=0.1)
    fig.savefig(out / "f2_decomposition.png")
    plt.close(fig)


def kv_grid(run: Path, bin_s: float = 2.0) -> tuple[list[dict[str, float]], str]:
    """A run's KV timeline in memory (no file written into results/), and its mode."""
    events = load_events(run)
    if has_exact(events):
        ivs, _ = exact_intervals(events)
        mode = "approximate" if approximate(run, events) else "exact"
        return timeline(ivs, bin_s, EXACT_STATES), mode
    ivs, _ = intervals(events)
    return timeline(ivs, bin_s, STATES), "reconstructed"


def server_marks(run: Path) -> tuple[list[float], list[float]]:
    """Times of llama.cpp purges (kv_state reason server_purge) and of failed tasks
    (task_done status ctx_overflow / kv_infeasible / error), for ticks on the F3 panels."""
    purges, fails = [], []
    for e in load_events(run):
        if e["type"] == "kv_state" and e.get("reason") == "server_purge":
            purges.append(e["t"])
        elif e["type"] == "task_done" and e.get("status") not in ("ok", "forced"):
            fails.append(e["t"])
    return purges, fails


def fig_f3(runs: list[R.Run], out: Path) -> None:
    cells = R.main_cells(runs).get(R.HEAD, [])
    p, _ = R.picks(R.tuning_cells(runs), "B1").get(R.HEAD, (None, None))
    b1s = sorted((c for c in cells if c["cfg"] == "B1"), key=lambda c: (p is None or c["k"] != p["k"], c["k"]))
    ours = next((c for c in cells if c["cfg"] == "ours"), None)
    show = [(c, f"B1 K={c['k']}" if c["cfg"] == "B1" else f"ours ω={c['omega']:g}")
            for c in (b1s[:1] + [ours]) if c]
    show = [(c, name) for c, name in show if (c["runs"][0]["dir"] / "events.jsonl").exists()]
    if not show:
        return
    grids = [(c, name, *kv_grid(c["runs"][0]["dir"], 4.0)) for c, name in show]
    peak = max((sum(g[s] for s in GPU_STATES if s in g) for _, _, grid, _ in grids for g in grid), default=0)
    fig, axes = plt.subplots(len(show), 1, figsize=(W, 0.75 + 1.3 * len(show)), sharex=True, squeeze=False,
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0.04)
    handles = {}
    for a, (c, name, grid, mode) in zip(axes[:, 0], grids):
        run = c["runs"][0]
        if not grid:
            continue
        t = [g["t"] for g in grid]
        gpu = [s for s in GPU_STATES if s in grid[0] and any(g[s] for g in grid)]
        polys = a.stackplot(t, *[[g[s] / 1000 for g in grid] for s in gpu], colors=[COLOR[s] for s in gpu], lw=0)
        handles.update(dict(zip(gpu, polys)))
        for s in (x for x in HOST_TIERS if x in grid[0] and any(g[x] for g in grid)):
            (h,) = a.plot(t, [g[s] / 1000 for g in grid], color=COLOR[s], lw=0.8)
            handles[f"{s} (off-GPU)"] = h
        handles["-c (KV pool)"] = a.axhline(run["ctx"] / 1000, color="#c53030", lw=0.8, ls="--")
        top = max(run["ctx"], peak) / 1000 * 1.12   # one scale for both panels, never clipping a peak
        purges, fails = server_marks(run["dir"])
        if purges:
            handles["llama.cpp purge"] = a.scatter(purges, [top * 0.985] * len(purges), marker="v", s=9,
                                                   color="#4a5568", lw=0, zorder=4, clip_on=False)
        if fails:
            handles["failed task"] = a.scatter(fails, [top * 0.985] * len(fails), marker="x", s=12,
                                                          color="#c53030", lw=0.8, zorder=4, clip_on=False)
        a.set_title(f"{name}, dev {run['rep']} ({mode} states)"
                    + (f": {len(purges)} purges, {len(fails)} failed tasks" if purges or fails else
                       ": no purges, no failed tasks"), fontsize=6.5, pad=2)
        a.set_ylabel("GPU KV (k tokens)", fontsize=7)
        a.set_ylim(0, top)
        a.tick_params(labelsize=6.5)
    fig.supxlabel("time since the run started (s)\nactive = each request at its final token count "
                  "(an upper bound while it runs)", fontsize=6.3)
    order = [*(s for s in GPU_STATES if s in handles), *(k for k in handles if k.endswith("(off-GPU)")),
             *(k for k in ("-c (KV pool)", "llama.cpp purge", "failed task") if k in handles)]
    fig.legend([handles[k] for k in order], order, loc="outside upper center", ncol=3, frameon=False,
               fontsize=5.8, handlelength=1.2, columnspacing=0.9, borderaxespad=0.1)
    fig.savefig(out / "f3_timeline_16k_x4.png")
    plt.close(fig)


def fig_f4(runs: list[R.Run], out: Path) -> None:
    abl = [c for c in R.ablation_cells(runs) if c["goodput"] is not None]
    if not abl:
        return
    main = R.main_cells(runs).get(R.HEAD, [])
    ours = next((c for c in main if c["cfg"] == "ours"), None)
    p, _ = R.picks(R.tuning_cells(runs), "B1").get(R.HEAD, (None, None))
    b1 = next((c for c in main if c["cfg"] == "B1" and p and c["k"] == p["k"]),
              next((c for c in main if c["cfg"] == "B1"), None))
    refs = [(c, f"{c['cfg']} (p2_main)") for c in main if c["cfg"] == "ours_h0" and c["goodput"] is not None]
    rows = sorted([*refs, *((c, R.cell_name(c)) for c in abl)], key=lambda x: x[0]["goodput"])
    fig, a = plt.subplots(figsize=(W, 0.85 + 0.19 * len(rows)), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    if ours and ours["goodput"] is not None:   # ours' spread over its reps: the noise a preset must beat
        a.axvspan(ours["goodput_lo"], ours["goodput_hi"], color=MANAGED, alpha=0.12, lw=0,
                  label=f"ours min–max (n={ours['n']})")
        a.axvline(ours["goodput"], color=MANAGED, lw=1.2, label=f"ours mean {ours['goodput']:.2f}")
    if b1 and b1["goodput"] is not None:
        a.axvline(b1["goodput"], color=NOGATE, lw=1, ls="--", label=f"B1 K={b1['k']} mean {b1['goodput']:.2f} "
                  f"(n={b1['n']})")
    for i, (c, _) in enumerate(rows):
        multi = c["n"] > 1
        a.errorbar([c["goodput"]], [i], xerr=[[c["goodput"] - c["goodput_lo"]], [c["goodput_hi"] - c["goodput"]]]
                   if multi else None, fmt="o", color="#2d3748", ms=3.5, mfc="#2d3748" if multi else "white",
                   capsize=1.5, lw=0.8, zorder=3)
        if multi:
            reps = [r["goodput"] for r in c["runs"] if r["goodput"] is not None]
            a.scatter(reps, [i + 0.22] * len(reps), s=5, color="#718096", lw=0, zorder=2)
        txt = f"{R.rel(ours['goodput'], c['goodput']):+.1f}%" if ours and ours["goodput"] else ""
        if c["completion"] is not None and c["completion"] < 0.995:
            txt += f", {100 * c['completion']:.0f}% done"
        a.annotate(txt, (c["goodput_hi"] if multi else c["goodput"], i), textcoords="offset points", xytext=(5, 0),
                   va="center", fontsize=5.8, color="#2d3748")
    a.set_yticks(range(len(rows)), [f"{name} (n={c['n']})" for c, name in rows], fontsize=6.5)
    a.invert_yaxis()
    fig.supxlabel("overall goodput (correct tasks/min), 16k ×4 dev\nfilled: mean, min–max bar, reps in grey; "
                  "hollow: 1 run; text: Δ vs ours' mean", fontsize=6.3)
    a.tick_params(axis="x", labelsize=6.5)
    a.grid(axis="y", visible=False)
    xs = [x for c, _ in rows for x in (c["goodput_lo"], c["goodput_hi"]) if x is not None]
    xs += [v for v in ((b1 or {}).get("goodput"), (ours or {}).get("goodput_lo")) if v is not None]
    a.set_xlim(min(xs) - 0.3, max(xs) + 1.1)   # dots, not bars, so the axis need not start at 0
    fig.legend(*a.get_legend_handles_labels(), frameon=False, fontsize=5.8, loc="outside upper center", ncol=2,
               handlelength=1.6, columnspacing=0.8)
    fig.savefig(out / "f4_ablations.png")
    plt.close(fig)


def head_rep0(root: Path) -> list[Path]:
    return sorted((root / R.MAIN).glob("c16k_d4_*/rep0"))


def fig_states(root: Path, out: Path) -> None:
    runs = [p / "kv_summary.json" for p in head_rep0(root) if (p / "kv_summary.json").exists()]
    labels, rows, approx = [], [], []
    for p in runs:
        kv = json.loads(p.read_text(encoding="utf-8"))
        if kv.get("mode") not in ("exact", "approximate"):
            continue
        name = p.parts[-3].removeprefix("c16k_d4_")
        approx.append(kv["mode"] == "approximate")
        labels.append(f"{name} (approx.)" if approx[-1] else name)
        rows.append(kv["token_seconds"])
    if not rows:
        return
    fig, a = plt.subplots(figsize=(W, 2.4))
    bottom = [0.0] * len(rows)
    for s in (*GPU_STATES, *HOST_TIERS):
        vals = [r.get(s, 0) / 1e6 for r in rows]
        if not any(vals):
            continue
        bars = a.bar(labels, vals, bottom=bottom, color=COLOR[s], label=s if s in GPU_STATES else f"{s} (off-GPU)")
        for b, ap in zip(bars, approx, strict=True):
            if ap:
                b.set_hatch("//")
                b.set_alpha(0.6)
        bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
    a.set_ylabel("KV token·s (M)")
    a.tick_params(axis="x", labelsize=6.5, rotation=20)
    a.legend(frameon=False, ncol=4, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "kv_states.png")
    plt.close(fig)


def fig_timelines(root: Path, out: Path) -> None:
    for run in head_rep0(root):
        if not (run / "events.jsonl").exists():
            continue
        grid, mode = kv_grid(run, 2.0)
        if not grid:
            continue
        gpu = [s for s in GPU_STATES if s in grid[0]]
        t = [g["t"] for g in grid]
        fig, a = plt.subplots(figsize=(W, 1.9))
        a.stackplot(t, *[[g[s] / 1000 for g in grid] for s in gpu], colors=[COLOR[s] for s in gpu], labels=gpu, lw=0)
        for s in (x for x in HOST_TIERS if x in grid[0]):   # host memory: drawn, not stacked
            a.plot(t, [g[s] / 1000 for g in grid], color=COLOR[s], lw=0.8, label=f"{s} (off-GPU)")
        meta_p, kv_p = run / "meta.json", run / "kv_summary.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        ctx = (meta.get("kvm") or {}).get("ctx") or 16384
        a.axhline(ctx / 1000, color="#c53030", lw=0.8, ls="--", label="-c")
        kv = json.loads(kv_p.read_text(encoding="utf-8")) if kv_p.exists() else {}
        if cap := (kv.get("manager") or {}).get("cap"):
            a.axhline(float(cap) / 1000, color="#c53030", lw=0.6, ls=":", label="c - m")
        a.set_title(f"{run.parts[-2]} ({mode})", fontsize=7, pad=2)
        a.set_xlabel("time (s)")
        a.set_ylabel("KV tokens (k)")
        a.legend(frameon=False, ncol=5, fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.08))
        fig.tight_layout(pad=0.3)
        fig.savefig(out / f"timeline_{run.parts[-2]}.png")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="docs/figures/phase2")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs, _ = R.load_runs(root)
    for fig in (fig_f1, fig_f2, fig_f3, fig_f4):
        fig(runs, out)
    fig_states(root, out)
    fig_timelines(root, out)
    print(f"figures -> {out}: {', '.join(sorted(p.name for p in out.glob('*.png'))) or 'none'}")


if __name__ == "__main__":
    main()
