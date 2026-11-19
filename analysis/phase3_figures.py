"""Figures for the Phase 3 (TRIM) report, from results/runs.csv (run analysis.aggregate first) and
the run directories. Cells, labels and metrics are analysis.phase3_report's, so a figure and its
table never disagree.

    uv run --no-sync --with matplotlib python -m analysis.phase3_figures [--root results] [--out docs/figures/phase3]

  f6_selectors.png  ample memory (64k x1, B1 K=8): F1 vs KV held when a tool call starts, per selector,
                    on perf (p3_select, filled) and dev (p3_quality, hollow); bars = min-max F1 over reps
  f7_trim_dev.png   dev goodput with and without TRIM at 16k x4 and 8k x1 (P4's cells): mean, min-max,
                    reps; untrimmed rows from p2_main, ours' naive rows before the shrink-fast fix
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import phase3_report as P
from analysis.phase2_report import budget, rel
from analysis.phase2_figures import MANAGED, NOGATE, W

SEL_COLOR = {"none": "#2d3748", "oracle": "#b7791f", "referenced": "#805ad5", "relevance:2": "#2b6cb0",
             "relevance:3": "#63b3ed", "drop": "#c53030"}
ROW_STYLE = {   # (colour, marker, filled)
    "B1": (NOGATE, "o", True), "ours": (MANAGED, "s", True),
    "B1+oracle": ("#b7791f", "o", True), "B1+rel2": ("#2b6cb0", "o", True),
    "ours+oracle (naive)": ("#b7791f", "s", False), "ours+rel2 (naive)": ("#2b6cb0", "s", False),
    "ours+oracle": ("#b7791f", "s", True), "ours+rel2": ("#2b6cb0", "s", True)}


def fig_f6(runs: list[P.Run], out: Path) -> None:
    perf = P.by_trim(P.cells_by(runs, P.SELECT))
    dev = P.by_trim(P.cells_by(runs, P.QUALITY))
    if not perf and not dev:
        return
    pick = P.chosen(perf)
    fig, a = plt.subplots(figsize=(W, 2.35), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    for split, cs, filled in (("perf", perf, True), ("dev", dev, False)):
        for m in P.SEL_ORDER:
            c = cs.get(m)
            if c is None or c.get("held") is None:
                continue
            col = SEL_COLOR[m]
            a.errorbar([c["held"]], [c["f1"]], yerr=[[c["f1"] - c["f1_lo"]], [c["f1_hi"] - c["f1"]]], fmt="o",
                       color=col, mfc=col if filled else "white", mec=col, ms=4.5, capsize=1.5, lw=0.8, zorder=3)
            if split == "perf":
                name = m + (" ★" if m == pick else "")
                a.annotate(name, (c["held"], c["f1"]), textcoords="offset points", xytext=(4, -7 if m == "drop" else 3),
                           fontsize=5.8, color=col)
    for split, cs, ls in (("perf", perf, "-"), ("dev", dev, ":")):
        if "none" in cs:
            a.axhline(cs["none"]["f1"], color="#a0aec0", lw=0.7, ls=ls, zorder=1)
    a.plot([], [], "o", color="#4a5568", ms=4, label="perf (p1_long_200)")
    a.plot([], [], "o", mfc="white", mec="#4a5568", color="#4a5568", ms=4, label="dev (dev_long_200)")
    a.set_xlabel("KV held when a tool call starts (tokens, mean)", fontsize=6.5)
    a.set_ylabel("F1 (mean, min–max over 2 reps)", fontsize=6.5)
    a.tick_params(labelsize=6.5)
    a.legend(frameon=False, fontsize=6, loc="lower left")
    xs = [c["held"] for cs in (perf, dev) for c in cs.values() if c.get("held")]
    a.set_xlim(min(xs) - 80, max(xs) + 220)
    fig.savefig(out / "f6_selectors.png")
    plt.close(fig)


def fig_f7(runs: list[P.Run], out: Path) -> None:
    cells = P.main_cells(runs)
    pts = [b for b in P.POINTS if any(c["trim"] != "none" for c in (cells.get(b) or {}).values())]
    if not pts:
        return
    rows_by = {b: sorted(cells[b], key=P.row_key) for b in pts}
    fig, axes = plt.subplots(len(pts), 1, figsize=(W, 0.75 + sum(0.19 * len(rows_by[b]) + 0.3 for b in pts)),
                             squeeze=False, layout="constrained",
                             gridspec_kw={"height_ratios": [len(rows_by[b]) + 1 for b in pts]})
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0.04)
    for a, b in zip(axes[:, 0], pts):
        cs = cells[b]
        ours = cs.get("ours")
        xs, labels = [], []
        for i, name in enumerate(rows_by[b]):
            c = cs[name]
            base = "B1" if name.startswith("B1 K=") else name
            col, mk, filled = ROW_STYLE.get(base, ("#718096", "^", True))
            reps = [r["goodput"] for r in c["runs"] if r["goodput"] is not None]
            a.scatter(reps, [i + 0.24] * len(reps), s=5, color="#718096", lw=0, zorder=2)
            a.errorbar([c["goodput"]], [i], xerr=[[c["goodput"] - c["goodput_lo"]], [c["goodput_hi"] - c["goodput"]]],
                       fmt=mk, color=col, mfc=col if filled else "white", mec=col, ms=4, capsize=1.5, lw=0.9, zorder=3)
            xs += [c["goodput_lo"], c["goodput_hi"]]
            txt = f"{rel(ours['goodput'], c['goodput']):+.0f}% vs ours" if ours and c is not ours else ""
            if c["completion"] is not None and c["completion"] < 0.995:
                txt += (", " if txt else "") + f"{100 * c['completion']:.0f}% done"
            a.annotate(txt, (c["goodput_hi"], i), textcoords="offset points", xytext=(5, 0), va="center",
                       fontsize=5.6, color="#2d3748")
            labels.append(f"{name}{' ★' if c.get('star') else ''}" + ("" if name.startswith("B1 K=") or "ours" in name
                                                                      else f" K={c['k']}"))
        a.set_yticks(range(len(labels)), labels, fontsize=6.2)
        a.invert_yaxis()
        a.grid(axis="y", visible=False)
        a.tick_params(axis="x", labelsize=6.5)
        a.set_xlim(min(xs) - 0.4, max(xs) + 3.2)
        a.set_title(f"{budget(*b)}: dev, 3 reps", fontsize=7, pad=2)
    fig.supxlabel("overall goodput (correct tasks/min); dot: mean, bar: min–max, grey: reps\nhollow: ours before the "
                  "shrink-fast fix; untrimmed rows: Phase 2 (p2_main)", fontsize=5.8)
    fig.savefig(out / "f7_trim_dev.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="docs/figures/phase3")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = P.load(root)
    for fig in (fig_f6, fig_f7):
        fig(runs, out)
    print(f"figures -> {out}: {', '.join(sorted(p.name for p in out.glob('*.png'))) or 'none'}")


if __name__ == "__main__":
    main()
