"""Figures for an experiment directory.

    uv run python analysis/plots.py results/phase1_sweep [--x p.max_concurrency] [--hue p.rate]

Writes into <exp>/figures/:
  sweep_*.png     throughput / tail latency / peak memory / quality vs. the swept knob
  timeline_*.png  per-run memory, queue length and in-flight requests over time
  context_*.png   per-request prompt length vs. latency (context growth effect)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentserve.metrics.system import load_run  # noqa: E402
from analysis.summarize import collect  # noqa: E402

SWEEP_METRICS = [
    ("throughput_tasks_per_min", "finished tasks / min"),
    ("task_latency_p50", "task latency p50 (s)"),
    ("task_latency_p95", "task latency p95 (s)"),
    ("req_queue_p95", "request queue wait p95 (s)"),
    ("peak_phys_footprint_mb", "peak backend memory (MB)"),
    ("peak_gpu_mem_used_mb", "peak GPU memory (MB)"),
    ("failure_rate", "failure rate"),
    ("f1", "answer F1"),
    ("preemptions_total", "preemptions"),
    ("recompute_tokens_total", "recomputed prompt tokens"),
    ("reuse_miss_tokens_total", "lost-KV tokens between agent steps"),
    ("mix.long_ctx.slowdown_p95", "long-task slowdown p95"),
]


def plot_sweep(df, x: str, hue: str | None, out: Path) -> None:
    metrics = [(m, lbl) for m, lbl in SWEEP_METRICS if m in df and df[m].notna().any()]
    ncol = (len(metrics) + 2) // 3
    fig, axes = plt.subplots(3, ncol, figsize=(4 * ncol, 10), squeeze=False)
    categorical = not pd.api.types.is_numeric_dtype(df[x])
    for ax, (m, lbl) in zip(axes.flat, metrics):
        groups = df.groupby(hue) if hue and hue in df else [(None, df)]
        if categorical:
            cats = list(dict.fromkeys(df[x]))
            groups = list(groups)
            w = 0.8 / len(groups)
            for gi, (g, gdf) in enumerate(groups):
                agg = gdf.groupby(x)[m].mean().reindex(cats)
                ax.bar([i + gi * w for i in range(len(cats))], agg.values, width=w,
                       label=f"{hue.removeprefix('p.')}={g}" if g is not None else None)
            ax.set_xticks([i + w * (len(groups) - 1) / 2 for i in range(len(cats))])
            ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=7)
            ax.set_title(lbl, fontsize=10)
            ax.grid(alpha=0.3, axis="y")
            continue
        for g, gdf in groups:
            # average over seeds, show min/max as a band
            agg = gdf.groupby(x)[m].agg(["mean", "min", "max"]).sort_index()
            ax.plot(agg.index, agg["mean"], marker="o", label=f"{hue.removeprefix('p.')}={g}" if g is not None else None)
            ax.fill_between(agg.index, agg["min"], agg["max"], alpha=0.15)
        ax.set_xlabel(x.removeprefix("p."))
        ax.set_title(lbl, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    if hue:
        axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_timeline(run_dir: Path, out: Path) -> None:
    d = load_run(run_dir)
    mon, tasks = d["monitor"], d["tasks"]
    if mon.empty:
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for col in ("phys_footprint_bytes", "rss_bytes", "gpu_mem_used_bytes"):
        if col in mon and mon[col].notna().any():
            s = mon.dropna(subset=[col])
            a1.plot(s.t, s[col] / 2**20, label=col.removesuffix("_bytes"))
    a1.set_ylabel("MB")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    for col in ("queue_len", "inflight", "active_tasks"):
        if col in mon and mon[col].notna().any():
            s = mon.dropna(subset=[col])
            a2.step(s.t, s[col], where="post", label=col)
    if not tasks.empty:
        for t in tasks.t_done:
            a2.axvline(t, color="k", alpha=0.08)
    a2.set_xlabel("time (s)")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    fig.suptitle(run_dir.name, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_context(run_dir: Path, out: Path) -> None:
    req = load_run(run_dir)["requests"]
    if req.empty:
        return
    ok = req[req.status == "ok"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.scatter(ok.prompt_tokens, ok.exec_time, s=8, alpha=0.5, c=ok.step, cmap="viridis")
    a1.set_xlabel("prompt tokens")
    a1.set_ylabel("request exec time (s)")
    a1.set_title("context size vs. latency (colour = step)", fontsize=10)
    a2.scatter(ok.prompt_tokens, ok.cached_tokens / ok.prompt_tokens.clip(lower=1), s=8, alpha=0.5)
    a2.set_xlabel("prompt tokens")
    a2.set_ylabel("cached fraction")
    a2.set_title("prefix reuse", fontsize=10)
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("--x", help="swept column (default: first p.* column with >1 value)")
    ap.add_argument("--hue", help="second swept column for separate lines")
    args = ap.parse_args()
    exp = Path(args.exp_dir)
    figs = exp / "figures"
    figs.mkdir(exist_ok=True)
    df = collect(exp)
    if df.empty:
        sys.exit(f"no runs in {exp}")
    swept = [c for c in df.columns if c.startswith("p.") and c != "p.seed" and df[c].nunique() > 1]
    x = args.x or (swept[0] if swept else None)
    hue = args.hue or next((c for c in swept if c != x), None)
    if x:
        plot_sweep(df, x, hue, figs / f"sweep_{x.removeprefix('p.')}.png")
    for run in df.run:
        plot_timeline(exp / run, figs / f"timeline_{run}.png")
        plot_context(exp / run, figs / f"context_{run}.png")
    print(f"figures -> {figs}")


if __name__ == "__main__":
    main()
