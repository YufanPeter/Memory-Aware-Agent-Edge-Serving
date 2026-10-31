"""Figures for the Phase 1 report, from results/ (run analysis.aggregate first).

    uv run --with matplotlib python -m analysis.phase1_figures [--out docs/figures/phase1]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from costmodel.calibrated import fit_decode, fit_slot_ops  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
LABEL = {"active": "in a request (computing or queued at GPU)", "idle_tool": "idle (waiting on a tool)",
         "stale": "stale (task finished)"}
C = {"active": "#2b6cb0", "idle_tool": "#dd6b20", "stale": "#a0aec0", "pred": "#718096"}


def points(root: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(open(root / "points.csv", encoding="utf-8")))


def num(row: dict[str, str], key: str) -> float | None:
    v = row.get(key)
    return float(v) if v not in (None, "") else None


def sweep(rows: list[dict[str, str]], name: str) -> list[dict[str, str]]:
    return [r for r in rows if r["sweep"] == name]


def fig_m0(root: Path, out: Path) -> None:
    run = sorted(p for p in (root / "microbench").iterdir() if p.is_dir() and p.name.startswith("a10g"))[-1]
    decode = fit_decode(run / "batched_f16.txt")
    frozen = json.loads(sorted(Path("predictions").glob("predictions_*.json"))[0].read_text())
    pred = next(p for p in frozen["predictions"] if p["hardware"].startswith("A10G") and p["kv_type"] == "f16")
    pred_dec = {int(k): v for k, v in pred["profiles"]["react_short"]["decode_tok_s_by_batch"].items()}
    csvs = sorted((root / "microbench").glob("slot_offload_a10g*.csv"))
    rows = list(csv.DictReader(open(csvs[-1], encoding="utf-8")))
    ops = fit_slot_ops(csvs[-1])

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    bs = sorted(decode)
    a.plot(bs, [decode[x] for x in bs], "o-", color=C["active"], label="measured")
    a.plot(sorted(pred_dec), [pred_dec[x] for x in sorted(pred_dec)], "s--", color=C["pred"],
           label="predicted (spec sheet)")
    a.set_xscale("log", base=2)
    a.set_xlabel("concurrent decoding requests")
    a.set_ylabel("total decode tokens/s")
    a.set_title("(a) Decode throughput vs batch, 2k context")
    a.legend(frameon=False)

    ns = sorted({int(r["n_tokens"]) for r in rows})
    med = lambda key, n: sorted(float(r[key]) for r in rows if int(r["n_tokens"]) == n)[1] / 1e3  # noqa: E731
    b.plot(ns, [med("reprefill_ms", n) for n in ns], "o-", color=C["active"], label="drop: re-prefill")
    b.plot(ns, [med("save_ms", n) + med("restore_ms", n) for n in ns], "o-", color=C["idle_tool"],
           label="offload: save + restore")
    b.axhline(0.8, color=C["pred"], ls=":", lw=1)
    b.text(ns[0], 0.84, "median tool delay 0.8 s", color=C["pred"], fontsize=7)
    b.set_xlabel("context tokens in the slot")
    b.set_ylabel("seconds")
    b.set_title("(b) Cost of giving KV up")
    b.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "m0_microbench.png")
    plt.close(fig)
    print(f"m0: save {ops['save_Bps'] / 1e9:.2f} GB/s, restore {ops['restore_Bps'] / 1e9:.2f} GB/s")


def fig_e1(rows: list[dict[str, str]], out: Path) -> None:
    rs = sorted(sweep(rows, "e1_concurrency"), key=lambda r: int(r["k"]))
    if not rs:
        return
    k = [int(r["k"]) for r in rs]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(9.6, 2.8))
    for key, lab, col in (("steady_throughput_per_min", "completed", C["active"]),
                          ("steady_goodput_per_min", "correct (EM)", C["idle_tool"])):
        y = [num(r, key) for r in rs]
        a.plot(k, y, "o-", color=col, label=lab)
        lo = [num(r, key + "_min") for r in rs]
        hi = [num(r, key + "_max") for r in rs]
        for x, l, h in zip(k, lo, hi):
            if l is not None:
                a.vlines(x, l, h, color=col, lw=2, alpha=0.5)
    split = sorted(sweep(rows, "e1_split"), key=lambda r: int(r["k"]))
    if split:
        a.plot([int(r["k"]) for r in split], [num(r, "steady_throughput_per_min") for r in split], "D--",
               color=C["pred"], label="completed, per-slot KV (no unified pool)")
    a.set_xscale("log", base=2)
    a.set_xlabel("K (concurrent agents)")
    a.set_ylabel("tasks/min (steady state)")
    a.set_title("(a) Throughput")
    a.legend(frameon=False, fontsize=7)
    for key, lab, ls in (("service_p50_s", "p50", "-"), ("service_p95_s", "p95", "--")):
        b.plot(k, [num(r, key) for r in rs], "o" + ls, color=C["active"], label=f"task service time {lab}")
    b.plot(k, [num(r, "ttft_p50_s") for r in rs], "^:", color=C["idle_tool"], label="TTFT p50 (per request)")
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    b.set_xlabel("K")
    b.set_ylabel("seconds")
    b.set_title("(b) Latency")
    b.legend(frameon=False, loc="upper left", fontsize=7)
    bottom = [0.0] * len(rs)
    for s in ("active", "idle_tool", "stale"):
        share = [num(r, f"kv_share_{s}") or 0 for r in rs]
        mean = [num(r, "kv_mean_tokens") or 0 for r in rs]
        h = [m * f / 1000 for m, f in zip(mean, share)]
        c.bar([str(x) for x in k], h, bottom=bottom, color=C[s], label=LABEL[s])
        bottom = [x + y for x, y in zip(bottom, h)]
    c.set_xlabel("K")
    c.set_ylabel("mean KV held (k tokens)")
    c.set_title("(c) KV held, by owner state")
    c.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "e1_concurrency.png")
    plt.close(fig)


def fig_e1_poisson(rows: list[dict[str, str]], out: Path) -> None:
    rs = sorted(sweep(rows, "e1_poisson"), key=lambda r: float(r["point"].lstrip("r")))
    if not rs:
        return
    x = [float(r["point"].lstrip("r")) * 60 for r in rs]
    fig, a = plt.subplots(figsize=(3.6, 2.8))
    a.plot(x, [num(r, "latency_trim_p50_s") for r in rs], "o-", color=C["active"], label="p50")
    a.plot(x, [num(r, "latency_trim_p95_s") for r in rs], "o--", color=C["active"], label="p95")
    a.plot(x, [num(r, "queue_wait_p95_s") for r in rs], "^:", color=C["idle_tool"], label="queue wait p95")
    a.set_xlabel("arrival rate (tasks/min), K=16")
    a.set_ylabel("seconds")
    a.set_yscale("log")
    a.set_title("Open-loop latency vs load")
    a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "e1_poisson.png")
    plt.close(fig)


def fig_e2(rows: list[dict[str, str]], out: Path) -> None:
    rs = sweep(rows, "e2_budget") + sweep(rows, "e2_budget_k")
    if not rs:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    colors = {2: "#68d391", 4: "#2b6cb0", 8: "#dd6b20", 16: "#9b2c2c"}
    for k in sorted({int(r["k"]) for r in rs}):
        pts = sorted((int(r["ctx"]) // 1024, 100 * float(r["completed"]) / int(r["n_tasks"]))
                     for r in rs if int(r["k"]) == k)
        a.plot([x for x, _ in pts], [y for _, y in pts], "o-", color=colors.get(k), label=f"K={k}")
    a.set_xscale("log", base=2)
    a.set_xticks([8, 16, 32, 64], ["8k", "16k", "32k", "64k"])
    a.set_xlabel("KV pool -c (tokens), long-context mix")
    a.set_ylabel("% of tasks completed")
    a.set_title("(a) Completion vs budget (K doubles)")
    a.legend(frameon=False, fontsize=7)
    best = []
    for ctx in sorted({int(r["ctx"]) for r in rs}):
        ok = [r for r in rs if int(r["ctx"]) == ctx and float(r["completed"]) / int(r["n_tasks"]) >= 0.85]
        if ok:
            best.append(max(ok, key=lambda r: num(r, "steady_throughput_per_min") or 0))
    x = [int(r["ctx"]) // 1024 for r in best]
    b.bar([str(v) + "k" for v in x], [num(r, "steady_throughput_per_min") for r in best], color=C["active"])
    for i, r in enumerate(best):
        b.text(i, (num(r, "steady_throughput_per_min") or 0) + 1, f"K={r['k']}\nidle {100 * (num(r, 'kv_share_idle_tool') or 0):.0f}%",
               ha="center", fontsize=7)
    b.set_ylim(0, 60)
    b.set_xlabel("KV pool -c (tokens)")
    b.set_ylabel("tasks/min (steady state)")
    b.set_title("(b) Largest tested K completing ≥85%")
    fig.tight_layout()
    fig.savefig(out / "e2_budget.png")
    plt.close(fig)


def fig_e3_e4(rows: list[dict[str, str]], out: Path) -> None:
    e1 = {r["point"]: r for r in sweep(rows, "e1_concurrency")}
    d = {r["point"]: r for r in sweep(rows, "e3_delay")}
    if "k16" in e1:
        d["d1"] = e1["k16"]
    e4 = {r["point"]: r for r in sweep(rows, "e4_prefix")}
    if "k8" in e1:
        e4["pin_cache"] = e1["k8"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    order = [p for p in ("d0", "d1", "d4") if p in d]
    if order:
        x = [{"d0": 0, "d1": 1, "d4": 4}[p] for p in order]
        a.plot(x, [num(d[p], "steady_throughput_per_min") for p in order], "o-", color=C["active"],
               label="tasks/min")
        a.set_xlabel("tool-delay scale (median 0.8 s × scale), K=16")
        a.set_ylabel("tasks/min (steady state)")
        a.set_ylim(0, 60)
        a2 = a.twinx()
        a2.set_ylim(0, 100)
        a2.plot(x, [100 * (num(d[p], "kv_share_idle_tool") or 0) for p in order], "s--", color=C["idle_tool"])
        a2.set_ylabel("% of KV token·s held idle in tools", color=C["idle_tool"])
        a.set_title("(a) Idle periods")
    labels = [p for p in ("pin_cache", "pin_nocache", "nopin_cache", "nopin_nocache") if p in e4]
    if labels:
        y = [num(e4[p], "steady_throughput_per_min") for p in labels]
        b.bar(range(len(labels)), y, color=[C["active"] if "nocache" not in p else C["stale"] for p in labels])
        b.set_xticks(range(len(labels)), [p.replace("_", "\n") for p in labels])
        for i, p in enumerate(labels):
            hit = num(e4[p], "cache_hit_rate")
            b.text(i, (y[i] or 0) * 1.01, f"hit {hit:.0%}" if hit is not None else "", ha="center", fontsize=7)
        b.set_ylabel("tasks/min (steady state)")
        b.set_title("(b) Prefix reuse, K=8")
    fig.tight_layout()
    fig.savefig(out / "e3_e4.png")
    plt.close(fig)


def fig_summary(rows: list[dict[str, str]], out: Path) -> None:
    """The two main Phase 1 results side by side, for the 2-page report."""
    e1 = sorted(sweep(rows, "e1_concurrency"), key=lambda r: int(r["k"]))
    split = sorted(sweep(rows, "e1_split"), key=lambda r: int(r["k"]))
    e2 = sweep(rows, "e2_budget") + sweep(rows, "e2_budget_k")
    if not (e1 and e2):
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.5))
    a.plot([int(r["k"]) for r in e1], [num(r, "steady_throughput_per_min") for r in e1], "o-",
           color=C["active"], label="unified KV pool (default)")
    a.plot([int(r["k"]) for r in split], [num(r, "steady_throughput_per_min") for r in split], "D--",
           color=C["idle_tool"], label="per-slot KV regions")
    a.set_xscale("log", base=2)
    a.set_xticks([1, 4, 8, 16, 32], ["1", "4", "8", "16", "32"])
    a.set_ylim(0, 70)
    a.set_xlabel("K (concurrent agents), ample KV pool")
    a.set_ylabel("tasks/min (steady state)")
    a.set_title("(a) Throughput vs concurrency (E1)")
    a.legend(frameon=False, fontsize=7, loc="lower right")
    colors = {2: "#68d391", 4: "#2b6cb0", 8: "#dd6b20", 16: "#9b2c2c"}
    for k in sorted({int(r["k"]) for r in e2}):
        pts = sorted((int(r["ctx"]) // 1024, 100 * float(r["completed"]) / int(r["n_tasks"]))
                     for r in e2 if int(r["k"]) == k)
        b.plot([x for x, _ in pts], [y for _, y in pts], "o-", color=colors.get(k), label=f"K={k}")
    b.set_xscale("log", base=2)
    b.set_xticks([8, 16, 32, 64], ["8k", "16k", "32k", "64k"])
    b.set_xlabel("KV budget -c (tokens), long-context mix")
    b.set_ylabel("% of tasks completed")
    b.set_title("(b) Completion vs KV budget (E2)")
    b.legend(frameon=False, fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "summary.png")
    plt.close(fig)


def fig_pressure(root: Path, out: Path) -> None:
    """KV held over time under a tight budget (E2, -c 16k, K=4), against the pool limit."""
    from analysis.kv_timeline import intervals, load_events, timeline
    run = root / "e2_budget_k/c16k_k4/rep0"
    if not (run / "events.jsonl").exists():
        return
    ivs, _ = intervals(load_events(run))
    grid = timeline(ivs, 2.0)
    t = [r["t"] for r in grid]
    ys = [[r[s] / 1000 for r in grid] for s in ("active", "idle_tool", "stale")]
    fig, a = plt.subplots(figsize=(3.5, 1.75))
    a.stackplot(t, *ys, colors=[C["active"], C["idle_tool"], C["stale"]],
                labels=["in a request", "idle (waiting on a tool)", "stale (task finished)"])
    a.axhline(16.384, color="#c53030", lw=1, ls="--")
    a.text(t[-1] * 0.99, 16.9, "KV budget (16k)", color="#c53030", fontsize=7, ha="right")
    a.set_xlim(0, t[-1])
    a.set_ylim(0, 19)
    a.set_xlabel("time (s)")
    a.set_ylabel("KV tokens held (k)")
    a.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.5,
             handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "pressure_timeline.png")
    plt.close(fig)


def fig_e5(root: Path, out: Path) -> None:
    for name, run in (("e5_timeline_k16", root / "e1_concurrency/k16/rep0"),
                      ("e5_timeline_d4", root / "e3_delay/d4/rep0")):
        path = run / "kv_timeline.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        t = [float(r["t"]) for r in rows]
        ys = [[float(r[s]) / 1000 for r in rows] for s in ("active", "idle_tool", "stale")]
        fig, a = plt.subplots(figsize=(7.2, 2.4))
        a.stackplot(t, *ys, colors=[C["active"], C["idle_tool"], C["stale"]],
                    labels=[LABEL[s] for s in ("active", "idle_tool", "stale")])
        a.set_xlabel("time (s)")
        a.set_ylabel("KV tokens held (k)")
        a.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7)
        a.set_xlim(0, t[-1])
        fig.tight_layout()
        fig.savefig(out / f"{name}.png")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/figures/phase1")
    args = ap.parse_args()
    root, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_m0(root, out)
    if (root / "points.csv").exists():
        rows = points(root)
        fig_e1(rows, out)
        fig_e1_poisson(rows, out)
        fig_e2(rows, out)
        fig_e3_e4(rows, out)
        fig_summary(rows, out)
    fig_e5(root, out)
    fig_pressure(root, out)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
