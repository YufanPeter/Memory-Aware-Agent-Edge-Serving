"""Phase 2 idea A, stage 1: KV cost and answer quality of each observation-size level.

Reads results/<sweep>/<level>/rep*/tasks.jsonl and prints one row per level:
quality (F1, EM, supporting-fact recall, paired F1 change vs L0 with a bootstrap 95% CI),
agent behaviour (LLM calls, searches, lookups, step-cap share) and KV cost (tokens one search
adds to the context, peak context per task, prefill tokens per task).

    uv run python -m analysis.obs_levels results/p2a_stage1 [--json out.json]

Token counts come from the server's own numbers: the context at a step is prompt_n + cache_n,
so one observation adds ctx[i+1] - ctx[i] - predicted_n[i] tokens (plus ~5 formatting tokens).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path
from typing import Any


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[i]


def task_kv(t: dict[str, Any]) -> dict[str, Any]:
    steps = t["trace"]["steps"]
    ctx = [s["llm"]["prompt_n"] + s["llm"]["cache_n"] for s in steps if s["llm"].get("ok")]
    pred = [s["llm"]["predicted_n"] for s in steps if s["llm"].get("ok")]
    obs = []      # tokens added by each search observation
    for i, s in enumerate(steps[:-1]):
        if s["action"] == "search" and i + 1 < len(ctx):
            obs.append(ctx[i + 1] - ctx[i] - pred[i])
    return {
        "peak": (ctx[-1] + pred[-1]) if ctx else 0,
        "prefill": sum(s["llm"]["prompt_n"] for s in steps if s["llm"].get("ok")),
        "search_obs": obs,
        "searches": sum(s["action"] == "search" for s in steps),
        "lookups": sum(s["action"] == "lookup" for s in steps),
    }


def load_level(d: Path) -> list[list[dict[str, Any]]]:
    reps = []
    for rep in sorted(d.glob("rep*")):
        f = rep / "tasks.jsonl"
        if (rep / "summary.json").exists() and f.exists():
            reps.append([json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line])
    return reps


def per_question_f1(reps: list[list[dict[str, Any]]]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for tasks in reps:
        for t in tasks:
            acc.setdefault(t["qid"], []).append(t["f1"])
    return {q: sum(v) / len(v) for q, v in acc.items()}


def bootstrap_ci(diffs: list[float], n: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def summarize(reps: list[list[dict[str, Any]]], ref: dict[str, float] | None) -> dict[str, Any]:
    tasks = [t for r in reps for t in r]
    kv = [task_kv(t) for t in tasks]
    obs = [x for k in kv for x in k["search_obs"]]
    peaks = [k["peak"] for k in kv]
    out: dict[str, Any] = {
        "reps": len(reps),
        "tasks": len(tasks),
        "completed": sum(t["status"] in ("ok", "forced") for t in tasks),
        "forced_share": sum(t["status"] == "forced" for t in tasks) / len(tasks),
        "f1": st.mean(t["f1"] for t in tasks),
        "f1_per_rep": [round(st.mean(t["f1"] for t in r), 4) for r in reps],
        "em": st.mean(t["em"] for t in tasks),
        "sf_recall": st.mean(t["sf_recall"] for t in tasks),
        "llm_calls": st.mean(t["n_llm_calls"] for t in tasks),
        "searches": st.mean(k["searches"] for k in kv),
        "lookups": st.mean(k["lookups"] for k in kv),
        "obs_tokens_mean": st.mean(obs) if obs else 0.0,
        "obs_tokens_p50": pct(obs, 0.5),
        "obs_tokens_p95": pct(obs, 0.95),
        "peak_mean": st.mean(peaks),
        "peak_p50": pct(peaks, 0.5),
        "peak_p95": pct(peaks, 0.95),
        "prefill_per_task": st.mean(k["prefill"] for k in kv),
    }
    if ref is not None:
        mine = per_question_f1(reps)
        common = sorted(set(mine) & set(ref))
        diffs = [mine[q] - ref[q] for q in common]
        lo, hi = bootstrap_ci(diffs)
        out.update({"df1_vs_l0": sum(diffs) / len(diffs), "df1_ci95": [lo, hi], "paired_n": len(common)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir")
    ap.add_argument("--json")
    args = ap.parse_args()
    root = Path(args.sweep_dir)
    levels = sorted(p.name for p in root.iterdir() if p.is_dir())
    data = {lv: load_level(root / lv) for lv in levels}
    data = {lv: r for lv, r in data.items() if r}
    ref = per_question_f1(data["l0"]) if "l0" in data else None
    rows = {lv: summarize(r, None if lv == "l0" else ref) for lv, r in data.items()}

    print(f"{'level':5} {'reps':>4} {'done':>7} {'F1':>6} {'F1/rep':>15} {'dF1 vs L0 [95% CI]':>24} {'EM':>5} "
          f"{'SFrec':>5} {'calls':>5} {'srch':>4} {'lkup':>4} {'cap%':>4} "
          f"{'obs tok p50/mean/p95':>21} {'peak p50/mean/p95':>19} {'prefill/task':>12}")
    for lv, s in rows.items():
        d = (f"{s['df1_vs_l0']:+.3f} [{s['df1_ci95'][0]:+.3f},{s['df1_ci95'][1]:+.3f}]"
             if "df1_vs_l0" in s else "-")
        print(f"{lv:5} {s['reps']:>4} {s['completed']:>3}/{s['tasks']:<3} {s['f1']:6.3f} "
              f"{str(s['f1_per_rep']):>15} {d:>24} {s['em']:5.2f} {s['sf_recall']:5.2f} "
              f"{s['llm_calls']:5.2f} {s['searches']:4.2f} {s['lookups']:4.2f} {100 * s['forced_share']:4.0f} "
              f"{s['obs_tokens_p50']:>6.0f}/{s['obs_tokens_mean']:>6.0f}/{s['obs_tokens_p95']:>6.0f} "
              f"{s['peak_p50']:>5.0f}/{s['peak_mean']:>5.0f}/{s['peak_p95']:>5.0f} {s['prefill_per_task']:>12.0f}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
