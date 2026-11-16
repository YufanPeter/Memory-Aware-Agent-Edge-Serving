"""Pre-registered statistics for Phase 2 (docs/phase2_spec.md §8.2, Metrics).

- Throughput claims: Welch t-test on per-run steady goodput (≥ 3 reps per cell), plus a
  bootstrap CI of Δ over steady-window completions, resampling runs across reps and tasks
  within each run (hierarchical, so both run-to-run and task-level noise enter the CI).
- Noise: the CV of B1's reps under memory pressure, and a B1-vs-B1 null.
- Quality equivalence: TOST on per-question F1 averaged across reps, paired by question, with
  margin 0.05 on dev and 0.03 on test-500; the B1 rep-1-vs-rep-2 null must pass the same margin.
  The pairing is over questions answered in every run on both sides, so a completion gap is not
  read as a quality gap (the failures-as-0 version is kept as tost_f1_failures0).

The Student t distribution is computed here (regularized incomplete beta), since the project
has no scipy dependency.

    uv run --no-sync python -m analysis.stats results/p1_dev/c16k_d4_b1 results/p1_dev/c16k_d4_ours \
        [--margin 0.05] [--boot 4000]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

ANSWERED = ("ok", "forced", "parse_fail")   # the runner's completed statuses (runner.summarize)


# --- Student t -------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t >= 0 else tail


def t_ppf(p: float, df: float) -> float:
    """Inverse of t_cdf by bisection."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --- run-level helpers -----------------------------------------------------------------

def cv(xs: list[float]) -> float | None:
    """Coefficient of variation (sample SD / mean)."""
    if len(xs) < 2 or not statistics.mean(xs):
        return None
    return statistics.stdev(xs) / statistics.mean(xs)


def welch(a: list[float], b: list[float], alpha: float = 0.05) -> dict[str, Any]:
    """Welch t-test of mean(b) − mean(a), two-sided, with its (1 − alpha) CI."""
    if len(a) < 2 or len(b) < 2:
        return {"diff": statistics.mean(b) - statistics.mean(a) if a and b else None, "p": None,
                "note": "need >= 2 runs per side"}
    ma, mb, va, vb = statistics.mean(a), statistics.mean(b), statistics.variance(a), statistics.variance(b)
    se2 = va / len(a) + vb / len(b)
    diff = mb - ma
    if se2 == 0:
        return {"diff": diff, "t": None, "df": None, "p": 0.0 if diff else 1.0, "ci": [diff, diff]}
    t = diff / math.sqrt(se2)
    df = se2 ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    p = 2 * (1 - t_cdf(abs(t), df))
    h = t_ppf(1 - alpha / 2, df) * math.sqrt(se2)
    return {"diff": diff, "t": t, "df": df, "p": p, "ci": [diff - h, diff + h]}


def load_tasks(run: Path) -> list[dict[str, Any]]:
    with open(run / "tasks.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def steady_window(tasks: list[dict[str, Any]]) -> tuple[list[float], float] | None:
    """EM of the completions inside the run's steady window, and the window length: the
    window between the 10th and 90th percentile completion time, as runner.summarize uses."""
    answered = [r for r in tasks if r["status"] in ANSWERED]
    done = sorted(r["arrival_s"] + r["latency_s"] for r in answered)
    if len(done) < 10:
        return None
    lo, hi = done[len(done) // 10], done[(9 * len(done)) // 10]
    if hi <= lo:
        return None
    win = [float(r["em"]) for r in answered if lo <= r["arrival_s"] + r["latency_s"] <= hi]
    return win, hi - lo


def bootstrap_delta(runs_a: list[tuple[list[float], float]], runs_b: list[tuple[list[float], float]],
                    metric: str = "goodput", n_boot: int = 4000, alpha: float = 0.05,
                    seed: int = 0) -> dict[str, Any]:
    """CI of mean steady rate (per min) of b minus a. Each replicate resamples each side's runs
    with replacement, then each sampled run's window completions with replacement; the window
    length is held fixed. metric: goodput (Σ EM) or throughput (count)."""
    rng = np.random.default_rng(seed)

    def rate(win: np.ndarray, w: float) -> float:
        return 60.0 * (win.sum() if metric == "goodput" else len(win)) / w

    def side(runs: list[tuple[list[float], float]]) -> float:
        picks = rng.integers(0, len(runs), len(runs))
        vals = []
        for i in picks:
            em, w = runs[i]
            arr = np.asarray(em, dtype=float)
            vals.append(rate(arr[rng.integers(0, len(arr), len(arr))], w))
        return float(np.mean(vals))

    point = (statistics.mean(rate(np.asarray(e, dtype=float), w) for e, w in runs_b)
             - statistics.mean(rate(np.asarray(e, dtype=float), w) for e, w in runs_a))
    deltas = np.array([side(runs_b) - side(runs_a) for _ in range(n_boot)])
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {"metric": metric, "delta": point, "ci": [float(lo), float(hi)], "n_boot": n_boot}


def per_question(runs: list[Path], key: str = "f1", only: set[str] | None = None) -> dict[str, float]:
    """Per-question metric averaged across reps (keyed by qid). Failed tasks score 0, as in
    the runner's EM/F1 means; `only` restricts it to a set of qids (see answered_in_all)."""
    acc: dict[str, list[float]] = {}
    for run in runs:
        for r in load_tasks(run):
            if only is None or r["qid"] in only:
                acc.setdefault(r["qid"], []).append(float(r[key]))
    return {q: statistics.mean(v) for q, v in acc.items()}


def answered_in_all(runs: list[Path]) -> set[str]:
    """Qids answered (status in ANSWERED) in every one of the runs. A quality-equivalence test
    paired over these compares answers, not completion: with failed tasks scored 0, a baseline
    that fails tasks would look worse on F1 for a reason the completion column already shows."""
    sets = [{r["qid"] for r in load_tasks(run) if r["status"] in ANSWERED} for run in runs]
    return set.intersection(*sets) if sets else set()


def tost_paired(diffs: list[float], margin: float, alpha: float = 0.05) -> dict[str, Any]:
    """Two one-sided t-tests for |mean diff| < margin; equivalent iff the (1 − 2·alpha) CI lies
    inside (−margin, margin)."""
    n = len(diffs)
    if n < 2:
        return {"n": n, "equivalent": None}
    m, sd = statistics.mean(diffs), statistics.stdev(diffs)
    se = sd / math.sqrt(n) if sd > 0 else 1e-12
    h = t_ppf(1 - alpha, n - 1) * se
    p_lo = 1 - t_cdf((m + margin) / se, n - 1)   # H0: diff <= -margin
    p_hi = t_cdf((m - margin) / se, n - 1)       # H0: diff >= +margin
    return {"n": n, "mean": m, "sd": sd, "ci90": [m - h, m + h], "margin": margin,
            "p": max(p_lo, p_hi), "equivalent": (m - h > -margin) and (m + h < margin)}


def paired_diffs(a: dict[str, float], b: dict[str, float]) -> list[float]:
    return [b[q] - a[q] for q in sorted(set(a) & set(b))]


def reps(point: Path) -> list[Path]:
    return sorted(p for p in point.glob("rep*") if (p / "summary.json").exists())


def compare(point_a: Path, point_b: Path, margin: float = 0.05, n_boot: int = 4000) -> dict[str, Any]:
    ra, rb = reps(point_a), reps(point_b)
    if not ra or not rb:
        raise SystemExit(f"no finished reps under {point_a if not ra else point_b}")

    def steady(runs: list[Path], key: str) -> list[float]:
        vals = [json.loads((r / "summary.json").read_text(encoding="utf-8")).get(key) for r in runs]
        return [float(v) for v in vals if v is not None]

    wa = [w for r in ra if (w := steady_window(load_tasks(r)))]
    wb = [w for r in rb if (w := steady_window(load_tasks(r)))]
    ga, gb = steady(ra, "steady_goodput_per_min"), steady(rb, "steady_goodput_per_min")
    both = answered_in_all(ra + rb)
    fa, fb = per_question(ra, only=both), per_question(rb, only=both)
    out: dict[str, Any] = {
        "a": {"point": str(point_a), "reps": len(ra), "steady_goodput": ga, "cv": cv(ga)},
        "b": {"point": str(point_b), "reps": len(rb), "steady_goodput": gb, "cv": cv(gb)},
        "welch_goodput": welch(ga, gb),
        "welch_throughput": welch(steady(ra, "steady_throughput_per_min"), steady(rb, "steady_throughput_per_min")),
        "bootstrap_goodput": bootstrap_delta(wa, wb, "goodput", n_boot) if wa and wb else None,
        "bootstrap_throughput": bootstrap_delta(wa, wb, "throughput", n_boot) if wa and wb else None,
        "tost_f1": tost_paired(paired_diffs(fa, fb), margin),   # over qids answered in every run
        "tost_f1_failures0": tost_paired(paired_diffs(per_question(ra), per_question(rb)), margin),
    }
    if len(ra) >= 2:   # the null: rep 0 vs rep 1 of a must pass the same margin, paired the same way
        null = answered_in_all(ra[:2])
        out["null_tost_f1"] = tost_paired(paired_diffs(per_question(ra[:1], only=null),
                                                       per_question(ra[1:2], only=null)), margin)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="baseline point directory (holds rep*/)")
    ap.add_argument("b", help="compared point directory")
    ap.add_argument("--margin", type=float, default=0.05, help="TOST margin on F1 (0.05 dev, 0.03 test-500)")
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()
    print(json.dumps(compare(Path(args.a), Path(args.b), args.margin, args.boot), indent=1, default=float))


if __name__ == "__main__":
    main()
