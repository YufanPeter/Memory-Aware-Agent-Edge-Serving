"""Collect every run under results/<sweep>/<point>/rep<i>/ into one table.

For each run: summary.json (runner), kv_summary.json (analysis.kv_timeline, computed if
missing), and the server parameters parsed from meta.json's server_cmd. Writes
results/runs.csv (one row per run) and results/points.csv (mean, min and max over repeats).

    uv run python -m analysis.aggregate [results]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

from analysis.kv_timeline import analyze

METRICS = ["completed", "failed", "throughput_tasks_per_min", "goodput_tasks_per_min",
           "steady_throughput_per_min", "steady_goodput_per_min", "latency_p50_s", "latency_p95_s",
           "latency_trim_p50_s", "latency_trim_p95_s", "queue_wait_mean_s", "queue_wait_p95_s", "em", "f1",
           "decode_tokens_mean", "prefill_tokens_total", "cached_tokens_total", "cache_hit_rate", "makespan_s"]
KV = ["peak_tokens", "p95_tokens", "mean_tokens", "mean_idle_share_while_busy", "evictions",
      "evicted_tokens", "eviction_rate"]


def flag(cmd: str, name: str) -> str | None:
    m = re.search(rf"(?:^| ){re.escape(name)} (\S+)", cmd)
    return m.group(1) if m else None


def run_row(run: Path) -> dict[str, Any]:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    kv_path = run / "kv_summary.json"
    kv = json.loads(kv_path.read_text(encoding="utf-8")) if kv_path.exists() else analyze(run)
    cmd = meta.get("server_cmd", "")
    sweep, point, rep = run.parts[-3], run.parts[-2], run.parts[-1]
    row: dict[str, Any] = {
        "sweep": sweep, "point": point, "rep": rep, "k": meta["k"], "ctx": flag(cmd, "-c"),
        "np": flag(cmd, "-np"), "cache_ram": flag(cmd, "--cache-ram"), "pin": meta["pin_slots"],
        "cache_prompt": meta["cache_prompt"], "delay_scale": meta["delay_scale"],
        "workload": Path(meta["workload"]).stem, "n_tasks": summary["n_tasks"],
    }
    row.update({m: summary.get(m) for m in METRICS})
    with open(run / "tasks.jsonl", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f]
    ttft = sorted(s["llm"]["ttft_s"] for t in tasks for s in t["trace"]["steps"] if s["llm"].get("ttft_s") is not None)
    # Service time (admit -> done): in closed-loop runs every task arrives at t=0, so latency is
    # mostly queueing; service time is what one agent experiences once it runs.
    service = sorted(t["service_s"] for t in tasks)
    q = lambda xs, p: round(xs[int(p * (len(xs) - 1))], 4) if xs else None  # noqa: E731
    row.update({"ttft_p50_s": q(ttft, 0.5), "ttft_p95_s": q(ttft, 0.95),
                "service_p50_s": q(service, 0.5), "service_p95_s": q(service, 0.95)})
    row.update({f"kv_{m}": kv.get(m) for m in KV})
    for s, v in (kv.get("share") or {}).items():
        row[f"kv_share_{s}"] = v
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="results")
    args = ap.parse_args()
    root = Path(args.root)
    runs = sorted(p.parent for p in root.glob("*/*/rep*/summary.json"))
    rows = [run_row(r) for r in runs]
    if not rows:
        raise SystemExit(f"no runs under {root}")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(root / "runs.csv", "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["sweep"], r["point"]), []).append(r)
    numeric = [k for k in fields if all(isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)
                                        for r in rows if r.get(k) is not None) and k not in ("k", "n_tasks")]
    points = []
    for (sweep, point), rs in groups.items():
        p: dict[str, Any] = {k: rs[0][k] for k in ("sweep", "point", "k", "ctx", "np", "pin", "cache_prompt",
                                                    "delay_scale", "workload", "n_tasks")}
        p["reps"] = len(rs)
        for k in numeric:
            vals = [r[k] for r in rs if r.get(k) is not None]
            if vals:
                p[k] = round(statistics.mean(vals), 4)
                if len(vals) > 1:
                    p[f"{k}_min"], p[f"{k}_max"] = min(vals), max(vals)
        points.append(p)
    pfields = list(dict.fromkeys(k for p in points for k in p))
    with open(root / "points.csv", "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pfields)
        w.writeheader()
        w.writerows(points)
    print(f"{len(rows)} runs, {len(points)} points -> {root / 'runs.csv'}, {root / 'points.csv'}")


if __name__ == "__main__":
    main()
