"""Reconstruct KV-pool occupancy over time from a run's events.jsonl (E5).

llama-server exposes no KV gauge, so occupancy is rebuilt per slot from the per-request
token counts the runner logs: after a request a slot holds prompt_n + cache_n + predicted_n
tokens. Every slot-second is classified by what its owner is doing:

  active    a request is running on the slot
  idle_tool the owning task is between requests (almost all of it inside a tool call)
  stale     the task finished; its KV stays until the slot's next task overwrites it

Evictions: when a task's next request reuses fewer tokens than its slot held after the
previous request, llama-server dropped that slot's KV while it was idle (it purges idle
slots when the pool needs room). The missing tokens were recomputed.

    uv run python -m analysis.kv_timeline results/e1_concurrency/k16/rep0 [--bin 0.5]

Writes kv_timeline.csv (time × state tokens) and kv_summary.json into the run directory.
These are upper bounds on residency once evictions happen, since the eviction moment is
not observed; kv_summary.json reports how many tokens that affects.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

STATES = ("active", "idle_tool", "stale")


def load_events(run: Path) -> list[dict[str, Any]]:
    with open(run / "events.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def intervals(events: list[dict[str, Any]]) -> tuple[list[tuple[int, float, float, str, int]], dict[str, Any]]:
    """(slot, t0, t1, state, tokens) intervals, plus eviction statistics."""
    reqs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    done: dict[str, float] = {}
    for ev in events:
        if ev["type"] == "req_end" and ev.get("ok"):
            reqs[ev["task_id"]].append(ev)
        elif ev["type"] == "task_done":
            done[ev["task_id"]] = ev["t"]

    out: list[tuple[int, float, float, str, int]] = []
    per_slot: dict[int, list[tuple[float, float, str, int]]] = defaultdict(list)
    evicted_tokens = n_evictions = n_followups = 0
    by_slot: dict[int, int] = defaultdict(int)
    for task, rs in reqs.items():
        rs.sort(key=lambda r: r["t_start"])
        for i, r in enumerate(rs):
            slot = int(r["slot"])
            prompt = int(r["prompt_n"]) + int(r["cache_n"])
            after = prompt + int(r["predicted_n"])
            per_slot[slot].append((r["t_start"], r["t_end"], "active", after))
            if i + 1 < len(rs):
                nxt = rs[i + 1]
                per_slot[slot].append((r["t_end"], nxt["t_start"], "idle_tool", after))
                if int(nxt["slot"]) == slot:
                    n_followups += 1
                    # The next prompt starts with this request's prompt plus (most of) its output.
                    lost = prompt - int(nxt["cache_n"])
                    if lost > 0.05 * prompt:
                        n_evictions += 1
                        evicted_tokens += lost
                        by_slot[slot] += 1
            else:
                end = done.get(task, r["t_end"])
                per_slot[slot].append((r["t_end"], end, "idle_tool", after))
                per_slot[slot].append((end, float("inf"), "stale", after))

    # A stale interval lasts until the slot's next activity.
    for slot, ivs in per_slot.items():
        ivs.sort(key=lambda x: x[0])
        for j, (t0, t1, state, tok) in enumerate(ivs):
            if state == "stale":
                nxt = next((v[0] for v in ivs[j + 1:] if v[0] >= t0), None)
                t1 = nxt if nxt is not None else t0
            if t1 > t0:
                out.append((slot, t0, t1, state, tok))
    stats = {"followup_requests": n_followups, "evictions": n_evictions, "evicted_tokens": evicted_tokens,
             "eviction_rate": round(n_evictions / n_followups, 4) if n_followups else None,
             "evictions_by_slot": dict(sorted(by_slot.items()))}
    return out, stats


def timeline(ivs: list[tuple[int, float, float, str, int]], bin_s: float) -> list[dict[str, float]]:
    if not ivs:
        return []
    t_end = max(t1 for _, _, t1, _, _ in ivs)
    n = int(t_end / bin_s) + 1
    grid = [{s: 0.0 for s in STATES} for _ in range(n)]
    for _, t0, t1, state, tok in ivs:
        b0, b1 = int(t0 / bin_s), min(n - 1, int(t1 / bin_s))
        for b in range(b0, b1 + 1):
            lo, hi = max(t0, b * bin_s), min(t1, (b + 1) * bin_s)
            if hi > lo:
                grid[b][state] += tok * (hi - lo) / bin_s
    return [{"t": round(i * bin_s, 3), **row} for i, row in enumerate(grid)]


def summarize(ivs: list[tuple[int, float, float, str, int]], grid: list[dict[str, float]],
              stats: dict[str, Any]) -> dict[str, Any]:
    tok_s = {s: sum(tok * (t1 - t0) for _, t0, t1, st, tok in ivs if st == s) for s in STATES}
    total = sum(tok_s.values())
    totals = sorted(sum(row[s] for s in STATES) for row in grid)
    busy = [row for row in grid if sum(row[s] for s in STATES) > 0]
    return {
        "token_seconds": {s: round(v) for s, v in tok_s.items()},
        "share": {s: round(tok_s[s] / total, 4) if total else None for s in STATES},
        "peak_tokens": round(totals[-1]) if totals else 0,
        "p95_tokens": round(totals[int(0.95 * (len(totals) - 1))]) if totals else 0,
        "mean_tokens": round(sum(totals) / len(totals)) if totals else 0,
        "mean_idle_share_while_busy": round(sum(r["idle_tool"] / max(1e-9, sum(r[s] for s in STATES))
                                                for r in busy) / len(busy), 4) if busy else None,
        **stats,
    }


def analyze(run: Path, bin_s: float = 0.5) -> dict[str, Any]:
    ivs, stats = intervals(load_events(run))
    grid = timeline(ivs, bin_s)
    with open(run / "kv_timeline.csv", "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t", *STATES])
        w.writeheader()
        w.writerows({k: round(v, 1) for k, v in row.items()} for row in grid)
    summary = summarize(ivs, grid, stats)
    (run / "kv_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--bin", type=float, default=0.5)
    args = ap.parse_args()
    for r in args.runs:
        print(r, json.dumps(analyze(Path(r), args.bin)))


if __name__ == "__main__":
    main()
