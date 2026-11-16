"""KV-pool occupancy over time from a run's events.jsonl (E5; Phase 2 P6).

Three modes, chosen per run:

**Exact (Phase 2, pinned).** Runs with the KV manager (any policy, observe included) log a
`kv_state` event for every slot transition, with the slot's resident tokens after it. Each slot
is then a sequence of (state, tokens) intervals, exact for every series but `active`:

  active     a request is in flight on the slot (ledger state ACTIVE), valued at the slot's size
             after the request (the next kv_state's resident count): cells grow from the reused
             prefix to that count, so this is an upper bound inside the request, as in Phase 1
  idle_tool  the owner is in a tool (IDLE; also a slot dropped to its prefix)
  ready      the owner waits at the gate (READY)
  stale      the task finished, its KV is still resident (STALE)
  op         a slot op holds the cells (SAVING, RESTORING, or UNKNOWN after an op timeout)
  offloaded  tokens saved to the host tier (OFFLOADED): not in the GPU pool
  host       tokens in llama.cpp's own host prompt cache (HOST_CACHED, B2 observe; approximate)

Shares are over the GPU series only. kv_tick (1 Hz) gives the ledger U, pressure and queue
lengths; kv_op the op costs and the stall share (save/restore block the server's main loop);
kv_violation, kv_truncated, kv_exceeded, kv_gate and kv_admit the counters and waits P6 reports.

**Approximate (Phase 2, unpinned: B2).** The same intervals, but request charges wait in an
unassigned bucket until the stream names the slot, and HOST_CACHED moves are inferred from the
launch rule, not observed (spec §3); the host series misses an entry whose slot a later request
reused. Reported, never part of the exact-token·s claim.

**Reconstructed (Phase 1).** llama-server exposes no KV gauge, so occupancy is rebuilt per
slot from the per-request token counts: after a request a slot holds prompt_n + cache_n +
predicted_n tokens, and is active / idle_tool / stale as above. When a task's next request
reuses fewer tokens than its slot held, the server dropped that KV while the slot was idle;
the eviction moment is not observed, so these are upper bounds once evictions happen.

The recompute statistics (evictions, evicted_tokens) come from req_end in both modes: in a
managed run they count the tokens a drop or erase made the owner re-prefill.

    uv run python -m analysis.kv_timeline results/e1_concurrency/k16/rep0 [--bin 0.5]

Writes kv_timeline.csv (time × series tokens) and kv_summary.json into the run directory.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

STATES = ("active", "idle_tool", "stale")
EXACT_STATES = ("active", "idle_tool", "ready", "stale", "op", "other", "offloaded", "host")
GPU_STATES = ("active", "idle_tool", "ready", "stale", "op", "other")
SERIES = {"ACTIVE": "active", "IDLE": "idle_tool", "DROPPED": "idle_tool", "READY": "ready", "STALE": "stale",
          "SAVING": "op", "RESTORING": "op", "UNKNOWN": "op", "OFFLOADED": "offloaded", "HOST_CACHED": "host",
          "FREE": None}
VERSION = 3   # kv_summary.json format; analysis.aggregate recomputes older summaries of Phase 2 runs

Interval = tuple[int, float, float, str, int]   # (slot, t0, t1, series, tokens)


def load_events(run: Path) -> list[dict[str, Any]]:
    with open(run / "events.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def has_exact(events: list[dict[str, Any]]) -> bool:
    return any(ev["type"] == "kv_state" for ev in events)


def approximate(run: Path, events: list[dict[str, Any]]) -> bool:
    """Unpinned observe runs (B2): request charges wait in an unassigned bucket and HOST_CACHED
    moves are inferred, so their breakdown is approximate (spec §3)."""
    meta = run / "meta.json"
    if meta.exists() and (json.loads(meta.read_text(encoding="utf-8")).get("kvm") or {}).get("pinned") is False:
        return True
    return any(ev["type"] == "kv_state" and (ev.get("to") == "HOST_CACHED" or ev.get("reason") == "host_cached")
               for ev in events)


def series_of(state: Any) -> str | None:
    s = str(state).upper()
    return SERIES[s] if s in SERIES else "other"


def run_end(events: list[dict[str, Any]]) -> float:
    done = [ev["t"] for ev in events if ev["type"] == "task_done"]
    return max(done) if done else max((ev["t"] for ev in events), default=0.0)


def recompute_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Follow-up requests on the same slot whose cache_n fell short of the previous prompt."""
    reqs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        if ev["type"] == "req_end" and ev.get("ok"):
            reqs[ev["task_id"]].append(ev)
    evicted_tokens = n_evictions = n_followups = 0
    by_slot: dict[int, int] = defaultdict(int)
    for rs in reqs.values():
        rs.sort(key=lambda r: r["t_start"])
        for r, nxt in zip(rs, rs[1:], strict=False):
            slot = int(r["slot"])
            if int(nxt["slot"]) != slot:
                continue
            n_followups += 1
            # The next prompt starts with this request's prompt plus (most of) its output.
            prompt = int(r["prompt_n"]) + int(r["cache_n"])
            lost = prompt - int(nxt["cache_n"])
            if lost > 0.05 * prompt:
                n_evictions += 1
                evicted_tokens += lost
                by_slot[slot] += 1
    return {"followup_requests": n_followups, "evictions": n_evictions, "evicted_tokens": evicted_tokens,
            "eviction_rate": round(n_evictions / n_followups, 4) if n_followups else None,
            "evictions_by_slot": dict(sorted(by_slot.items()))}


def intervals(events: list[dict[str, Any]]) -> tuple[list[Interval], dict[str, Any]]:
    """Phase 1 reconstruction: (slot, t0, t1, state, tokens) intervals, plus eviction statistics."""
    reqs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    done: dict[str, float] = {}
    for ev in events:
        if ev["type"] == "req_end" and ev.get("ok"):
            reqs[ev["task_id"]].append(ev)
        elif ev["type"] == "task_done":
            done[ev["task_id"]] = ev["t"]

    out: list[Interval] = []
    per_slot: dict[int, list[tuple[float, float, str, int]]] = defaultdict(list)
    for task, rs in reqs.items():
        rs.sort(key=lambda r: r["t_start"])
        for i, r in enumerate(rs):
            slot = int(r["slot"])
            after = int(r["prompt_n"]) + int(r["cache_n"]) + int(r["predicted_n"])
            per_slot[slot].append((r["t_start"], r["t_end"], "active", after))
            if i + 1 < len(rs):
                per_slot[slot].append((r["t_end"], rs[i + 1]["t_start"], "idle_tool", after))
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
    return out, recompute_stats(events)


def exact_intervals(events: list[dict[str, Any]]) -> tuple[list[Interval], dict[str, Any]]:
    """Intervals from kv_state transitions. Open intervals close at the last task_done (what a
    slot holds after every task finished costs nobody anything)."""
    t_end = run_end(events)
    cur: dict[int, tuple[float, str, int, int]] = {}   # slot -> (t0, state, tokens, charge)
    out: list[Interval] = []
    reasons: Counter[str] = Counter()
    purges = purged = 0

    def close(slot: int, t1: float, nxt: dict[str, Any] | None = None) -> None:
        t0, state, tok, charge = cur[slot]
        if state == "ACTIVE":
            # Cells grow from the reused prefix (after the launch-time seq_rm, sc:4355) to the
            # response's count: value the request at that count, Phase 1's per-request bound.
            post = int(nxt.get("resident") or 0) if nxt is not None else 0
            tok = post if post > 0 else max(tok, charge)
        s = series_of(state)
        if s is not None and tok > 0 and min(t1, t_end) > t0:
            out.append((slot, t0, min(t1, t_end), s, tok))

    for ev in sorted((e for e in events if e["type"] == "kv_state"), key=lambda e: e["t"]):
        slot, tok = int(ev["slot"]), int(ev.get("tokens") or 0)
        reasons[str(ev.get("reason"))] += 1
        if ev.get("reason") == "server_purge":
            purges += 1
            purged += max(0, (cur[slot][2] if slot in cur else 0) - tok)
        if slot in cur:
            close(slot, ev["t"], ev)
        cur[slot] = (ev["t"], str(ev.get("to")), tok, int(ev.get("charge") or 0))
    for slot in list(cur):
        close(slot, t_end)
    stats = recompute_stats(events)
    stats.update({"state_reasons": dict(sorted(reasons.items())), "server_purges": purges,
                  "server_purged_tokens": purged})
    return out, stats


def timeline(ivs: list[Interval], bin_s: float, states: tuple[str, ...] = STATES) -> list[dict[str, float]]:
    if not ivs:
        return []
    t_end = max(t1 for _, _, t1, _, _ in ivs)
    n = int(t_end / bin_s) + 1
    grid = [{s: 0.0 for s in states} for _ in range(n)]
    for _, t0, t1, state, tok in ivs:
        b0, b1 = int(t0 / bin_s), min(n - 1, int(t1 / bin_s))
        for b in range(b0, b1 + 1):
            lo, hi = max(t0, b * bin_s), min(t1, (b + 1) * bin_s)
            if hi > lo:
                grid[b][state] += tok * (hi - lo) / bin_s
    return [{"t": round(i * bin_s, 3), **row} for i, row in enumerate(grid)]


def summarize(ivs: list[Interval], grid: list[dict[str, float]], stats: dict[str, Any],
              states: tuple[str, ...] = STATES, gpu: tuple[str, ...] | None = None) -> dict[str, Any]:
    gpu = gpu or states
    tok_s = {s: sum(tok * (t1 - t0) for _, t0, t1, st, tok in ivs if st == s) for s in states}
    total = sum(tok_s[s] for s in gpu)
    totals = sorted(sum(row[s] for s in gpu) for row in grid)
    busy = [row for row in grid if sum(row[s] for s in gpu) > 0]
    out = {
        "token_seconds": {s: round(v) for s, v in tok_s.items()},
        "share": {s: round(tok_s[s] / total, 4) if total else None for s in gpu},
        "peak_tokens": round(totals[-1]) if totals else 0,
        "p95_tokens": round(totals[int(0.95 * (len(totals) - 1))]) if totals else 0,
        "mean_tokens": round(sum(totals) / len(totals)) if totals else 0,
        "mean_idle_share_while_busy": round(sum(r["idle_tool"] / max(1e-9, sum(r[s] for s in gpu))
                                                for r in busy) / len(busy), 4) if busy else None,
        **stats,
    }
    for s in (x for x in states if x not in gpu):   # off-GPU tiers: peak tokens held there
        out[f"peak_{s}_tokens"] = round(max((row[s] for row in grid), default=0.0))
    return out


def _q(xs: list[float], p: float) -> float | None:
    xs = sorted(xs)
    return round(xs[int(p * (len(xs) - 1))], 4) if xs else None


def manager_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Counters, waits, op costs and ledger gauges from the KV manager's events (spec §6.3)."""
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        if ev["type"].startswith("kv_"):
            by[ev["type"]].append(ev)
    if not by:
        return {}
    t0 = min((ev["t"] for ev in events), default=0.0)
    span = max(1e-9, run_end(events) - t0)
    out: dict[str, Any] = {"event_counts": {k: len(v) for k, v in sorted(by.items())}}
    out["violations"] = dict(sorted(Counter(str(v.get("kind")) for v in by["kv_violation"]).items()))
    out["n_violations"] = len(by["kv_violation"])
    out["kv_truncated"] = len(by["kv_truncated"])
    out["kv_exceeded"] = len(by["kv_exceeded"])
    ops: dict[str, dict[str, float]] = {}
    for o in by["kv_op"]:
        d = ops.setdefault(str(o.get("op")), {"n": 0, "ok": 0, "tokens": 0, "freed": 0, "wall_ms": 0.0,
                                               "server_ms": 0.0})
        d["n"] += 1
        d["ok"] += bool(o.get("ok"))
        d["tokens"] += int(o.get("tokens") or 0)
        d["freed"] += int(o.get("freed") or 0)
        d["wall_ms"] += float(o.get("wall_ms") or 0.0)
        d["server_ms"] += float(o.get("server_ms") or 0.0)
    out["ops"] = {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in sorted(ops.items())}
    out["op_reasons"] = dict(sorted(Counter(f"{o.get('op')}:{o.get('reason')}" for o in by["kv_op"]).items()))
    # save and restore hold the server's main loop (E0a stall): every slot waits for them.
    stall_ms = sum(float(o.get("server_ms") or 0.0) for o in by["kv_op"] if o.get("op") in ("save", "restore"))
    out["op_stall_s"] = round(stall_ms / 1000, 3)
    out["op_stall_share"] = round(stall_ms / 1000 / span, 5)
    out["wasted_saves"] = sum(1 for o in by["kv_op"] if o.get("reason") == "wasted_save" or o.get("wasted"))
    for name, kind in (("gate", "kv_gate"), ("admit", "kv_admit")):
        waits = [float(e.get("wait_s") or 0.0) for e in by[kind]]
        out[f"{name}_n"] = len(waits)
        out[f"{name}_wait_mean_s"] = round(statistics.mean(waits), 4) if waits else None
        out[f"{name}_wait_p95_s"] = _q(waits, 0.95)
    for prio in (0, 1):
        waits = [float(e.get("wait_s") or 0.0) for e in by["kv_gate"] if e.get("prio") == prio]
        out[f"gate_wait_p95_s_prio{prio}"] = _q(waits, 0.95)
    out["gate_evictions"] = sum(len(e.get("victims") or []) for e in by["kv_gate"])
    out["decisions"] = len(by["kv_decision"])
    out["expiries"] = len(by["kv_expire"])
    recon = by["kv_recon"]
    out["recon_n"] = len(recon)
    out["recon_mismatch"] = sum(1 for r in recon if r.get("diff"))
    ticks = by["kv_tick"]
    if ticks:
        U = [float(t.get("U") or 0) for t in ticks]
        out["U_mean"] = round(statistics.mean(U))
        out["U_peak"] = round(max(U))
        out["cap"] = ticks[-1].get("cap")
        out["pressure_share"] = round(sum(1 for t in ticks if t.get("pressure")) / len(ticks), 4)
        out["gate_q_mean"] = round(statistics.mean(float(t.get("gate_q") or 0) for t in ticks), 3)
        out["admit_q_mean"] = round(statistics.mean(float(t.get("admit_q") or 0) for t in ticks), 3)
        out["agents_mean"] = round(statistics.mean(float(t.get("agents") or 0) for t in ticks), 3)
        out["agents_max"] = max(int(t.get("agents") or 0) for t in ticks)
        out["host_bytes_peak"] = max(int(t.get("host_bytes") or 0) for t in ticks)
        out["offloaded_tok_peak"] = max(int(t.get("offloaded_tok") or 0) for t in ticks)
    return out


def kvdbg_check(events: list[dict[str, Any]], log: Path, window_s: float = 1.5) -> dict[str, Any]:
    """LLAMA_KV_CACHE_DEBUG validation (spec §3): attention span (pad256(used_max_p1)) over the
    run, and used cells against the ledger. The server's clock is aligned to the runner's by the
    first task launch, and U comes from 1 Hz ticks, so used ≤ U is checked against the largest U
    within ±window_s: approximate. used ≤ cap is exact."""
    from analysis.server_log import find_slot_samples, first_launch_s, pad256
    samples = find_slot_samples(log)
    if not samples:
        return {}
    out: dict[str, Any] = {"ubatches": len(samples),
                           "span_mean": round(statistics.mean(pad256(n) for _, n, _ in samples)),
                           "span_max": max(pad256(n) for _, n, _ in samples),
                           "used_max": max(u for _, _, u in samples)}
    ticks = sorted((float(e["t"]), float(e.get("U") or 0), e.get("cap")) for e in events if e["type"] == "kv_tick")
    first_req = next((e["t"] for e in events if e["type"] == "req_start"), None)
    launch = first_launch_s(log)
    if not ticks or first_req is None or launch is None:
        return out
    cap = ticks[-1][2]
    offset = launch - first_req
    ts = [t for t, _, _ in ticks]
    over_u = 0
    for t_srv, _, used in samples:
        t = t_srv - offset
        lo, hi = bisect.bisect_left(ts, t - window_s), bisect.bisect_right(ts, t + window_s)
        if hi > lo and used > max(u for _, u, _ in ticks[lo:hi]):
            over_u += 1
    out.update({"used_gt_U_approx": over_u, "cap": cap,
                "used_gt_cap": sum(1 for _, _, u in samples if cap is not None and u > cap)})
    return out


def analyze(run: Path, bin_s: float = 0.5) -> dict[str, Any]:
    events = load_events(run)
    exact = has_exact(events)
    if exact:
        ivs, stats = exact_intervals(events)
        states, gpu = EXACT_STATES, GPU_STATES
    else:
        ivs, stats = intervals(events)
        states, gpu = STATES, STATES
    grid = timeline(ivs, bin_s, states)
    with open(run / "kv_timeline.csv", "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t", *states])
        w.writeheader()
        w.writerows({k: round(v, 1) for k, v in row.items()} for row in grid)
    mode = ("approximate" if approximate(run, events) else "exact") if exact else "reconstructed"
    summary = {"timeline_version": VERSION, "mode": mode, **summarize(ivs, grid, stats, states, gpu)}
    if exact:
        summary["active_tokens"] = "post_request_bound"
    if mgr := manager_stats(events):
        summary["manager"] = mgr
    if (run / "server.log").exists() and (dbg := kvdbg_check(events, run / "server.log")):
        summary["kvdbg"] = dbg
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
