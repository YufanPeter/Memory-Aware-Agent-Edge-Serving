"""Per-run metrics beyond summary.json, grouped by the project's topic, memory-aware agent edge serving
(docs/phase2_ideas.md section 0). Computed offline from a run directory (summary.json, tasks.jsonl,
events.jsonl, meta.json); no server or GPU needed.

    uv run python -m analysis.run_metrics results/p3_main16k/r1_l110_kv/rep0 [more runs ...]

memory      KV occupancy mean / peak (tokens and share of the pool), in-flight agents mean / peak, KV each
            in-flight agent holds on its own, KV left by finished tasks, idle-KV share, the shared system prompt (forked tokens), pressure actions
            (KV dropped, gate waits, refusals), selective KV (trims, KV tokens deleted, server time per trim)
recompute   re-prefilled tokens per task, prefill tokens per task, prefix-cache hit rate
agent       task latency split into admission wait and service time, LLM calls per task, forced finishes,
            output tokens per request
edge        GPU memory peak, GPU utilisation mean, llama-server host RSS peak (power is not recorded)

Definitions. The window is from the first admission to the last completion. Token counts are the server's
(llama.cpp's prompt_n, cache_n and predicted_n per request, the trim action's deleted tokens, the fork's
tokens). KV occupancy, managed runs: the slot states of the manager's kv_state events with the tokens the
server reported, a running request counted at the tokens it ends with (prompt + reused + generated), as the
Phase 1 reconstruction does for fixed K (analysis/kv_timeline.py); the manager's books, which hold a running
request at its reserved size, stay the safety check (check_kvm, "books" in the reports). KV per agent = KV held by slots of tasks still in flight
(not the KV finished tasks leave behind until a slot is reused) / mean in-flight agents; under the KV
manager it excludes the shared system prompt (one golden copy), under fixed K every agent holds its own copy.
Re-prefilled tokens (measured): for each follow-up request of a task, the tokens the slot held after the
previous request (prompt_n + cache_n + predicted_n - 1: the last sampled token never enters the KV) minus the
KV tokens selective KV deleted in between, minus the tokens the server reused now (cache_n). Without a loss
this is 0 (or a few tokens at a tokenisation seam); after a drop it is the context recomputed. Text deletion
(--selective text) changes the prompt text without logging the deleted tokens, so for those runs the figure
is an estimate: prefill tokens beyond the text appended since the previous request, chars converted to
tokens at the request's own ratio (checked: within 2% of the measured prefill difference to kv deletion).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

from analysis import kv_timeline

IN_FLIGHT = ("ready", "active", "idle", "waiting")   # kv_state states of a slot whose task is still running


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def _load(run: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    tasks = [json.loads(x) for x in (run / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if x]
    with open(run / "events.jsonl", encoding="utf-8") as f:
        events = [json.loads(x) for x in f if x.strip()]
    return s, meta, tasks, events


def _window(events: list[dict[str, Any]]) -> tuple[float, float]:
    admits = [e["t"] for e in events if e["type"] == "task_admit"]
    dones = [e["t"] for e in events if e["type"] == "task_done"]
    return min(admits), max(dones)


def _time_weighted(steps: list[tuple[float, float]], t0: float, t1: float) -> tuple[float, float]:
    """Mean and peak of a step function given as (time, value from then on), over [t0, t1]."""
    area, peak, cur, last = 0.0, 0.0, 0.0, t0
    for t, v in steps:
        if t > t0:
            tt = min(t, t1)
            area += cur * (tt - last)
            last = tt
        cur = v
        if t0 <= t <= t1:
            peak = max(peak, v)
    area += cur * max(0.0, t1 - last)
    return (area / (t1 - t0) if t1 > t0 else 0.0), peak


def in_flight(events: list[dict[str, Any]], t0: float, t1: float) -> tuple[float, float]:
    n, steps = 0, []
    for e in events:
        if e["type"] == "task_admit":
            n += 1
            steps.append((e["t"], n))
        elif e["type"] == "task_done":
            n -= 1
            steps.append((e["t"], n))
    return _time_weighted(steps, t0, t1)


RESIDENT = ("ready", "active", "idle", "waiting", "stale", "golden")   # edgeserve.kvm.RESIDENT


def managed_kv(events: list[dict[str, Any]], t0: float, t1: float, actual: bool = True) -> dict[str, float | None]:
    """Occupancy, golden (shared prefix) tokens, KV of in-flight and finished tasks and idle-KV share from the
    manager's kv_state events. actual=True counts a running request at the tokens the server reports at its
    end; actual=False reproduces the manager's books (its `resident` field), used to check the replay."""
    ends: dict[str, list[int]] = defaultdict(list)
    for e in events:
        if e["type"] == "req_end":
            ends[e["task_id"]].append((e.get("prompt_n") or 0) + (e.get("cache_n") or 0) + (e.get("predicted_n") or 0))
    ends_it = {k: iter(v) for k, v in ends.items()}
    slots: dict[int, tuple[str, int, int]] = {}
    series: dict[str, list[tuple[float, float]]] = {k: [] for k in ("resident", "golden", "inflight", "stale")}
    tok_s = {"idle": 0.0, "all": 0.0}
    last, mismatch = None, 0

    def own(state: str, tokens: int, shared: int) -> int:
        return tokens if state == "golden" else tokens - min(shared, tokens)

    for e in events:
        if e["type"] != "kv_state":
            continue
        if last is not None and e["t"] > t0:
            dt = min(e["t"], t1) - max(last, t0)
            if dt > 0:
                for state, tokens, shared in slots.values():
                    if state in RESIDENT:
                        tok_s["all"] += own(state, tokens, shared) * dt
                        if state == "idle":
                            tok_s["idle"] += own(state, tokens, shared) * dt
        last = e["t"]
        tokens = e["tokens"]
        if e["to"] == "active" and e.get("task_id") is not None:
            end = next(ends_it.get(e["task_id"], iter(())), None)
            if actual and end:
                tokens = end
        slots[e["slot"]] = (e["to"], tokens, e.get("shared", 0))
        resident = sum(own(*v) for v in slots.values() if v[0] in RESIDENT)
        if not actual and "resident" in e and resident != e["resident"]:
            mismatch += 1
        series["resident"].append((e["t"], resident))
        series["golden"].append((e["t"], sum(tk for st_, tk, _ in slots.values() if st_ == "golden")))
        series["inflight"].append((e["t"], sum(own(*v) for v in slots.values() if v[0] in IN_FLIGHT)))
        series["stale"].append((e["t"], sum(own(*v) for v in slots.values() if v[0] == "stale")))
    mean, peak = _time_weighted(series["resident"], t0, t1)
    return {"kv_mean": mean, "kv_peak": peak, "golden_mean": _time_weighted(series["golden"], t0, t1)[0],
            "inflight_mean": _time_weighted(series["inflight"], t0, t1)[0],
            "stale_mean": _time_weighted(series["stale"], t0, t1)[0],
            "idle_share": tok_s["idle"] / tok_s["all"] if tok_s["all"] else None, "books_mismatch": mismatch}


def recomputed_measured(tasks: list[dict[str, Any]]) -> float:
    """Re-prefilled tokens from the server's counts (see the module docstring), summed over all requests."""
    total = 0.0
    for t in tasks:
        prev = None
        for st_ in (t.get("trace") or {}).get("steps", []):
            llm = st_.get("llm")
            if not llm:
                continue
            if not llm.get("ok"):
                prev = None
                continue
            if prev is not None:
                total += max(0, prev - llm["cache_n"])
            trim = st_.get("trim") or {}
            deleted = trim.get("tokens", 0) if trim.get("mechanism") == "kv" and trim.get("ok") else 0
            prev = llm["prompt_n"] + llm["cache_n"] + llm["predicted_n"] - 1 - deleted
    return total


def recomputed_estimate(tasks: list[dict[str, Any]], events: list[dict[str, Any]]) -> tuple[float, int]:
    """Re-prefilled tokens for text-deletion runs, estimated (see the module docstring); tasks skipped."""
    chars: dict[str, list[int]] = defaultdict(list)
    for e in events:
        if e["type"] == "req_start":
            chars[e["task_id"]].append(e["prompt_chars"])
    total, skipped = 0.0, 0
    for t in tasks:
        tr = t.get("trace") or {}
        steps = [s for s in tr.get("steps", []) if s.get("llm")]
        reqs = chars.get(t["task_id"], [])
        if not steps or len(steps) != len(reqs):
            skipped += 1
            continue
        segs = tr.get("segments") or []
        prefix_chars = len(segs[0][1]) if segs and segs[0][0] == "shared_prefix" else 0
        prev_chars = prev_out = trimmed = None
        for st_, c in zip(steps, reqs):
            llm = st_["llm"]
            n_total = (llm.get("prompt_n") or 0) + (llm.get("cache_n") or 0)
            if not llm.get("ok") or not c or not n_total:
                prev_chars, prev_out, trimmed = c, len(st_.get("output") or ""), (st_.get("trim") or {}).get("chars", 0)
                continue
            per_char = n_total / c
            if prev_chars is None:
                appended = c - prefix_chars
            else:
                appended = c - prev_chars - prev_out + (trimmed or 0)
            total += max(0.0, llm["prompt_n"] - max(0, appended) * per_char)
            prev_chars, prev_out = c, len(st_.get("output") or "")
            trimmed = (st_.get("trim") or {}).get("chars", 0)
    return total, skipped


def run_metrics(run: Path) -> dict[str, Any]:
    run = Path(run)
    s, meta, tasks, events = _load(run)
    km = s.get("kvm") or {}
    pool = km.get("pool_tokens") or meta.get("pool_tokens") \
        or (meta.get("server_props") or {}).get("default_generation_settings", {}).get("n_ctx")
    t0, t1 = _window(events)
    agents_mean, agents_peak = in_flight(events, t0, t1)
    if km:
        kv = managed_kv(events, t0, t1)
    else:
        tl = kv_timeline.analyze(run)
        sh = tl["share"]
        kv = {"kv_mean": tl["mean_tokens"], "kv_peak": tl["peak_tokens"], "golden_mean": 0.0,
              "inflight_mean": tl["mean_tokens"] * (sh["active"] + sh["idle_tool"]),
              "stale_mean": tl["mean_tokens"] * sh["stale"], "idle_share": sh["idle_tool"]}
    own_per_agent = kv["inflight_mean"] / agents_mean if agents_mean else None
    text_deletion = (meta.get("selective") or {}).get("mechanism") == "text"
    if text_deletion:
        recompute, skipped = recomputed_estimate(tasks, events)
    else:
        recompute, skipped = recomputed_measured(tasks), 0
    n = len(tasks) or 1
    samples = [e for e in events if e["type"] == "sample" and t0 <= e["t"] <= t1]
    util = [e["gpu_util"] for e in samples if e.get("gpu_util") is not None]
    gu = km.get("give_ups", {})
    r = lambda x, d=1: round(x, d) if isinstance(x, (int, float)) else x  # noqa: E731
    return {
        # memory
        "pool_tokens": pool,
        "kv_mean": r(kv["kv_mean"], 0), "kv_peak": r(kv["kv_peak"], 0),
        "kv_mean_share": r(kv["kv_mean"] / pool, 3) if pool else None,
        "agents_mean": r(agents_mean, 2), "agents_peak": agents_peak,
        "kv_per_agent": r(own_per_agent, 0), "kv_finished_tasks": r(kv["stale_mean"], 0),
        "idle_share": r(kv["idle_share"], 3),
        "shared_prefix_tokens": r(kv["golden_mean"], 0), "forked_tokens": km.get("forked_tokens", 0),
        "drops": sum(v for k, v in gu.items() if k.endswith(":drop")),
        "gate_waits": km.get("gate_waits", 0), "gate_wait_s": r(km.get("gate_wait_s", 0.0)),
        "refused": km.get("refused", 0),
        "trims": s.get("trims", 0), "trim_kv_tokens": s.get("trim_tokens_total", 0),
        "trim_chars": s.get("trim_chars_total", 0), "trim_server_ms": s.get("trim_server_ms_mean"),
        # recompute
        "recompute_per_task": r(recompute / n, 0), "recompute_measured": not text_deletion,
        "recompute_skipped_tasks": skipped,
        "prefill_per_task": r((s.get("prefill_tokens_total") or 0) / n, 0),
        "cache_hit_rate": s.get("cache_hit_rate"),
        # agent
        "admit_wait_p50": r(_pct([t["queue_wait_s"] for t in tasks], 0.5), 2),
        "admit_wait_p95": r(_pct([t["queue_wait_s"] for t in tasks], 0.95), 2),
        "service_p50": r(_pct([t["service_s"] for t in tasks], 0.5), 2),
        "service_p95": r(_pct([t["service_s"] for t in tasks], 0.95), 2),
        "llm_calls_per_task": r(s.get("n_requests", 0) / n, 2),
        "forced_share": r(s.get("statuses", {}).get("forced", 0) / n, 3),
        "em": s.get("em"), "output_tokens_per_request": r(s.get("decode_tokens_mean")),
        # edge
        "gpu_mem_peak_mib": max((e.get("gpu_mem_mib") or 0 for e in samples), default=None),
        "gpu_util_mean": r(st.mean(util)) if util else None,
        "host_rss_peak_mib": r(max((e.get("server_rss_mib") or 0 for e in samples), default=0), 0) or None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    for run in args.runs:
        print(run, json.dumps(run_metrics(Path(run)), indent=1))


if __name__ == "__main__":
    main()
