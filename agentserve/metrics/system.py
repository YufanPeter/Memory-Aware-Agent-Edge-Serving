"""System metrics computed offline from a run directory (events.jsonl + tasks.jsonl).
Independent of which backend/scheduler/task produced the run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FINISHED = ("ok", "max_steps")      # task produced an answer (possibly forced)
FAILED = ("llm_error", "timeout", "error")


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    with open(path) as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def load_run(run_dir: Path | str) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    ev = read_jsonl(run_dir / "events.jsonl")
    tasks = read_jsonl(run_dir / "tasks.jsonl")
    req = ev[ev.kind == "request_done"].copy() if not ev.empty else pd.DataFrame()
    mon = ev[ev.kind == "monitor"].copy() if not ev.empty else pd.DataFrame()
    if not req.empty:
        req["queue_wait"] = req.t_dispatch - req.t_enqueue
        req["exec_time"] = req.t_done - req.t_dispatch
        req["latency"] = req.t_done - req.t_enqueue
        req["ttft"] = req.t_first_token - req.t_send
    if not tasks.empty:
        tasks["latency"] = tasks.t_done - tasks.t_arrive
        tasks["admission_wait"] = tasks.t_start - tasks.t_arrive
        tasks["mix"] = tasks.tags.map(lambda t: t.get("mix"))
        for k in ("em", "f1"):
            tasks[k] = tasks.scores.map(lambda s, k=k: s.get(k, np.nan))
        if not req.empty:
            per = req.groupby("task_id").agg(
                n_requests=("request_id", "count"), queue_total=("queue_wait", "sum"),
                exec_total=("exec_time", "sum"), max_prompt_tokens=("prompt_tokens", "max"),
                completion_tokens=("completion_tokens", "sum"))
            tasks = tasks.merge(per, left_on="task_id", right_index=True, how="left")
            # Slowdown vs. an idle system: time a task would take with zero queueing.
            service = tasks.exec_total.fillna(0) + tasks.tool_delay_total.fillna(0)
            tasks["slowdown"] = tasks.latency / service.where(service > 0)
    return {"events": ev, "tasks": tasks, "requests": req, "monitor": mon}


def _pct(s: pd.Series, prefix: str) -> dict[str, float]:
    s = s.dropna()
    if s.empty:
        return {}
    return {f"{prefix}_mean": s.mean(), f"{prefix}_p50": s.quantile(0.5),
            f"{prefix}_p95": s.quantile(0.95), f"{prefix}_p99": s.quantile(0.99), f"{prefix}_max": s.max()}


def summarize_run(run_dir: Path | str) -> dict[str, Any]:
    d = load_run(run_dir)
    tasks, req, mon = d["tasks"], d["requests"], d["monitor"]
    out: dict[str, Any] = {}
    if tasks.empty:
        return {"n_tasks": 0}

    fin = tasks[tasks.status.isin(FINISHED)]
    makespan = tasks.t_done.max() - tasks.t_arrive.min()
    out.update(
        n_tasks=len(tasks), n_finished=len(fin), n_ok=int((tasks.status == "ok").sum()),
        n_max_steps=int((tasks.status == "max_steps").sum()),
        n_failed=int(tasks.status.isin(FAILED).sum()),
        failure_rate=float(tasks.status.isin(FAILED).mean()),
        makespan_s=makespan,
        throughput_tasks_per_min=len(fin) / makespan * 60 if makespan > 0 else np.nan,
        em=tasks.em.fillna(0).mean(), f1=tasks.f1.fillna(0).mean(),
        steps_mean=tasks.n_steps.mean(),
    )
    out.update(_pct(fin.latency, "task_latency"))
    out.update(_pct(fin.get("queue_total", pd.Series(dtype=float)), "task_queue"))
    out.update(_pct(tasks.admission_wait, "admission_wait"))
    out.update(_pct(fin.get("slowdown", pd.Series(dtype=float)), "slowdown"))

    if not req.empty:
        ok = req[req.status == "ok"]
        out.update(
            n_requests=len(req), request_failures=int((req.status != "ok").sum()),
            prompt_tokens_total=int(req.prompt_tokens.sum()),
            cached_tokens_total=int(req.cached_tokens.sum()),
            completion_tokens_total=int(req.completion_tokens.sum()),
            prefix_cache_hit_ratio=req.cached_tokens.sum() / max(1, req.prompt_tokens.sum()),
            decode_tok_per_s=ok.completion_tokens.sum() / makespan if makespan > 0 else np.nan,
            prompt_tok_per_s=(ok.prompt_tokens - ok.cached_tokens).sum() / makespan if makespan > 0 else np.nan,
        )
        for col, name in (("preemptions", "preemptions_total"), ("preempt_wait", "preempt_wait_total_s"),
                          ("recompute_tokens", "recompute_tokens_total"), ("reuse_miss_tokens", "reuse_miss_tokens_total"),
                          ("n_chunks", "chunks_total")):
            if col in req:
                out[name] = float(req[col].fillna(0).sum())
        out["ctx_overflow_failures"] = int((req.status == "ctx_overflow").sum())
        out.update(_pct(ok.latency, "req_latency"))
        out.update(_pct(ok.queue_wait, "req_queue"))
        out.update(_pct(ok.ttft, "req_ttft"))
        out.update(_pct(ok.prompt_tokens, "req_prompt_tokens"))

    if not mon.empty:
        for col in ("phys_footprint_bytes", "rss_bytes", "gpu_mem_used_bytes", "sys_used_bytes"):
            if col in mon:
                s = mon[col].dropna()
                if not s.empty:
                    out[f"peak_{col.removesuffix('_bytes')}_mb"] = s.max() / 2**20
                    out[f"mean_{col.removesuffix('_bytes')}_mb"] = s.mean() / 2**20
        for col in ("queue_len", "inflight", "active_tasks", "requests_deferred"):
            if col in mon and not mon[col].dropna().empty:
                out[f"max_{col}"] = float(mon[col].max())
                out[f"mean_{col}"] = float(mon[col].mean())

    ev = d["events"]
    if not ev.empty:
        ch = ev[ev.kind == "chunk_done"]
        if not ch.empty:  # actual prefill work done by the backend, incl. recomputation
            out["prefill_tokens_total"] = float((ch.prompt_tokens - ch.cached_tokens).clip(lower=0).sum())
        evict = ev[ev.kind == "slot_evict"]
        for mode in ("erase", "swap", "stale"):
            out[f"evictions_{mode}"] = int((evict["mode"] == mode).sum()) if not evict.empty else 0
        restores = ev[ev.kind == "slot_restore"]
        out["slot_restores"] = len(restores)
        out["slot_op_errors"] = int((ev.kind == "slot_op_error").sum())
        out["slot_ops_ms_total"] = float(evict["ms"].sum() if not evict.empty else 0) + float(
            restores["ms"].sum() if not restores.empty else 0)
        st = ev[ev.kind == "scheduler_stats"]
        if not st.empty:
            row = st.iloc[-1]
            out["policy"] = row.get("policy")
            out["policy_overhead_ms"] = float(row.get("policy_overhead_s", 0)) * 1e3
            out["policy_overhead_us_per_decision"] = float(row.get("policy_overhead_s", 0)) * 1e6 / max(1, row.get("decisions", 1))
            if pd.notna(row.get("blocked_decisions")):
                out["mem_blocked_decisions"] = float(row["blocked_decisions"])
    if not mon.empty:
        for col in ("kv_used_est", "kv_active_proj", "paused", "in_tool", "server_purges", "kv_tokens_used"):
            if col in mon and not mon[col].dropna().empty:
                out[f"max_{col}"] = float(mon[col].max())

    from agentserve.metrics.fairness import by_group
    out["by_mix"] = by_group(tasks, "mix")
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in out.items()}
