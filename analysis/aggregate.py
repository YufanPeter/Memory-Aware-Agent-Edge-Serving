"""Collect every run under results/<sweep>/<point>/rep<i>/ into one table.

For each run: summary.json (runner), kv_summary.json (analysis.kv_timeline, computed if
missing, or recomputed if a Phase 2 run's summary predates the exact timeline), and the server
parameters parsed from meta.json's server_cmd. Writes results/runs.csv (one row per run) and
results/points.csv (mean, min and max over repeats; CV of the steady rates; violation runs).

Phase 2 columns (empty for Phase 1 runs):
  policy, gate, admission, omega, ...  the manager config (meta.json["kvm"])
  mgr_*        the manager's own summary (summary.json["kv"]), flattened
  slog_*       server.log counts (summary.json["server_log"], or counted from server.log);
               slog_purge_violations / slog_overflow_violations judge them by the run's gate
  kvx_*        from the KV events: op costs and stall share, waits, U, pressure, violations
  kvdbg_*      LLAMA_KV_CACHE_DEBUG runs: span and used-vs-ledger checks
  kv_infeasible, kv_infeasible_ids, kv_truncated   counted as failures / separately (spec §2.2)
  kv_exceeded, kv_exceeded_ids   tasks whose prompt reached c - m, listed under every gate (B1
               included), so the overlap of those sets across configurations can be reported
  work_tok_per_s  (prefill + decode tokens) / makespan, with tokens per task, so trajectory
               drift between configurations is visible next to tasks/min
  recompute_tokens  re-prefilled tokens from the traces, one estimator for every run
  gave_up_offloaded / _dropped / _erased  victims by action, from the per-task manager counters

Phase 3 (TRIM) columns. Requests of kind 'trim' (eager TRIM's prefill-only requests) are server
work, so prefill_per_task and work_tok_per_s include them, but they are not agent steps:
decode, TTFT and the per-request hit rate leave them out. A request whose prompt was compacted
(trace step `trimmed_obs`, the eager trim request or the lazy next step) loses the cached
tail on purpose, so recompute_tokens skips it and trim_lost_tokens counts it instead:
  trim_requests_per_task, trim_prefill_per_task   the eager requests and their prefill
  trim_lost_tokens   cells dropped or re-prefilled at a compaction (removed + recomputed); in
               lazy mode a manager eviction in the same idle period lands here too
  context_mean, context_peak_mean   prompt tokens per agent request, and the per-task peak of
               the slot's cells (Q1: how much KV a task needs), for every run

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

from analysis import server_log
from analysis.kv_timeline import VERSION, analyze

METRICS = ["completed", "failed", "throughput_tasks_per_min", "goodput_tasks_per_min",
           "steady_throughput_per_min", "steady_goodput_per_min", "latency_p50_s", "latency_p95_s",
           "latency_trim_p50_s", "latency_trim_p95_s", "queue_wait_mean_s", "queue_wait_p95_s", "em", "f1",
           "decode_tokens_mean", "prefill_tokens_total", "cached_tokens_total", "cache_hit_rate", "makespan_s"]
KV = ["peak_tokens", "p95_tokens", "mean_tokens", "mean_idle_share_while_busy", "evictions",
      "evicted_tokens", "eviction_rate"]


KVM_FIELDS = ["policy", "lifecycle", "gate", "admission", "omega", "theta", "timing", "ttl_x", "delay_model",
              "victim", "actions", "host_frac", "prior_delay_scale", "margin"]
MANAGER_KEYS = ["n_violations", "kv_truncated", "op_stall_s", "op_stall_share", "wasted_saves", "gate_n",
                "gate_wait_mean_s", "gate_wait_p95_s", "gate_wait_p95_s_prio0", "gate_wait_p95_s_prio1",
                "admit_wait_mean_s", "admit_wait_p95_s", "gate_evictions", "decisions", "expiries", "recon_n",
                "recon_mismatch", "U_mean", "U_peak", "cap", "pressure_share", "gate_q_mean", "admit_q_mean",
                "agents_mean", "agents_max", "host_bytes_peak", "offloaded_tok_peak"]
RATE_CV = ["steady_throughput_per_min", "steady_goodput_per_min", "throughput_tasks_per_min",
           "goodput_tasks_per_min"]


def flag(cmd: str, name: str) -> str | None:
    m = re.search(rf"(?:^| ){re.escape(name)} (\S+)", cmd)
    return m.group(1) if m else None


def flatten(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Scalars of d (and of its nested dicts, one level down) as prefix_key columns."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float, str, bool)) or vv is None:
                    out[f"{prefix}{k}_{kk}"] = vv
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[f"{prefix}{k}"] = v
    return out


def kv_summary(run: Path, meta: dict[str, Any]) -> dict[str, Any]:
    path = run / "kv_summary.json"
    if path.exists():
        kv = json.loads(path.read_text(encoding="utf-8"))
        if "kvm" not in meta or kv.get("timeline_version", 1) >= VERSION:
            return kv
    return analyze(run)


def server_log_row(run: Path, summary: dict[str, Any], gate: str | None) -> dict[str, Any]:
    sl = summary.get("server_log") or server_log.count(run / "server.log")
    if not sl:
        return {}
    sl = server_log.judge(dict(sl), gate)
    sl.pop("gate", None)
    sl.pop("kvdbg", None)   # older summaries: kv_summary's kvdbg_* columns are the single source
    return flatten(sl, "slog_")


def is_trim(step: dict[str, Any]) -> bool:
    """Eager TRIM's prefill-only request: server work, not an agent step."""
    return step.get("kind") == "trim"


def _tok(llm: dict[str, Any], *keys: str) -> int:
    return sum(int(llm.get(k) or 0) for k in keys)


def work_row(tasks: list[dict[str, Any]], makespan: float | None) -> dict[str, Any]:
    """Work-normalized throughput: tokens the server computed per second, and per task.
    Prefill counts every request (TRIM's included); decode, the hit rate and the context
    sizes count the agent's own steps."""
    all_steps = [s for t in tasks for s in t["trace"]["steps"]]
    agent = [s["llm"] for s in all_steps if not is_trim(s)]
    trim = [s["llm"] for s in all_steps if is_trim(s)]
    pre = sum(_tok(s, "prompt_n") for s in agent + trim)
    dec = sum(_tok(s, "predicted_n") for s in agent)
    hits = [_tok(s, "cache_n") / _tok(s, "prompt_n", "cache_n")
            for s in agent if s.get("ok") and _tok(s, "prompt_n", "cache_n") > 0]
    final, peaks = [], []
    for t in tasks:
        own = [s["llm"] for s in t["trace"]["steps"] if not is_trim(s)]
        if own and own[-1].get("ok"):
            final.append(_tok(own[-1], "prompt_n", "cache_n", "predicted_n"))
        cells = [_tok(s["llm"], "prompt_n", "cache_n", "predicted_n") - 1
                 for s in t["trace"]["steps"] if s["llm"].get("ok")]
        if cells:
            peaks.append(max(cells))
    ctx = [_tok(s, "prompt_n", "cache_n") for s in agent if s.get("ok")]
    n = len(tasks) or 1
    return {"work_tok_per_s": round((pre + dec) / makespan, 1) if makespan else None,
            "prefill_tok_per_s": round(pre / makespan, 1) if makespan else None,
            "decode_tok_per_s": round(dec / makespan, 1) if makespan else None,
            "prefill_per_task": round(pre / n, 1), "decode_per_task": round(dec / n, 1),
            "final_context_mean": round(statistics.mean(final), 1) if final else None,
            "req_hit_rate_mean": round(statistics.mean(hits), 4) if hits else None,
            "context_mean": round(statistics.mean(ctx), 1) if ctx else None,
            "context_peak_mean": round(statistics.mean(peaks), 1) if peaks else None,
            "trim_requests_per_task": round(len(trim) / n, 3),
            "trim_prefill_per_task": round(sum(_tok(s, "prompt_n") for s in trim) / n, 1)}


def recompute_split(tasks: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    """(recompute, trim_lost) from the traces: for each pair of consecutive ok requests of a
    task, the cells the slot held after the first (prompt + output - 1) minus what the next one
    found cached. Pairs whose second request carries a compacted prompt (step `trimmed_obs`)
    lost that tail by design and go to trim_lost; None when the runs have no such pair."""
    lost, trim_lost, seen, seen_trim = 0, 0, False, False
    for t in tasks:
        prev = None
        for s in (t.get("trace") or {}).get("steps") or []:
            llm = s.get("llm") or {}
            if not llm.get("ok"):
                continue
            seen = True
            if prev is not None:
                d = max(0, prev - int(llm.get("cache_n") or 0))
                if s.get("trimmed_obs") is not None:
                    trim_lost, seen_trim = trim_lost + d, True
                else:
                    lost += d
            prev = int(llm.get("prompt_n") or 0) + int(llm.get("cache_n") or 0) + int(llm.get("predicted_n") or 0) - 1
    return (lost if seen else None), (trim_lost if seen_trim else None)


def recompute_tokens(tasks: list[dict[str, Any]]) -> int | None:
    """Prompt tokens re-prefilled because cached KV was lost, from the traces alone: for each
    pair of consecutive ok requests of a task, the cells the slot held after the first
    (prompt + output - 1) minus what the next one found cached. It is the manager's own count
    (kvm.on_req_end; equal on every managed run without TRIM, whose compactions the manager also
    counts) but needs no manager, so B1, B2 and ours share one estimator. kv_timeline's
    kv_evicted_tokens is not used here: it skips requests that moved slot (most of unpinned B2's
    losses) and losses under 5% of the prompt. TRIM compactions are excluded (recompute_split)."""
    return recompute_split(tasks)[0]


def give_up_row(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Victims by action, summed from the manager's per-task counters (tasks.jsonl kv.offloaded /
    dropped / erased). summary.kv.evictions is not this: it counts every eviction op started,
    including wasted saves and clean-ups of stale slots. Empty for runs without a manager."""
    kvs = [t["kv"] for t in tasks if isinstance(t.get("kv"), dict)]
    if not kvs:
        return {}
    return {f"gave_up_{a}": sum(int(k.get(a) or 0) for k in kvs) for a in ("offloaded", "dropped", "erased")}


def run_row(run: Path) -> dict[str, Any]:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    kv = kv_summary(run, meta)
    kvm = meta.get("kvm") or {}
    cmd = meta.get("server_cmd", "")
    sweep, point, rep = run.parts[-3], run.parts[-2], run.parts[-1]
    row: dict[str, Any] = {
        "sweep": sweep, "point": point, "rep": rep, "k": meta["k"],
        # mock runs have no server command: the pool's capacity is the budget
        "ctx": flag(cmd, "-c") or (meta.get("mock_pool") or {}).get("capacity"),
        "np": flag(cmd, "-np"), "cache_ram": flag(cmd, "--cache-ram"), "pin": meta["pin_slots"],
        "cache_prompt": meta["cache_prompt"], "delay_scale": meta["delay_scale"],
        "workload": Path(meta["workload"]).stem, "n_tasks": summary["n_tasks"],
        "unified": "--no-kv-unified" not in cmd if cmd else None,
    }
    row.update({f: kvm.get(f) for f in KVM_FIELDS})
    row.update({m: summary.get(m) for m in METRICS})
    statuses = summary.get("statuses", {})
    row.update({f"status_{s}": n for s, n in sorted(statuses.items())})
    with open(run / "tasks.jsonl", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f]
    ttft = sorted(s["llm"]["ttft_s"] for t in tasks for s in t["trace"]["steps"]
                  if s["llm"].get("ttft_s") is not None and not is_trim(s))
    # Service time (admit -> done): in closed-loop runs every task arrives at t=0, so latency is
    # mostly queueing; service time is what one agent experiences once it runs.
    service = sorted(t["service_s"] for t in tasks)
    q = lambda xs, p: round(xs[int(p * (len(xs) - 1))], 4) if xs else None  # noqa: E731
    row.update({"ttft_p50_s": q(ttft, 0.5), "ttft_p95_s": q(ttft, 0.95),
                "service_p50_s": q(service, 0.5), "service_p95_s": q(service, 0.95)})
    infeasible = sorted(t["task_id"] for t in tasks if t["status"] == "kv_infeasible")
    mgr = kv.get("manager") or {}
    row.update({"kv_infeasible": len(infeasible), "kv_infeasible_ids": ";".join(infeasible),
                "kv_truncated": mgr.get("kv_truncated")})
    mgr_sum = summary.get("kv") if isinstance(summary.get("kv"), dict) else {}
    if "kv_exceeded" in mgr_sum or any("exceeded" in (t.get("kv") or {}) for t in tasks):
        over = sorted(mgr_sum.get("kv_exceeded")
                      or [t["task_id"] for t in tasks if (t.get("kv") or {}).get("exceeded")])
        row.update({"kv_exceeded": len(over), "kv_exceeded_ids": ";".join(over)})
    row.update(work_row(tasks, summary.get("makespan_s")))
    row["recompute_tokens"], trim_lost = recompute_split(tasks)
    if trim_lost is not None or meta.get("trim", "none") != "none":
        tr = summary.get("trim") or {}
        row.update({"trim": meta.get("trim"), "trim_eager": meta.get("trim_eager"), "trim_lost_tokens": trim_lost,
                    "trim_kept_char_share": tr.get("kept_char_share"), "trim_sf_kept_share": tr.get("sf_kept_share")})
    row.update(give_up_row(tasks))
    row.update({f"kv_{m}": kv.get(m) for m in KV})
    row["kv_mode"] = kv.get("mode", "reconstructed")
    for s, v in (kv.get("share") or {}).items():
        row[f"kv_share_{s}"] = v
    for k, v in kv.items():
        if k.startswith("peak_") and k.endswith("_tokens") and k != "peak_tokens":
            row[f"kv_{k}"] = v
    if isinstance(summary.get("kv"), dict):
        row.update(flatten(summary["kv"], "mgr_"))
    row.update(server_log_row(run, summary, kvm.get("gate")))
    if mgr:
        row.update({f"kvx_{k}": mgr.get(k) for k in MANAGER_KEYS})
        row.update({f"kvx_violation_{k}": n for k, n in (mgr.get("violations") or {}).items()})
        for op, d in (mgr.get("ops") or {}).items():
            row.update({f"kvx_op_{op}_n": d.get("n"), f"kvx_op_{op}_server_ms": d.get("server_ms"),
                        f"kvx_op_{op}_wall_ms": d.get("wall_ms")})
    if dbg := kv.get("kvdbg"):
        row.update({f"kvdbg_{k}": v for k, v in dbg.items()})
    row["violation_run"] = bool((mgr.get("n_violations") or 0) or row.get("slog_purge_violations")
                                or row.get("slog_overflow_violations"))
    return row


def point_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["sweep"], r["point"]), []).append(r)
    numeric = [k for k in fields if all(isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)
                                        for r in rows if r.get(k) is not None) and k not in ("k", "n_tasks")]
    points = []
    for rs in groups.values():
        p: dict[str, Any] = {k: rs[0].get(k) for k in ("sweep", "point", "k", "ctx", "np", "cache_ram", "pin",
                                                        "cache_prompt", "delay_scale", "workload", "n_tasks",
                                                        "unified", *KVM_FIELDS)}
        p["reps"] = len(rs)
        p["violation_runs"] = sum(1 for r in rs if r.get("violation_run"))
        ids = sorted({i for r in rs for i in (r.get("kv_infeasible_ids") or "").split(";") if i})
        p["kv_infeasible_ids_union"] = ";".join(ids)
        if any("kv_exceeded_ids" in r for r in rs):
            ex = {i for r in rs for i in (r.get("kv_exceeded_ids") or "").split(";") if i}
            p["kv_exceeded_ids_union"] = ";".join(sorted(ex))
        for k in numeric:
            vals = [r[k] for r in rs if r.get(k) is not None]
            if vals:
                p[k] = round(statistics.mean(vals), 4)
                if len(vals) > 1:
                    p[f"{k}_min"], p[f"{k}_max"] = min(vals), max(vals)
                    if k in RATE_CV and statistics.mean(vals):
                        p[f"{k}_cv"] = round(statistics.stdev(vals) / statistics.mean(vals), 4)
        points.append(p)
    return points


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

    points = point_rows(rows, fields)
    pfields = list(dict.fromkeys(k for p in points for k in p))
    with open(root / "points.csv", "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pfields)
        w.writeheader()
        w.writerows(points)
    print(f"{len(rows)} runs, {len(points)} points -> {root / 'runs.csv'}, {root / 'points.csv'}")


if __name__ == "__main__":
    main()
