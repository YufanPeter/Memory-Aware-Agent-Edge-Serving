"""Replay a frozen workload against llama-server with FIFO admission and fixed
concurrency K (the Phase 1 baseline).

Tasks enter one FIFO queue at their arrival time; K workers each take the next
task and run it to completion. Worker k sends every request with id_slot=k, so a
task's whole conversation stays in one slot's KV cache while it waits for tools.

    uv run python -m edgeserve.runner --workload workloads/smoke.json --k 4 --out results/smoke_k4
    uv run python -m edgeserve.runner --workload workloads/smoke.json --k 2 --mock   # no server needed

Outputs in --out: events.jsonl, tasks.jsonl, meta.json, summary.json.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import psutil

from edgeserve.agent import Agent, TaskTrace
from edgeserve.corpus import Corpus
from edgeserve.data import load_split
from edgeserve.eval import exact_match, f1_score, sf_recall
from edgeserve.events import EventLog
from edgeserve.llm import LlamaServerClient, MockLLM
from edgeserve.profiles import PROFILES_PATH, load_profiles
from edgeserve.workload import load_workload


def mock_script(prompt: str) -> str:
    """Searches the question's first capitalized phrase once, then finishes with it."""
    if prompt.endswith("finish["):
        return "unknown]"
    task_part = prompt.rsplit("Question: ", 1)[-1]   # skip the few-shot examples
    n_obs = task_part.count("\nObservation ")
    question = task_part.split("<|im_end|>")[0]
    entity = (re.findall(r"[A-Z][\w'-]+(?: [A-Z][\w'-]+)*", question) or ["Wikipedia"])[0]
    if "Context:\n" in prompt or n_obs >= 1:
        return f" I have enough information.\nAction {n_obs + 1}: finish[{entity}]"
    return f" I need to search {entity}.\nAction 1: search[{entity}]\nObservation"


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:   # rsynced copy without .git: tools/mem_aws.sh push writes REVISION
        rev = Path("REVISION")
        return rev.read_text(encoding="utf-8").strip() if rev.exists() else "unknown"


async def _monitor(events: EventLog, base_url: str | None, server_pid: int | None,
                   interval: float, stop: asyncio.Event) -> None:
    """Periodic memory / server-state samples: GPU memory, server RSS, llama-server /metrics."""
    has_smi = shutil.which("nvidia-smi") is not None
    client = httpx.AsyncClient(timeout=2.0) if base_url else None
    while not stop.is_set():
        s: dict[str, Any] = {}
        if has_smi:   # async subprocess: a blocking call here would stall every agent's stream
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            out = (await proc.communicate())[0].decode()
            if out.strip():
                mem, util = out.strip().splitlines()[0].split(",")
                s["gpu_mem_mib"], s["gpu_util"] = float(mem), float(util)
        if server_pid:
            try:
                s["server_rss_mib"] = psutil.Process(server_pid).memory_info().rss / 2**20
            except psutil.Error:
                pass
        if client:
            try:
                r = await client.get(f"{base_url}/metrics")
                for line in r.text.splitlines():
                    if line.startswith("llamacpp:") and " " in line:
                        k, v = line.split(" ", 1)
                        s[k.removeprefix("llamacpp:")] = float(v)
            except Exception:
                pass
        events.emit("sample", **s)
        try:
            await asyncio.wait_for(stop.wait(), interval)
        except asyncio.TimeoutError:
            pass
    if client:
        await client.aclose()


def _quantile(xs: list[float], p: float) -> float | None:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def _req_sum(row: dict[str, Any], key: str) -> int:
    return sum(int(s["llm"].get(key) or 0) for s in row["trace"]["steps"])


def summarize(rows: list[dict[str, Any]], makespan: float) -> dict[str, Any]:
    lat = [r["latency_s"] for r in rows]
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    answered = [r for r in rows if r["status"] in ("ok", "forced", "parse_fail")]
    mean = lambda k: round(statistics.mean(r[k] for r in rows), 4) if rows else None  # noqa: E731

    # Steady state: completions between the 10th and 90th percentile completion time, so the
    # ramp-up and the drain tail (a few long tasks finishing alone) do not set the rate.
    done = sorted(r["arrival_s"] + r["latency_s"] for r in answered)
    steady: dict[str, Any] = {"steady_window_s": None, "steady_throughput_per_min": None,
                              "steady_goodput_per_min": None}
    if len(done) >= 10 and done[(9 * len(done)) // 10] > done[len(done) // 10]:
        lo, hi = done[len(done) // 10], done[(9 * len(done)) // 10]
        win = [r for r in answered if lo <= r["arrival_s"] + r["latency_s"] <= hi]
        steady = {"steady_window_s": round(hi - lo, 3),
                  "steady_throughput_per_min": round(60 * len(win) / (hi - lo), 3),
                  "steady_goodput_per_min": round(60 * sum(r["em"] for r in win) / (hi - lo), 3)}
    # Tail latency without the first and last 10% of arrivals (open-loop warm-up and drain).
    by_arrival = sorted(rows, key=lambda r: r["arrival_s"])
    trim = by_arrival[len(rows) // 10: len(rows) - len(rows) // 10] or by_arrival
    prompt_n = sum(_req_sum(r, "prompt_n") for r in rows)
    cache_n = sum(_req_sum(r, "cache_n") for r in rows)
    return {
        "n_tasks": len(rows), "statuses": statuses, "completed": len(answered),
        "failed": len(rows) - len(answered),
        "makespan_s": round(makespan, 3),
        "throughput_tasks_per_min": round(60 * len(answered) / makespan, 3) if makespan else None,
        "goodput_tasks_per_min": round(60 * sum(r["em"] for r in answered) / makespan, 3) if makespan else None,
        **steady,
        "latency_p50_s": _quantile(lat, 0.5), "latency_p95_s": _quantile(lat, 0.95),
        "latency_p99_s": _quantile(lat, 0.99),
        "latency_trim_p50_s": _quantile([r["latency_s"] for r in trim], 0.5),
        "latency_trim_p95_s": _quantile([r["latency_s"] for r in trim], 0.95),
        "queue_wait_mean_s": mean("queue_wait_s"),
        "queue_wait_p95_s": _quantile([r["queue_wait_s"] for r in rows], 0.95),
        "em": mean("em"), "f1": mean("f1"), "sf_recall": mean("sf_recall"), "steps_mean": mean("n_llm_calls"),
        "decode_tokens_mean": round(statistics.mean(_req_sum(r, "predicted_n") for r in rows), 1) if rows else None,
        "prefill_tokens_total": prompt_n, "cached_tokens_total": cache_n,
        "cache_hit_rate": round(cache_n / (prompt_n + cache_n), 4) if prompt_n + cache_n else None,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wl_text = Path(args.workload).read_text(encoding="utf-8")
    wl = load_workload(args.workload)
    profiles = load_profiles(args.profiles)
    items = {it.id: it for it in load_split(wl["meta"].get("split", "dev"))}
    corpus = Corpus.load(with_bm25=True)   # search suggestions and react_long neighbors

    llm: Any
    if args.mock:
        llm = MockLLM(mock_script, prefill_s_per_token=1e-5, decode_s_per_token=1e-3)
    else:
        llm = LlamaServerClient(args.base_url, seed=args.seed, cache_prompt=not args.no_cache)
    props = await llm.props()
    total_slots = props.get("total_slots")
    if args.pin and total_slots is not None and args.k > total_slots:
        raise SystemExit(f"K={args.k} but the server has only {total_slots} slots (--parallel)")

    events = EventLog(out / "events.jsonl")
    agent = Agent(llm, corpus, events, seed=args.seed, delay_scale=args.delay_scale)
    queue: asyncio.Queue = asyncio.Queue()
    rows: list[dict[str, Any]] = []
    tasks_f = open(out / "tasks.jsonl", "w", buffering=1, encoding="utf-8", newline="\n")

    async def arrive() -> None:
        for t in sorted(wl["tasks"], key=lambda t: t["arrival_s"]):
            delay = t["arrival_s"] - (time.monotonic() - events.t0)
            if delay > 0:
                await asyncio.sleep(delay)
            events.emit("task_arrive", task_id=t["task_id"], profile=t["profile"])
            await queue.put((t, time.monotonic()))
        for _ in range(args.k):
            await queue.put(None)

    async def worker(k: int) -> None:
        slot = k if args.pin else -1
        while (entry := await queue.get()) is not None:
            t, t_arrive = entry
            t_admit = time.monotonic()
            events.emit("task_admit", task_id=t["task_id"], worker=k, slot=slot)
            item = items[t["qid"]]
            try:
                trace = await asyncio.wait_for(agent.run(t["task_id"], item, profiles[t["profile"]], slot),
                                               args.task_timeout)
            except asyncio.TimeoutError:
                trace = TaskTrace(t["task_id"], item.id, t["profile"], item.qtype, slot, status="timeout")
            except Exception as e:   # one broken task must not abort the whole run
                trace = TaskTrace(t["task_id"], item.id, t["profile"], item.qtype, slot, status="error",
                                  error=f"{type(e).__name__}: {e}"[:500])
            t_done = time.monotonic()
            row = {
                "task_id": t["task_id"], "qid": item.id, "profile": t["profile"], "qtype": item.qtype,
                "worker": k, "status": trace.status, "error": trace.error,
                "prediction": trace.prediction, "gold": item.answer,
                "em": exact_match(trace.prediction, item.answer), "f1": f1_score(trace.prediction, item.answer),
                "sf_recall": sf_recall(trace.sf_seen, item.sf_set),
                "arrival_s": events.rel(t_arrive), "queue_wait_s": round(t_admit - t_arrive, 4),
                "latency_s": round(t_done - t_arrive, 4), "service_s": round(t_done - t_admit, 4),
                "n_llm_calls": len(trace.steps), "trace": trace.to_dict(),
            }
            rows.append(row)
            tasks_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            events.emit("task_done", task_id=t["task_id"], status=trace.status, em=row["em"], f1=row["f1"])

    stop = asyncio.Event()
    mon = asyncio.create_task(_monitor(events, None if args.mock else args.base_url,
                                       args.server_pid, args.sample_interval, stop))
    t0 = time.monotonic()
    await asyncio.gather(arrive(), *(worker(k) for k in range(args.k)))
    makespan = time.monotonic() - t0
    stop.set()
    await mon
    events.close()
    tasks_f.close()
    await llm.close()

    meta = {
        "workload": args.workload, "workload_sha": hashlib.sha256(wl_text.encode()).hexdigest()[:16],
        "workload_meta": wl["meta"], "k": args.k, "pin_slots": args.pin, "cache_prompt": not args.no_cache,
        "delay_scale": args.delay_scale, "seed": args.seed, "mock": args.mock,
        "profiles": {n: p.__dict__ for n, p in profiles.items()},
        "server_props": {k: props.get(k) for k in ("build_info", "model_path", "total_slots",
                                                   "default_generation_settings") if k in props},
        "git_rev": _git_rev(), "host": platform.node(), "platform": platform.platform(),
        "started_wall": events.wall0,
    }
    if (out / "server_cmd.txt").exists():   # written by scripts/sweep.py or start_server.sh
        meta["server_cmd"] = (out / "server_cmd.txt").read_text(encoding="utf-8").strip()
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8", newline="\n")
    summary = summarize(rows, makespan)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1, help="fixed concurrency (number of workers)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--profiles", default=str(PROFILES_PATH))
    ap.add_argument("--no-pin", dest="pin", action="store_false", help="let the server choose slots")
    ap.add_argument("--no-cache", action="store_true", help="send cache_prompt=false")
    ap.add_argument("--delay-scale", type=float, default=1.0, help="multiply all tool delays")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-timeout", type=float, default=900.0)
    ap.add_argument("--server-pid", type=int, help="sample this process's RSS")
    ap.add_argument("--sample-interval", type=float, default=1.0)
    ap.add_argument("--mock", action="store_true", help="use the scripted MockLLM instead of a server")
    args = ap.parse_args()
    print(json.dumps(asyncio.run(run(args)), indent=1))


if __name__ == "__main__":
    main()
