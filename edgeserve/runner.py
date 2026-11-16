"""Replay a frozen workload against llama-server with FIFO admission and fixed
concurrency K (the Phase 1 baseline).

Tasks enter one FIFO queue at their arrival time; K workers each take the next
task and run it to completion. Worker k sends every request with id_slot=k, so a
task's whole conversation stays in one slot's KV cache while it waits for tools.

    uv run python -m edgeserve.runner --workload workloads/smoke.json --k 4 --out results/smoke_k4
    uv run python -m edgeserve.runner --workload workloads/smoke.json --k 2 --mock   # no server needed

Phase 2: `--policy NAME` (a preset of configs/kvm.yaml, overrides via --kvm JSON) puts the KV
manager (edgeserve/kvm.py) between the agents and the server: admission by FIFO ticket, the
dispatch gate, the slot lifecycle, and exact KV accounting in kv_* events. `--k` is then the
fixed K, or K_max for KV admission. `--mock --mock-capacity N` replaces the server with
MockPoolLLM, a bounded KV cell pool with slot actions, so every preset runs on a laptop:

    uv run python -m edgeserve.runner --workload workloads/smoke.json --k 8 --policy ours \
        --mock --mock-capacity 4096 --out results/mock_ours

Without --policy and --mock-capacity the runner behaves exactly as in Phase 1.

Outputs in --out: events.jsonl, tasks.jsonl, meta.json, summary.json.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import platform
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import psutil

from edgeserve.agent import Agent, TaskTrace
from edgeserve.corpus import Corpus
from edgeserve.data import load_split
from edgeserve.eval import exact_match, f1_score, sf_recall
from edgeserve.events import EventLog
from edgeserve.kvm import KVManager, ManagedLLM
from edgeserve.llm import LlamaServerClient, MockLLM, SlotControl
from edgeserve.mockpool import MockPoolLLM
from edgeserve.policies import KVM_CONFIG_PATH, load_policy
from edgeserve.profiles import PROFILES_PATH, load_profiles
from edgeserve.workload import load_workload

LLAMA_CACHE_RAM_DEFAULT = 8192.0   # MiB: llama-server enables its host prompt cache unless told otherwise


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


def server_flag(cmd: str, *flags: str) -> str | None:
    """Value of a llama-server flag in a server_cmd.txt line (last occurrence wins)."""
    toks = cmd.split()
    val = None
    for i, tok in enumerate(toks):
        for f in flags:
            if tok == f and i + 1 < len(toks):
                val = toks[i + 1]
            elif tok.startswith(f + "="):
                val = tok.split("=", 1)[1]
    return val


def cache_ram_mib(cmd: str) -> float:
    v = server_flag(cmd, "--cache-ram", "-cram")
    return float(v) if v is not None else LLAMA_CACHE_RAM_DEFAULT


def kv_status(status: str, error: str | None) -> str:
    """The gate refuses a context that cannot fit (tok(P) >= c - m): a failure of its own kind."""
    return "kv_infeasible" if status == "ctx_overflow" and (error or "").startswith("kv_infeasible") else status


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

    server_cmd = (out / "server_cmd.txt").read_text(encoding="utf-8").strip() \
        if (out / "server_cmd.txt").exists() else None   # written by scripts/sweep.py or start_server.sh
    cfg = None
    if args.policy:
        overrides = json.loads(args.kvm) if args.kvm else {}
        cfg = load_policy(args.policy, overrides, args.kvm_config)
        cfg.k, cfg.pinned, cfg.delay_scale = args.k, args.pin, args.delay_scale
        if cfg.lifecycle != "observe" and not args.pin:
            raise SystemExit(f"policy {args.policy}: the managed and server lifecycles need pinned slots")
        if server_cmd and "slot_dir" not in overrides and (sp := server_flag(server_cmd, "--slot-save-path")):
            cfg.slot_dir = sp
    # --cache-ram with cache_idle_slots saves and clears every idle slot at each launch, and a
    # pinned request to a cleared slot never reloads (f_keep = 0/0 = NaN, sc:1853): spec §5.
    if server_cmd and not args.mock and (args.pin or (cfg and cfg.lifecycle != "observe")):
        if cache_ram_mib(server_cmd) != 0:
            raise SystemExit("refusing a pinned or managed run against a server with --cache-ram != 0 "
                             f"({cache_ram_mib(server_cmd):g} MiB): {server_cmd}")

    llm: Any
    ctl: Any = None
    pool: MockPoolLLM | None = None
    if args.mock and (cfg is not None or args.mock_capacity is not None):
        capacity = args.mock_capacity or (cfg.ctx if cfg and cfg.ctx else None)
        if cfg is not None and not capacity:
            raise SystemExit("--policy with --mock needs --mock-capacity (or ctx in --kvm)")
        slot_dir = out / "mock_slots"
        slot_dir.mkdir(exist_ok=True)
        for f in slot_dir.glob("*.bin"):
            f.unlink()
        pool = MockPoolLLM(mock_script, capacity=capacity, n_slots=args.mock_slots, t_prefill=args.mock_prefill_s,
                           t_decode=args.mock_decode_s, cache_ram_mib=args.mock_cache_ram, slot_dir=slot_dir,
                           op_timeout_s=cfg.op_timeout_s if cfg else 5.0)
        llm = ctl = pool
        if cfg is not None:
            cfg.slot_dir, cfg.cache_ram_mib = str(slot_dir), args.mock_cache_ram
    elif args.mock:
        llm = MockLLM(mock_script, prefill_s_per_token=1e-5, decode_s_per_token=1e-3)
    else:
        llm = LlamaServerClient(args.base_url, seed=args.seed, cache_prompt=not args.no_cache)
        if cfg is not None:
            ctl = SlotControl(args.base_url, op_timeout_s=cfg.op_timeout_s)
            if server_cmd and "cache_ram_mib" not in overrides:   # B2's HOST_CACHED sub-mode needs it
                cfg.cache_ram_mib = cache_ram_mib(server_cmd)
    props = await llm.props()
    total_slots = props.get("total_slots")
    if args.pin and total_slots is not None and args.k > total_slots:
        raise SystemExit(f"K={args.k} but the server has only {total_slots} slots (--parallel)")

    events = EventLog(out / "events.jsonl")
    kvm: KVManager | None = None
    if cfg is not None:
        n_ctx = int((props.get("default_generation_settings") or {}).get("n_ctx") or 0)
        if not cfg.ctx:
            cfg.ctx = n_ctx
        elif n_ctx and n_ctx != cfg.ctx:
            raise SystemExit(f"/props n_ctx {n_ctx} != kvm ctx {cfg.ctx}")
        if not cfg.ctx:
            raise SystemExit("no KV budget: the server reports no n_ctx and --kvm has no ctx")
        if not args.pin:   # B2 observe learns each request's slot from the stream
            llm.return_progress = True
        kvm = KVManager(ctl, events, cfg, llm, profiles=profiles)
        events.listeners.append(kvm.on_event)
    agent = Agent(ManagedLLM(llm, kvm) if kvm else llm, corpus, events, seed=args.seed,
                  delay_scale=args.delay_scale)
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

    tickets = itertools.count()

    async def worker(k: int) -> None:
        slot = k if args.pin else -1
        while (entry := await queue.get()) is not None:
            t, t_arrive = entry
            ticket = next(tickets)   # FIFO order of admission (head-of-line blocking in the manager)
            if kvm is not None:
                slot = await kvm.admit(t["task_id"], t["profile"], ticket, worker=k)
            t_admit = time.monotonic()
            events.emit("task_admit", task_id=t["task_id"], worker=k, slot=slot)
            item = items[t["qid"]]
            trace: TaskTrace | None = None
            try:
                trace = await asyncio.wait_for(agent.run(t["task_id"], item, profiles[t["profile"]], slot),
                                               args.task_timeout)
            except asyncio.TimeoutError:
                trace = TaskTrace(t["task_id"], item.id, t["profile"], item.qtype, slot, status="timeout")
            except Exception as e:   # one broken task must not abort the whole run
                trace = TaskTrace(t["task_id"], item.id, t["profile"], item.qtype, slot, status="error",
                                  error=f"{type(e).__name__}: {e}"[:500])
            finally:
                if kvm is not None:   # shielded: the slot's ledger entry must be settled either way
                    trace_status = trace.status if trace is not None else "cancelled"
                    await asyncio.shield(kvm.release(t["task_id"], trace_status))
            if kvm is not None:
                trace.status = kv_status(trace.status, trace.error)
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
            if kvm is not None:
                row["kv"] = kvm.task_stats(t["task_id"])
            rows.append(row)
            tasks_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            events.emit("task_done", task_id=t["task_id"], status=trace.status, em=row["em"], f1=row["f1"])

    stop = asyncio.Event()
    mon = asyncio.create_task(_monitor(events, None if args.mock else args.base_url,
                                       args.server_pid, args.sample_interval, stop))
    if kvm is not None:
        await kvm.start()
    t0 = time.monotonic()
    await asyncio.gather(arrive(), *(worker(k) for k in range(args.k)))
    makespan = time.monotonic() - t0
    if kvm is not None:
        await kvm.close()
    stop.set()
    await mon
    events.close()
    tasks_f.close()
    await llm.close()
    if ctl is not None and ctl is not llm:
        await ctl.close()

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
    if server_cmd is not None:
        meta["server_cmd"] = server_cmd
    if cfg is not None:
        meta["kvm"] = asdict(cfg) | {"policy": args.policy, "kvm_config": args.kvm_config}
    if pool is not None:
        meta["mock_pool"] = {"capacity": pool.capacity, "n_slots": pool.n_slots, "t_prefill": pool.t_prefill,
                             "t_decode": pool.t_decode, "cache_ram_mib": pool.cache_ram_mib}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8", newline="\n")
    summary = summarize(rows, makespan)
    if kvm is not None:
        summary["kv"] = kvm.summary()
    if pool is not None:
        summary["mock_pool"] = pool.counters()
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
    ap.add_argument("--policy", help="KV manager preset from --kvm-config (Phase 2); none: Phase 1 runner")
    ap.add_argument("--kvm", default="{}", help="JSON overrides of the preset's KVConfig fields")
    ap.add_argument("--kvm-config", default=str(KVM_CONFIG_PATH))
    ap.add_argument("--mock-capacity", type=int, help="with --mock: a bounded MockPoolLLM of this many KV cells")
    ap.add_argument("--mock-slots", type=int, default=64)
    ap.add_argument("--mock-prefill-s", type=float, default=1e-5, help="MockPoolLLM seconds per prefill token")
    ap.add_argument("--mock-decode-s", type=float, default=1e-3, help="MockPoolLLM seconds per decoded token")
    ap.add_argument("--mock-cache-ram", type=float, default=0.0, help="MockPoolLLM --cache-ram emulation (MiB)")
    args = ap.parse_args()
    print(json.dumps(asyncio.run(run(args)), indent=1))


if __name__ == "__main__":
    main()
