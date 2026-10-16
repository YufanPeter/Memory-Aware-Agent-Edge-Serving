"""Assembly layer: build every layer from a concrete config and run one experiment point.

This is the only module that knows about all layers, and it only uses their
abstract interfaces plus the registry.
"""
from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from agentserve.agents.base import Agent
from agentserve.core import registry
from agentserve.core.events import Clock, EventLog
from agentserve.core.types import GenerationRequest, TaskResult, TaskSpec
from agentserve.evaluators.base import Evaluator
from agentserve.metrics.system import summarize_run
from agentserve.monitors.base import Monitor, MonitorContext
from agentserve.tasks.base import Task
from agentserve.workloads.mix import build_specs


def _git_rev() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


async def _monitor_loop(m: Monitor, ctx: MonitorContext, events: EventLog) -> None:
    while True:
        try:
            if (d := await m.sample(ctx)) is not None:
                events.emit("monitor", source=m.registry_name, **d)
        except Exception as e:
            events.emit("monitor_error", source=m.registry_name, error=repr(e))
        await asyncio.sleep(m.interval)


async def _warmup(backend: Any, n: int) -> None:
    for i in range(n):
        await backend.generate(GenerationRequest(
            request_id=f"warmup:{i}", task_id=f"warmup:{i}", step=0,
            messages=[{"role": "user", "content": "Reply with OK."}], max_tokens=4,
        ))


async def run_point(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    registry.load_plugins(cfg.get("plugins"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    seed = int(cfg.get("seed", 0))
    run_cfg = cfg.get("run", {})

    clock = Clock()
    events = EventLog(out_dir / "events.jsonl", clock)

    # L0 + L1
    backend_cfg = dict(cfg["backend"])
    backend_cfg.setdefault("log_path", str(out_dir / "backend.log"))
    backend = registry.build("backend", backend_cfg, model=cfg.get("model", {}), clock=clock)
    # L2
    scheduler = registry.build("scheduler", cfg["scheduler"], backend=backend, events=events)
    # L3a: tasks (+ their evaluators)
    tasks: dict[str, Task] = {}
    evaluators: dict[str, list[Evaluator]] = {}
    for name, tcfg in cfg["tasks"].items():
        tcfg = dict(tcfg)
        ev_cfgs = tcfg.pop("evaluators", None)
        tasks[name] = registry.build("task", tcfg)
        evaluators[name] = [registry.build("evaluator", e) for e in (ev_cfgs or tasks[name].default_evaluators)]
    # L3b
    agent = Agent(**cfg["agent"])
    # L4
    wl = cfg["workload"]
    specs = build_specs(wl["mix"], int(wl["n_tasks"]), tasks, seed, wl.get("interleave", True))
    arrival = registry.build("arrival", wl["arrival"])
    monitors: list[Monitor] = [registry.build("monitor", m) for m in cfg.get("monitors", [])]

    task_timeout = float(run_cfg.get("task_timeout", 600))
    tasks_fh = open(out_dir / "tasks.jsonl", "w")

    async def launch(spec: TaskSpec) -> None:
        task = tasks[spec.task_name]
        t_arrive = clock.now()
        events.emit("task_arrive", task_id=spec.task_id, mix=spec.tags.get("mix"))
        await scheduler.admit_task(spec)
        t_start = clock.now()
        events.emit("task_start", task_id=spec.task_id, admission_wait=t_start - t_arrive)
        try:
            result = await asyncio.wait_for(
                agent.run(task.context(spec), task.tools(spec), scheduler, events), task_timeout)
        except asyncio.TimeoutError:
            result = TaskResult(task_id=spec.task_id, status="timeout", error=f">{task_timeout}s")
        except Exception as e:
            result = TaskResult(task_id=spec.task_id, status="error", error=repr(e))
        t_done = clock.now()
        scheduler.task_done(spec, result)
        gold = task.gold(spec.instance)
        scores: dict[str, float] = {}
        for ev in evaluators[spec.task_name]:
            scores.update(ev.score(result, gold))
        events.emit("task_done", task_id=spec.task_id, status=result.status, latency=t_done - t_arrive)
        rec = {
            "task_id": spec.task_id, "task_name": spec.task_name, "instance_id": spec.instance.instance_id,
            "tags": spec.tags, "meta": task.describe(spec.instance),
            "t_arrive": t_arrive, "t_start": t_start, "t_done": t_done,
            "status": result.status, "error": result.error,
            "prediction": result.prediction, "gold": gold, "scores": scores,
            "n_steps": result.n_steps,
            "tool_delay_total": sum(s.tool_delay_s for s in result.steps),
            "steps": [s.__dict__ for s in result.steps],
        }
        tasks_fh.write(json.dumps(rec, default=str) + "\n")
        tasks_fh.flush()

    wall_start = time.time()
    monitor_tasks: list[asyncio.Task] = []
    try:
        await backend.start()
        await _warmup(backend, int(run_cfg.get("warmup_requests", 2)))
        clock.t0 = time.perf_counter()  # t=0 is the start of measured work
        await scheduler.start()
        mctx = MonitorContext(backend=backend, scheduler=scheduler)
        monitor_tasks = [asyncio.create_task(_monitor_loop(m, mctx, events))
                         for m in monitors if m.available(mctx)]
        await arrival.drive(specs, launch, clock, seed)
    finally:
        for t in monitor_tasks:
            t.cancel()
        await asyncio.gather(*monitor_tasks, return_exceptions=True)
        meta = {
            "point": cfg.get("point"), "seed": seed, "wall_start": wall_start,
            "wall_seconds": time.time() - wall_start, "git_rev": _git_rev(),
            "host": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()},
            "backend": backend.meta(), "capabilities": backend.capabilities(),
            "monitors": [m.registry_name for m in monitors], "event_counts": events.counts,
        }
        await scheduler.stop()
        await backend.stop()
        events.close()
        tasks_fh.close()
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    summary = summarize_run(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
