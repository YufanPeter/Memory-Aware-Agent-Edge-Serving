"""Workload mix: *what* arrives. Combines any registered tasks with tag variants.

    workload:
      n_tasks: 40
      arrival: {type: poisson, rate: 0.5}
      mix:
        - {task: hotpotqa, weight: 0.5, tags: {context: short, max_tokens: 128}}
        - {task: hotpotqa, weight: 0.5, tags: {context: long,  max_tokens: 512}}

Sampling is deterministic in `seed`; instances are drawn without replacement
per task so repeated runs with the same seed see identical work.
"""
from __future__ import annotations

import random
from typing import Any

from agentserve.core.types import TaskInstance, TaskSpec
from agentserve.tasks.base import Task


def build_specs(mix: list[dict[str, Any]], n_tasks: int, tasks: dict[str, Task],
                seed: int, interleave: bool = True) -> list[TaskSpec]:
    rng = random.Random(f"mix:{seed}")
    weights = [float(c.get("weight", 1.0)) for c in mix]
    # Exact proportions (largest remainder) rather than i.i.d. draws -> less variance across seeds.
    raw = [w / sum(weights) * n_tasks for w in weights]
    counts = [int(r) for r in raw]
    for i in sorted(range(len(mix)), key=lambda i: raw[i] - counts[i], reverse=True)[: n_tasks - sum(counts)]:
        counts[i] += 1

    pools: dict[str, list[TaskInstance]] = {}
    for name in {c["task"] for c in mix}:
        pool = list(tasks[name].load())
        random.Random(f"pool:{seed}:{name}").shuffle(pool)
        pools[name] = pool

    labelled: list[tuple[int, dict[str, Any]]] = [(ci, mix[ci]) for ci, k in enumerate(counts) for _ in range(k)]
    if interleave:
        rng.shuffle(labelled)
    specs = []
    for idx, (ci, comp) in enumerate(labelled):
        pool = pools[comp["task"]]
        if not pool:
            raise ValueError(f"task {comp['task']!r} has fewer instances than requested")
        inst = pool.pop()
        tags = {"mix": comp.get("name", f"c{ci}"), "seed": seed, **comp.get("tags", {})}
        specs.append(TaskSpec(task_id=f"t{idx:04d}", task_name=comp["task"], instance=inst, tags=tags))
    return specs
