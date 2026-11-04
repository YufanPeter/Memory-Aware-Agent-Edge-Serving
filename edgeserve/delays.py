"""Controlled tool latency: the idle gap between an agent's LLM calls.

A delay is drawn from an RNG seeded by (seed, task_id, step, tool), so a task
sees the same delays no matter how the scheduler orders work.
"""
from __future__ import annotations

import math
import random
from typing import Any


def _sample(cfg: dict[str, Any], rng: random.Random) -> float:
    kind = cfg.get("type", "none")
    if kind == "none":
        d = 0.0
    elif kind == "fixed":
        d = float(cfg["seconds"])
    elif kind == "lognormal":   # parameterized by median and sigma of the underlying normal
        d = rng.lognormvariate(math.log(cfg["median"]), cfg.get("sigma", 0.5))
    elif kind == "exponential":
        d = rng.expovariate(1.0 / cfg["mean"])
    elif kind == "uniform":
        d = rng.uniform(cfg["low"], cfg["high"])
    else:
        raise ValueError(f"unknown delay type {kind!r}")
    if "max_s" in cfg:
        d = min(d, cfg["max_s"])
    return d


class DelaySampler:
    def __init__(self, default: dict[str, Any] | None = None,
                 per_tool: dict[str, dict[str, Any]] | None = None, scale: float = 1.0) -> None:
        self.default = default or {"type": "none"}
        self.per_tool = per_tool or {}
        self.scale = scale

    def sample(self, tool: str, seed: int, task_id: str, step: int) -> float:
        rng = random.Random(f"{seed}:{task_id}:{step}:{tool}")
        return self.scale * _sample(self.per_tool.get(tool, self.default), rng)
