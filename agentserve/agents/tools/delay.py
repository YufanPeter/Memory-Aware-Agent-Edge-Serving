"""Controlled tool latency models (simulate external API / tool execution time).

Each is sampled from a per-task RNG so a given (seed, task_id) always sees the
same delays regardless of scheduling order.
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

from agentserve.core.registry import register


class DelayModel(ABC):
    registry_name: str = "?"

    @abstractmethod
    def sample(self, rng: random.Random) -> float: ...


@register("delay", "none")
class NoDelay(DelayModel):
    def sample(self, rng: random.Random) -> float:
        return 0.0


@register("delay", "fixed")
class FixedDelay(DelayModel):
    def __init__(self, seconds: float = 1.0) -> None:
        self.seconds = seconds

    def sample(self, rng: random.Random) -> float:
        return self.seconds


@register("delay", "exponential")
class ExponentialDelay(DelayModel):
    def __init__(self, mean: float = 1.0, max_s: float | None = None) -> None:
        self.mean, self.max_s = mean, max_s

    def sample(self, rng: random.Random) -> float:
        d = rng.expovariate(1.0 / self.mean) if self.mean > 0 else 0.0
        return min(d, self.max_s) if self.max_s else d


@register("delay", "lognormal")
class LogNormalDelay(DelayModel):
    """Parameterized by median and sigma (of the underlying normal)."""

    def __init__(self, median: float = 1.0, sigma: float = 0.5, max_s: float | None = None) -> None:
        self.mu, self.sigma, self.max_s = math.log(median), sigma, max_s

    def sample(self, rng: random.Random) -> float:
        d = rng.lognormvariate(self.mu, self.sigma)
        return min(d, self.max_s) if self.max_s else d


@register("delay", "uniform")
class UniformDelay(DelayModel):
    def __init__(self, low: float = 0.5, high: float = 2.0) -> None:
        self.low, self.high = low, high

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)
