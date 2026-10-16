"""Arrival processes: decide *when* each TaskSpec is released into the system."""
from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from typing import Awaitable, Callable

from agentserve.core.events import Clock
from agentserve.core.registry import register
from agentserve.core.types import TaskSpec

Launch = Callable[[TaskSpec], Awaitable[None]]


class ArrivalProcess(ABC):
    registry_name: str = "?"

    @abstractmethod
    async def drive(self, specs: list[TaskSpec], launch: Launch, clock: Clock, seed: int) -> None:
        """Release every spec via `launch` and return once all launched tasks finished."""


class OpenLoop(ArrivalProcess):
    """Arrival times are fixed in advance and independent of completions."""

    @abstractmethod
    def times(self, n: int, rng: random.Random) -> list[float]: ...

    async def drive(self, specs: list[TaskSpec], launch: Launch, clock: Clock, seed: int) -> None:
        times = self.times(len(specs), random.Random(f"arrival:{seed}"))
        running = []
        for spec, t in zip(specs, times):
            spec.arrival_time = t
            delay = t - clock.now()
            if delay > 0:
                await asyncio.sleep(delay)
            running.append(asyncio.create_task(launch(spec)))
        await asyncio.gather(*running)


@register("arrival", "poisson")
class Poisson(OpenLoop):
    def __init__(self, rate: float = 0.5) -> None:
        self.rate = rate  # tasks per second

    def times(self, n: int, rng: random.Random) -> list[float]:
        t, out = 0.0, []
        for _ in range(n):
            out.append(t)
            t += rng.expovariate(self.rate)
        return out


@register("arrival", "uniform")
class Uniform(OpenLoop):
    def __init__(self, rate: float = 0.5) -> None:
        self.rate = rate

    def times(self, n: int, rng: random.Random) -> list[float]:
        return [i / self.rate for i in range(n)]


@register("arrival", "burst")
class Burst(OpenLoop):
    """Everything arrives at t=0 (stress test / offline batch)."""

    def times(self, n: int, rng: random.Random) -> list[float]:
        return [0.0] * n


@register("arrival", "phased")
class Phased(OpenLoop):
    """Piecewise Poisson rates, e.g. [{rate: 0.2, duration: 60}, {rate: 1.0, duration: 30}].
    The last phase repeats until all tasks have arrived."""

    def __init__(self, phases: list[dict]) -> None:
        self.phases = phases

    def times(self, n: int, rng: random.Random) -> list[float]:
        out: list[float] = []
        start = 0.0
        for i, ph in enumerate(self.phases):
            last = i == len(self.phases) - 1
            t = start
            while len(out) < n:
                t += rng.expovariate(ph["rate"])
                if not last and t > start + ph["duration"]:
                    break
                out.append(t)
            start += ph.get("duration", 0.0)
        return out


@register("arrival", "closed_loop")
class ClosedLoop(ArrivalProcess):
    """`users` concurrent clients, each submitting its next task when the last one finishes
    (plus optional think time)."""

    def __init__(self, users: int = 4, think_time: float = 0.0) -> None:
        self.users, self.think_time = users, think_time

    async def drive(self, specs: list[TaskSpec], launch: Launch, clock: Clock, seed: int) -> None:
        pending = list(reversed(specs))

        async def user() -> None:
            while pending:
                spec = pending.pop()
                spec.arrival_time = clock.now()
                await launch(spec)
                if self.think_time:
                    await asyncio.sleep(self.think_time)

        await asyncio.gather(*(user() for _ in range(self.users)))
