"""Periodic samplers. Each returns a flat dict that is written as a `monitor`
event; a monitor that does not apply on this host/backend returns None from
`available()` and is skipped."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from agentserve.backends.base import Backend


@dataclass
class MonitorContext:
    backend: Backend
    scheduler: Any  # schedulers.base.Scheduler; typed loosely to keep monitors independent of L2


class Monitor(ABC):
    registry_name: str = "?"

    def __init__(self, interval: float = 0.2, **params: Any) -> None:
        self.interval = interval
        self.params = params

    def available(self, ctx: MonitorContext) -> bool:
        return True

    @abstractmethod
    async def sample(self, ctx: MonitorContext) -> dict[str, Any] | None: ...
