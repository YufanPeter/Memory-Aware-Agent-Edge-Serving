"""Runtime-reported metrics via Backend.probe() (e.g. llama-server /metrics)."""
from __future__ import annotations

from typing import Any

from agentserve.core.registry import register
from agentserve.monitors.base import Monitor, MonitorContext


@register("monitor", "backend_probe")
class BackendProbeMonitor(Monitor):
    def __init__(self, interval: float = 0.5, keys: list[str] | None = None, **params: Any) -> None:
        super().__init__(interval, **params)
        self.keys = keys  # None = keep everything

    async def sample(self, ctx: MonitorContext) -> dict[str, Any] | None:
        d = await ctx.backend.probe()
        if self.keys is not None:
            d = {k: v for k, v in d.items() if k in self.keys}
        return d or None
