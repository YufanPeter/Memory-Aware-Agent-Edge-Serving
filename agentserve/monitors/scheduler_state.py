"""Queue length / in-flight / active tasks over time, from Scheduler.state()."""
from __future__ import annotations

from typing import Any

from agentserve.core.registry import register
from agentserve.monitors.base import Monitor, MonitorContext


@register("monitor", "scheduler_state")
class SchedulerStateMonitor(Monitor):
    async def sample(self, ctx: MonitorContext) -> dict[str, Any] | None:
        return ctx.scheduler.state()
