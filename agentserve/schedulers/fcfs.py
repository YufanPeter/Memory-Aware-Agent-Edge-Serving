"""FCFS baseline: one global FIFO queue, at most K requests in flight,
non-preemptive (no quantum). Optional task-level admission cap via
`max_active_tasks` (handled by the base class)."""
from __future__ import annotations

from agentserve.core.registry import register
from agentserve.schedulers.base import QueueItem, Scheduler


@register("scheduler", "fcfs")
class FCFSScheduler(Scheduler):
    def priority(self, item: QueueItem) -> tuple:
        return (item.seq,)  # original arrival order; a chunked request keeps its place
