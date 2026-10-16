"""Memory-pressure preemptive scheduling (#8).

Admission: a request (chunk) is dispatched only if the projected KV of all
running chunks plus its own fits in `(1 - headroom) * kv_budget`. On llama.cpp
an overflow of *active* KV fails every in-flight request, so this turns a hard
failure cliff into queueing.

Preemption: requests run in chunks; when a higher-priority request is blocked
by memory, lower-priority requests are not resumed at their chunk boundary
(strict reservation, unless `backfill`), and their now-idle KV is evicted by
the retention policy — `mode: erase` = recompute on resume (vLLM recompute),
`mode: swap` = save/restore (vLLM swap).

The ordering comes from any other registered policy (`order: fcfs | rr | srpt |
mlfq | las`); the class actually instantiated is MemoryAwareMixin + that policy.
Concurrency defaults to the number of backend slots: memory, not K, is the limiter.
"""
from __future__ import annotations

from typing import Any

from agentserve.backends.base import Backend
from agentserve.core import registry
from agentserve.core.events import EventLog
from agentserve.core.registry import register
from agentserve.schedulers.base import QueueItem, Scheduler


class MemoryAwareMixin(Scheduler):
    def __init__(self, backend: Backend, events: EventLog, headroom: float = 0.1, backfill: bool = False,
                 max_concurrency: int | None = None, **kw: Any) -> None:
        K = max_concurrency or int(backend.capabilities().get("max_concurrency", 4))
        super().__init__(backend, events, max_concurrency=K, **kw)
        if self.kv_budget() is None:
            raise ValueError("mem_aware needs a backend that reports kv_budget_tokens")
        self.headroom, self.backfill = headroom, backfill
        self.params.update(headroom=headroom, backfill=backfill)
        self.blocked_decisions = 0
        self.max_projected = 0

    def fits(self, item: QueueItem) -> bool:
        return self.active_kv() + self.projected_kv(item) <= (1 - self.headroom) * self.kv_budget()

    def pick_next(self) -> QueueItem | None:
        if len(self.inflight) >= self.K or not self.queue:
            return None
        for item in self.ordered_queue():
            if not self.inflight or self.fits(item):  # never deadlock on an oversized request
                self.max_projected = max(self.max_projected, self.active_kv() + self.projected_kv(item))
                return item
            self.blocked_decisions += 1
            if not self.backfill:
                return None  # reserve memory for the highest-priority request
        return None

    def stats(self) -> dict[str, Any]:
        return {**super().stats(), "blocked_decisions": self.blocked_decisions,
                "max_projected_kv": self.max_projected}


_COMBINED: dict[str, type] = {}


@register("scheduler", "mem_aware")
class MemoryAwareScheduler:
    """Factory: returns an instance of MemoryAwareMixin combined with the `order` policy."""

    def __new__(cls, backend: Backend, events: EventLog, order: str = "fcfs", **kw: Any) -> Scheduler:
        base = registry.get("scheduler", order)
        if order not in _COMBINED:
            _COMBINED[order] = type(f"MemoryAware_{order}", (MemoryAwareMixin, base),
                                    {"registry_name": f"mem_aware[{order}]"})
        if order == "fcfs":
            kw.setdefault("quantum", 32)  # chunk boundaries are the preemption points
        obj = _COMBINED[order](backend, events, **kw)
        obj.params["order"] = order
        return obj
