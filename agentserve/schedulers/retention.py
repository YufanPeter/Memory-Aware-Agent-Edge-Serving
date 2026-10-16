"""KV retention policies (#10): which idle agent's KV to give up when a slot or
pool space is needed, and how (erase = recompute later, swap = save/restore).

`server` leaves everything to llama.cpp (requests are not pinned; the server
picks slots by prompt similarity and purges idle slots itself) — this is the
simple fixed cache policy every managed policy is compared against.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from agentserve.core.registry import register

if TYPE_CHECKING:
    from agentserve.schedulers.base import Scheduler
    from agentserve.schedulers.slots import SlotState


class RetentionPolicy(ABC):
    registry_name: str = "?"
    managed: bool = True

    def __init__(self, mode: str = "erase", **params: Any) -> None:
        if mode not in ("erase", "swap"):
            raise ValueError("mode must be 'erase' or 'swap'")
        self.mode = mode
        self.params = params

    @abstractmethod
    def choose_victim(self, candidates: list["SlotState"], sched: "Scheduler") -> "SlotState":
        """Pick one idle, owned slot to evict. `candidates` is non-empty."""


@register("retention", "server")
class ServerManaged(RetentionPolicy):
    managed = False

    def choose_victim(self, candidates, sched):  # never called
        raise AssertionError


@register("retention", "lru")
class LRURetention(RetentionPolicy):
    def choose_victim(self, candidates, sched):
        return min(candidates, key=lambda s: s.last_used)


@register("retention", "largest")
class LargestRetention(RetentionPolicy):
    """Free the most memory per eviction."""

    def choose_victim(self, candidates, sched):
        return max(candidates, key=lambda s: (s.tokens, -s.last_used))


@register("retention", "tool_aware")
class ToolAwareRetention(RetentionPolicy):
    """Evict the agent expected to stay idle longest: one in a tool call with the
    most expected remaining time (learned per-tool mean minus elapsed). Agents
    waiting in the scheduler queue are expected back soonest and are evicted last.
    Ties -> larger KV first."""

    def expected_idle(self, slot: "SlotState", sched: "Scheduler") -> float:
        owner = slot.owner
        now = sched.clock.now()
        if owner in sched.tool_busy:
            tool, t0 = sched.tool_busy[owner]
            return max(0.0, sched.est.tool_mean(tool) - (now - t0)) + 1e-3
        if any(i.req.task_id == owner for i in sched.queue):
            return 0.0
        # between steps (thinking / just finished a tool): assume back very soon
        return 1e-4

    def choose_victim(self, candidates, sched):
        return max(candidates, key=lambda s: (self.expected_idle(s, sched), s.tokens, -s.last_used))
