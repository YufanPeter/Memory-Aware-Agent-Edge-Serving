"""Preemptive request/agent scheduling policies. Each is a priority function plus
a quantum on top of the shared chunked machinery in base.Scheduler.

  rr    (#4) round robin: every request yields after `quantum` tokens
  srpt  (#5) shortest (estimated) remaining processing time first
  mlfq  (#6) multi-level feedback queue / least attained service, request level
  las   (#7) agent-level least attained service: priority by the whole task's
             accumulated service, so short agents' later steps overtake long agents

All support `aging_s` (anti-starvation) and any KV retention policy.
"""
from __future__ import annotations

import math
from typing import Any

from agentserve.backends.base import Backend
from agentserve.core.events import EventLog
from agentserve.core.registry import register
from agentserve.schedulers.base import QueueItem, Scheduler


@register("scheduler", "rr")
class RoundRobinScheduler(Scheduler):
    def __init__(self, backend: Backend, events: EventLog, quantum: int = 32, **kw: Any) -> None:
        super().__init__(backend, events, quantum=quantum, **kw)

    def priority(self, item: QueueItem) -> tuple:
        return (item.t_ready, item.seq)  # back of the line after every quantum


@register("scheduler", "srpt")
class SRPTScheduler(Scheduler):
    """Remaining time = prefill of the not-yet-cached prompt + decode of the predicted
    remaining output (per-workload-mix running mean of output length, learned online)."""

    def __init__(self, backend: Backend, events: EventLog, quantum: int = 32, **kw: Any) -> None:
        super().__init__(backend, events, quantum=quantum, **kw)

    def remaining_ms(self, item: QueueItem) -> float:
        cached = self.slots.retained_tokens(item.req.task_id) if self.slots else 0
        if item.resumed:
            cached = item.prompt_est + item.gen_tokens  # expected to continue from its own KV
        uncached = max(0, item.prompt_est - cached)
        out_left = max(1.0, self.est.predicted_output(item.req) - item.gen_tokens)
        return uncached * self.est.prefill_ms.value + out_left * self.est.decode_ms.value

    def priority(self, item: QueueItem) -> tuple:
        return (self.remaining_ms(item), item.seq)


def _level(x: float, unit: float, levels: int) -> int:
    return min(levels - 1, int(math.log2(1 + x / unit))) if unit > 0 else 0


@register("scheduler", "mlfq")
class MLFQScheduler(Scheduler):
    """Level = floor(log2(1 + attained/quantum)); the chunk at level l is quantum*2^l,
    so a request is demoted exactly when it uses up its level's quantum.
    skip_join: new requests start at the level their prefill cost implies (FastServe)."""

    def __init__(self, backend: Backend, events: EventLog, quantum: int = 16, levels: int = 4,
                 skip_join: bool = True, **kw: Any) -> None:
        super().__init__(backend, events, quantum=quantum, levels=levels, skip_join=skip_join, **kw)
        self.levels, self.skip_join = levels, skip_join

    def attained(self, item: QueueItem) -> float:
        virt = 0.0
        if self.skip_join:  # prefill time expressed in decode-token equivalents
            virt = item.prompt_est * self.est.prefill_ms.value / self.est.decode_ms.value
        return item.gen_tokens + virt

    def level(self, item: QueueItem) -> int:
        return _level(self.attained(item), self.quantum, self.levels)

    def quantum_for(self, item: QueueItem) -> int | None:
        return self.quantum * (2 ** self.level(item))

    def priority(self, item: QueueItem) -> tuple:
        return (self.level(item), item.t_ready, item.seq)


@register("scheduler", "las")
class AgentLASScheduler(Scheduler):
    """Program-level least attained service (cf. Autellix PLAS). A task's service is the
    backend time of all its finished chunks; level = floor(log2(1 + service/base_s)).
    Within a level, earlier-arriving tasks first."""

    def __init__(self, backend: Backend, events: EventLog, quantum: int = 32, base_s: float = 1.0,
                 levels: int = 5, aging_s: float | None = 60.0, **kw: Any) -> None:
        super().__init__(backend, events, quantum=quantum, base_s=base_s, levels=levels,
                         aging_s=aging_s, **kw)
        self.base_s, self.levels = base_s, levels

    def level(self, item: QueueItem) -> int:
        return _level(self.task_service.get(item.req.task_id, 0.0), self.base_s, self.levels)

    def priority(self, item: QueueItem) -> tuple:
        tid = item.req.task_id
        return (self.level(item), self.task_arrival.get(tid, item.t_enqueue), item.seq)
