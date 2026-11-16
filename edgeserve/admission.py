"""Task admission (docs/phase2_spec.md §2.3) and the slot pool (§5).

Admission hands out FIFO tickets at queue.get and blocks the head of line: only the lowest
waiting ticket may be admitted, so a long task is never bypassed. A ticket is admitted iff

  (a) fewer than K tasks are admitted (fixed K, or the compute cap K_max);
  (b) no request waits at the gate and every admitted task has reached its first acquire
      (gated configurations only): never add pressure while agents wait for memory, and the
      /tokenize round trip between admit and the first acquire must not let a burst through;
  (c) kv:   sum of projected peaks R̂ (with the new one) <= (1 + omega)(c - m),
      duty: sum of duty-cycle-weighted peaks + R̂_j <= theta (c - m).

(c) is waived while nothing is admitted, so a task whose projection alone exceeds the bound
still runs (the gate keeps it safe; kv_infeasible catches what cannot run at all).

SlotPool gives a managed agent one slot for its lifetime: a STALE slot of the same prompt
family first (its prefix is reused), then a FREE one, then any STALE one.
"""
from __future__ import annotations

import asyncio
import bisect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence


@dataclass
class Ticket:
    ticket: int
    task_id: str
    R_hat: int
    t0: float = 0.0
    fut: asyncio.Future | None = field(default=None, repr=False)


class Admission:
    """FIFO tickets with head-of-line blocking; subclasses add condition (c)."""

    kind = "fixed"

    def __init__(self, k: int, cap: int, gated: bool = True) -> None:
        self.k, self.cap, self.gated = k, cap, gated
        self.queue: list[Ticket] = []
        self.blocked: str | None = None     # why the head was last held: kmax | gate | kv

    def push(self, t: Ticket) -> None:
        bisect.insort(self.queue, t, key=lambda x: x.ticket)

    def remove(self, t: Ticket) -> None:
        if t in self.queue:
            self.queue.remove(t)

    def head(self) -> Ticket | None:
        return self.queue[0] if self.queue else None

    def limit(self) -> float:
        return float(self.k)

    def can_admit(self, agents: Sequence[Any], R_hat_j: int, waiters: int, not_started: int = 0) -> bool:
        """`agents`: the admitted tasks (with .R_hat and .profile); `waiters`: the gate queue."""
        if len(agents) >= self.k:
            self.blocked = "kmax"
        elif self.gated and (waiters > 0 or not_started > 0):
            self.blocked = "gate"
        elif agents and not self.fits(agents, R_hat_j):
            self.blocked = self.kind
        else:
            self.blocked = None
        return self.blocked is None

    def fits(self, agents: Sequence[Any], R_hat_j: int) -> bool:
        return True


class FixedK(Admission):
    kind = "fixed"


class KVAdmission(Admission):
    kind = "kv"

    def __init__(self, k: int, cap: int, omega: float, gated: bool = True) -> None:
        super().__init__(k, cap, gated)
        self.omega = omega

    def limit(self) -> float:
        return (1.0 + self.omega) * self.cap

    def fits(self, agents: Sequence[Any], R_hat_j: int) -> bool:
        return sum(a.R_hat for a in agents) + R_hat_j <= self.limit()


class DutyAdmission(Admission):
    """Ablation `adm_duty`: an agent in a tool needs no memory under the managed lifecycle, so
    its peak is weighted by its profile's duty cycle (the share of time it has a request in
    flight). The new task counts in full."""

    kind = "duty"

    def __init__(self, k: int, cap: int, theta: float, duty: Callable[[str], float], gated: bool = True) -> None:
        super().__init__(k, cap, gated)
        self.theta, self.duty = theta, duty

    def limit(self) -> float:
        return self.theta * self.cap

    def fits(self, agents: Sequence[Any], R_hat_j: int) -> bool:
        return sum(self.duty(a.profile) * a.R_hat for a in agents) + R_hat_j <= self.limit()


def make_admission(cfg: Any, duty: Callable[[str], float]) -> Admission:
    cap, gated = cfg.ctx - cfg.margin, cfg.gate != "none"
    if cfg.admission == "kv":
        return KVAdmission(cfg.k, cap, cfg.omega, gated)
    if cfg.admission == "duty":
        return DutyAdmission(cfg.k, cap, cfg.theta, duty, gated)
    return FixedK(cfg.k, cap, gated)


class SlotPool:
    """Unowned slots: FREE (empty) or STALE (a finished task's KV, with its prompt family)."""

    def __init__(self, slots: Iterable[int]) -> None:
        self.free: set[int] = set(slots)
        self.stale: dict[int, str | None] = {}

    def take(self, family: str | None, busy: Callable[[int], bool] | None = None,
             size: Callable[[int], int] | None = None) -> int:
        """Same-family STALE (smallest first, least to free), then FREE (lowest id), then any
        STALE. Slots with an op in flight only if nothing else is left."""
        busy = busy or (lambda k: False)
        size = size or (lambda k: 0)
        for allow_busy in (False, True):
            ok = (lambda k: allow_busy or not busy(k))
            same = [k for k, f in self.stale.items() if family is not None and f == family and ok(k)]
            if same:
                return self.claim(min(same, key=lambda k: (size(k), k)))
            free = [k for k in self.free if ok(k)]
            if free:
                return self.claim(min(free))
            other = [k for k in self.stale if ok(k)]
            if other:
                return self.claim(min(other, key=lambda k: (size(k), k)))
        raise RuntimeError("slot pool exhausted: more agents than server slots (-np)")

    def claim(self, slot: int) -> int:
        self.free.discard(slot)
        self.stale.pop(slot, None)
        return slot

    def give(self, slot: int, stale: bool, family: str | None = None) -> None:
        self.claim(slot)
        if stale:
            self.stale[slot] = family
        else:
            self.free.add(slot)

    def erased(self, slot: int) -> None:
        """A STALE slot was erased: it is FREE now (no-op for an owned slot)."""
        if slot in self.stale:
            del self.stale[slot]
            self.free.add(slot)

    def __contains__(self, slot: int) -> bool:
        return slot in self.free or slot in self.stale
