"""Scheduler-side slot manager: owns the agent -> backend slot mapping and keeps
an explicit account of KV tokens per slot.

Each agent task is pinned to one slot while it has retained KV, so its next
step (or next chunk) hits the prefix cache. When a slot or pool space is
needed, the RetentionPolicy picks an idle victim, which is erased (recompute
later) or swapped out (slot save + erase, restored on return).

The account is kept within the KV budget by evicting proactively, so the
server never has to purge a slot behind the scheduler's back.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentserve.backends.base import Backend
from agentserve.core.events import EventLog
from agentserve.core.types import GenerationResult
from agentserve.schedulers.retention import RetentionPolicy

if TYPE_CHECKING:
    from agentserve.schedulers.base import Scheduler


@dataclass
class SlotState:
    id: int
    owner: str | None = None   # task id whose KV this slot holds (None = free / stale)
    tokens: int = 0            # KV tokens held (projected while active)
    active: bool = False
    last_used: float = 0.0


class SlotManager:
    def __init__(self, backend: Backend, n_slots: int, budget: int, retention: RetentionPolicy,
                 events: EventLog, sched: "Scheduler") -> None:
        self.backend = backend
        self.budget = budget
        self.retention = retention
        self.events = events
        self.sched = sched
        self.slots = [SlotState(i) for i in range(n_slots)]
        self.owner_slot: dict[str, int] = {}
        self.swapped: dict[str, int] = {}  # owner -> tokens in the saved image
        self.n_evictions = {"erase": 0, "swap": 0, "stale": 0}
        self.slot_ops_ms = 0.0

    # -- accounting ------------------------------------------------------------------

    def used(self) -> int:
        return sum(s.tokens for s in self.slots)

    def retained_tokens(self, owner: str) -> int:
        sid = self.owner_slot.get(owner)
        return self.slots[sid].tokens if sid is not None else 0

    def _idle_owned(self, exclude: int) -> list[SlotState]:
        return [s for s in self.slots if not s.active and s.owner is not None and s.id != exclude]

    def _stale(self, exclude: int) -> list[SlotState]:
        return [s for s in self.slots if not s.active and s.owner is None and s.tokens > 0 and s.id != exclude]

    # -- eviction ----------------------------------------------------------------------

    async def evict(self, slot: SlotState, reason: str) -> None:
        """Free a slot's KV. Slot-op failures degrade to 'erase' semantics instead of
        failing requests (the owner simply recomputes on return)."""
        owner, tokens = slot.owner, slot.tokens
        t0 = self.sched.clock.now()
        kind = "erase" if owner is not None else "stale"
        if owner is not None and self.retention.mode == "swap":
            try:
                r = await self.backend.slot_save(slot.id, self._image(owner))
                self.swapped[owner] = int(r.get("tokens", tokens))
                kind = "swap"
            except Exception as e:
                self.events.emit("slot_op_error", op="save", slot=slot.id, owner=owner, error=repr(e)[:200])
        try:
            await self.backend.slot_erase(slot.id)
        except Exception as e:
            self.events.emit("slot_op_error", op="erase", slot=slot.id, owner=owner, error=repr(e)[:200])
        ms = (self.sched.clock.now() - t0) * 1e3
        self.slot_ops_ms += ms
        self.n_evictions[kind] += 1
        self.events.emit("slot_evict", slot=slot.id, owner=owner, tokens=tokens, mode=kind,
                         reason=reason, ms=ms, policy=self.retention.registry_name)
        if owner is not None:
            self.owner_slot.pop(owner, None)
        slot.owner, slot.tokens = None, 0

    @staticmethod
    def _image(owner: str) -> str:
        return owner.replace(":", "_")

    async def make_room(self, extra: int, keep: SlotState) -> None:
        """Evict idle KV (stale first, then the retention policy's victim) until
        `extra` more tokens fit next to everything held now."""
        while self.used() + extra > self.budget:
            cands = self._stale(keep.id) or self._idle_owned(keep.id)
            if not cands:
                return  # only active requests left; the admission policy owns that risk
            victim = cands[0] if cands[0].owner is None else self.retention.choose_victim(cands, self.sched)
            await self.evict(victim, "memory")

    # -- acquire / release ---------------------------------------------------------------

    async def acquire(self, owner: str, need: int) -> int:
        """Return a slot for `owner` that will hold up to `need` tokens, evicting idle KV as required."""
        sid = self.owner_slot.get(owner)
        if sid is None or self.slots[sid].active:
            free = [s for s in self.slots if not s.active and s.owner is None]
            if free:
                # Prefer a stale slot: its shared prefix (system prompt) is reused by the server.
                slot = min(free, key=lambda s: (s.tokens == 0, s.tokens))
            else:
                cands = self._idle_owned(exclude=-1)
                if not cands:
                    raise RuntimeError("no idle slot: scheduler concurrency exceeds backend slots")
                slot = self.retention.choose_victim(cands, self.sched)
                await self.evict(slot, "slot")
            slot.owner = owner
            self.owner_slot[owner] = slot.id
            if owner in self.swapped:
                await self._restore(slot, owner)
            sid = slot.id
        slot = self.slots[sid]
        slot.active = True
        await self.make_room(max(0, need - slot.tokens), slot)
        slot.tokens = max(slot.tokens, need)
        return sid

    async def _restore(self, slot: SlotState, owner: str) -> None:
        img = self.swapped.pop(owner)
        if slot.tokens:  # restore needs an empty sequence plus free cells for the image
            await self.backend.slot_erase(slot.id)
            slot.tokens = 0
        await self.make_room(img, slot)
        t0 = self.sched.clock.now()
        try:
            r = await self.backend.slot_restore(slot.id, self._image(owner))
            slot.tokens = int(r.get("tokens", img))
            ok = True
        except Exception as e:  # fall back to recomputation
            self.events.emit("slot_op_error", op="restore", slot=slot.id, owner=owner, error=repr(e)[:200])
            ok = False
        ms = (self.sched.clock.now() - t0) * 1e3
        self.slot_ops_ms += ms
        self.events.emit("slot_restore", slot=slot.id, owner=owner, tokens=slot.tokens, ms=ms, ok=ok)
        asyncio.get_running_loop().create_task(self.backend.slot_discard(self._image(owner)))

    def release(self, sid: int, res: GenerationResult) -> None:
        slot = self.slots[sid]
        slot.active = False
        slot.last_used = self.sched.clock.now()
        if res.ok:
            slot.tokens = res.prompt_tokens + res.completion_tokens
        else:  # failed request: the server dropped this slot's KV
            slot.tokens = 0
            if slot.owner:
                self.owner_slot.pop(slot.owner, None)
            slot.owner = None

    def task_done(self, owner: str) -> None:
        sid = self.owner_slot.pop(owner, None)
        if sid is not None:
            self.slots[sid].owner = None  # tokens stay as stale KV until reused/evicted
        if self.swapped.pop(owner, None) is not None:
            asyncio.get_running_loop().create_task(self.backend.slot_discard(self._image(owner)))
