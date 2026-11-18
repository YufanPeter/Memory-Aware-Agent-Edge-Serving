"""KV manager: decides what every slot's KV does, so the shared pool never overflows.

Its jobs, all through controls llama-server exposes over HTTP:
1. Admission. A task is admitted (FIFO) while the sum of every in-flight task's projected peak KV
   stays within rho x pool. rho > 1 overcommits on purpose: agents in tool calls can be
   asked to give their KV back.
2. Gate. Before a request is sent, the manager checks that the slot's estimated size after the
   request fits next to everything else resident. If not, it frees room (finished agents' KV,
   then idle agents' KV, then KV of agents queued behind at the gate) or makes the request
   wait. A request that would overflow the pool is never sent: llama-server would fail every
   request in flight (Phase 1 E0a).
3. Shared prefix (policy fork_prefix, patched server only). One "golden" slot per system prompt
   holds it; a task's slot forks it before its first request and after its KV was dropped, so
   the prefix is stored once and never re-prefilled. Accounting counts each slot's own tokens
   (tokens - shared) plus the golden slots.
4. Lifecycle at tool calls. While an agent waits on a tool and the pool is under pressure, its
   KV accrues rent (pool token-seconds). When the rent reaches the cost of giving the KV up
   (drop = re-prefill later, or offload = slot save + restore), the KV is given up, at most
   once per tool call.

Slot states: free · ready (resident, owner between steps) · active (request in flight) ·
idle (owner in a tool call) · waiting (owner queued at the gate) · dropped · offloaded ·
stale (finished task; KV kept for the next task's shared-prefix reuse).
"""
from __future__ import annotations

import asyncio
import hashlib
import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from edgeserve.events import EventLog
from edgeserve.policies import Policy

# Projected peak context per profile: Phase 1 A10G p95 peak (docs/phase1_details.md §5.3). This is
# the prior; once PEAK_MIN_OBS tasks of a profile have finished, the manager uses their observed p95
# peak (x PEAK_SAFETY) instead, so the projection follows the actual model, prompt and compaction.
PEAK_MIN_OBS = 8
PEAK_WINDOW = 64
PEAK_SAFETY = 1.05
PEAK_P95 = {"react_short": 2500, "react_long": 6250, "react_explain": 3250, "oneshot_long": 3900,
            "closedbook": 600}
# Token estimates for text the server has not seen yet. Qwen2.5 averages ~4 chars/token on this
# workload; we round so that estimates err on the large side: added text counts 1 token per 3
# chars, removed text (compaction) 1 token per 4.5 chars.
CHARS_PER_TOKEN_ADD = 3.0
CHARS_PER_TOKEN_DEL = 4.5
# Phase 1 measurements on A10G / b11960 (docs/phase1_details.md §4.2); prefill and decode step
# time are refined online from server timings.
PREFILL_S_PER_TOKEN = 1 / 4550
SAVE_S_PER_TOKEN = 57344 / 1.35e9
RESTORE_S_PER_TOKEN = 57344 / 4.74e9
TICK_S = 0.05

RESIDENT = ("ready", "active", "idle", "waiting", "stale", "golden")
N_GOLDEN = 3   # one golden slot per system prompt: react, react_explain, oneshot


def family_of(profile: str) -> str:
    """Which system prompt (shared prefix) a profile's prompts start with."""
    if profile.startswith("oneshot") or profile == "closedbook":
        return "oneshot"
    return "explain" if profile == "react_explain" else "react"


@dataclass
class SlotRec:
    slot: int
    task_id: str | None = None
    profile: str = ""
    family: str = ""             # react | explain | oneshot: which shared prefix the slot's KV starts with
    shared: int = 0              # leading tokens of this slot's KV shared with a golden slot (fork)
    prefix_key: str = ""         # which golden prefix they are
    state: str = "free"
    tokens: int = 0              # tokens this slot holds (or has reserved) in the pool; 0 if not resident
    ctx_tokens: int = 0          # the owner's context after its last request (what it needs back)
    ctx_chars: int = 0           # characters of that context
    peak: int = 0                # projected peak, reserved at admission
    max_ctx: int = 0             # largest context the current task has reached
    seq: int = 0                 # placement order: the most recently placed KV likely sits at the top
    gave_up: bool = False        # already released once during the current tool call
    compact_next: bool = False   # the owner should compact its prompt before the next request
    timer: asyncio.Task | None = None
    restore: asyncio.Task | None = None


@dataclass
class KVStats:
    give_ups: dict[str, int] = field(default_factory=dict)    # "<reason>:<drop|offload|stale>" -> count
    gate_waits: int = 0
    gate_wait_s: float = 0.0
    admit_wait_s: float = 0.0
    slot_ops: dict[str, int] = field(default_factory=dict)
    slot_op_failures: int = 0
    slot_op_ms: float = 0.0
    restores_prefetched: int = 0
    peak_resident: int = 0
    forks: int = 0
    fork_failures: int = 0
    forked_tokens: int = 0


class KVManager:
    def __init__(self, llm: Any, events: EventLog, policy: Policy, pool_tokens: int, n_slots: int) -> None:
        self.llm, self.events, self.policy, self.pool = llm, events, policy, pool_tokens
        self.recs = {s: SlotRec(s) for s in range(n_slots)}
        self.cond = asyncio.Condition()
        self.admit_q: deque[str] = deque()
        self.waiters: deque[int] = deque()
        self.decoding = 0
        self.prefill_s_per_token = PREFILL_S_PER_TOKEN
        self._seq = itertools.count(1)
        self.peak_obs: dict[str, deque[int]] = {}
        self.stats = KVStats()
        self.golden: dict[str, tuple[int, int]] = {}     # prefix key -> (golden slot, prefix tokens)
        self.family_prefix: dict[str, int] = {}          # family -> tokens it shares through a golden slot
        self.golden_free = list(range(n_slots - N_GOLDEN, n_slots)) if policy.fork_prefix else []
        for g in self.golden_free:
            self.recs[g].state = "reserved"
        self._golden_lock = asyncio.Lock()

    def projected_peak(self, profile: str) -> int:
        obs = self.peak_obs.get(profile)
        if obs is None or len(obs) < PEAK_MIN_OBS:
            return PEAK_P95.get(profile, 4000)
        xs = sorted(obs)
        return int(xs[min(len(xs) - 1, int(0.95 * len(xs)))] * PEAK_SAFETY)

    # --- accounting -----------------------------------------------------------------------

    def resident(self) -> int:
        """Pool tokens in use: every slot's own tokens; shared prefixes count once (golden slots)."""
        return sum(r.tokens - min(r.shared, r.tokens) for r in self.recs.values())

    @staticmethod
    def own(rec: SlotRec, tokens: int) -> int:
        return tokens - min(rec.shared, tokens)

    def credit(self, family: str) -> int:
        """Tokens of a task's peak that live in a golden slot (shared prefix), not in its own slot."""
        return self.family_prefix.get(family, 0) if self.policy.fork_prefix else 0

    def reserved(self) -> int:
        """Projected peaks of every in-flight task (resident or not), each without the shared
        prefix it takes from a golden slot, plus the golden slots themselves."""
        tasks = sum(max(r.peak, r.tokens, r.ctx_tokens) - self.credit(r.family)
                    for r in self.recs.values() if r.task_id)
        return tasks + sum(r.tokens for r in self.recs.values() if r.state == "golden")

    def under_pressure(self) -> bool:
        return bool(self.admit_q or self.waiters) or self.resident() >= self.policy.pressure * self.pool

    def _set(self, rec: SlotRec, state: str, tokens: int | None = None) -> None:
        if tokens is not None:
            rec.tokens = tokens
        if state not in RESIDENT:
            rec.tokens = 0
            rec.shared = 0
            rec.prefix_key = ""
        if state != rec.state or tokens is not None:
            self.events.emit("kv_state", task_id=rec.task_id, slot=rec.slot, frm=rec.state, to=state,
                             tokens=rec.tokens, shared=rec.shared, resident=self.resident())
        rec.state = state
        self.stats.peak_resident = max(self.stats.peak_resident, self.resident())

    def estimate(self, rec: SlotRec, prompt_chars: int) -> int:
        """Tokens the slot will hold once this prompt is prefilled."""
        delta = prompt_chars - rec.ctx_chars
        rate = CHARS_PER_TOKEN_ADD if delta >= 0 else CHARS_PER_TOKEN_DEL
        return max(0, int(rec.ctx_tokens + delta / rate) + 1)

    # --- costs (section 1.2 of the plan) ---------------------------------------------------

    def drop_cost(self, s: int) -> float:
        """Pool token-seconds lost by re-prefilling s tokens: everyone resident waits."""
        return (self.resident() + s) * s * self.prefill_s_per_token

    def offload_cost(self, s: int) -> float:
        """Slot save + restore block the server (Phase 1 E0a), so everyone resident waits."""
        return (self.resident() + s) * s * (SAVE_S_PER_TOKEN + RESTORE_S_PER_TOKEN)

    def give_up_cost(self, s: int) -> float:
        return self.offload_cost(s) if self.policy.give_up == "offload" else self.drop_cost(s)

    # --- admission --------------------------------------------------------------------------

    def _choose_slot(self, profile: str) -> int:
        """A free slot for a new task: one whose stale KV starts with the same shared prefix
        (a finished task of the same family, so the system prompt is reused), then an empty
        one (nothing to lose), then another family's stale slot (its KV gets overwritten)."""
        family = family_of(profile)
        free = [r for r in self.recs.values()
                if r.task_id is None and r.state not in ("active", "golden", "reserved")]
        if not free:
            raise RuntimeError("admission granted but no slot is free; raise the server's -np")
        same = [r for r in free if r.tokens and r.family == family]
        if same:
            return max(same, key=lambda r: r.seq).slot
        empty = [r for r in free if not r.tokens]
        if empty:
            return min(empty, key=lambda r: r.slot).slot
        return max(free, key=lambda r: r.seq).slot

    async def admit(self, task_id: str, profile: str, slot: int | None = None) -> int:
        """Admit a task (FIFO, by projected peak KV when the policy admits by KV) and return its
        slot: the worker's own slot when `slot` is given, otherwise one chosen for prefix reuse."""
        peak = self.projected_peak(profile)
        t0 = time.monotonic()
        async with self.cond:
            if self.policy.admission:
                self.admit_q.append(task_id)
                own_peak = peak - self.credit(family_of(profile))
                await self.cond.wait_for(
                    lambda: self.admit_q[0] == task_id
                    and self.reserved() + own_peak <= self.policy.overcommit * self.pool)
                self.admit_q.popleft()
            if slot is None:
                slot = self._choose_slot(profile)
            rec = self.recs[slot]
            rec.family = family_of(profile)
            rec.task_id, rec.profile, rec.peak = task_id, profile, peak
            rec.ctx_tokens = rec.ctx_chars = rec.max_ctx = 0
            rec.gave_up = rec.compact_next = False
            self._set(rec, "ready" if rec.tokens else "free")
            self.cond.notify_all()
        wait = time.monotonic() - t0
        self.stats.admit_wait_s += wait
        if self.policy.admission:
            self.events.emit("admit_wait", task_id=task_id, slot=slot, wait_s=round(wait, 4))
        return slot

    # --- the agent's hooks ------------------------------------------------------------------

    def wants_compact(self, slot: int) -> bool:
        rec = self.recs[slot]
        if self.policy.compact == "always":
            return True
        return self.policy.compact == "on_drop" and rec.compact_next

    async def before_generate(self, slot: int, prompt_chars: int, n_predict: int,
                              prefix: str | None = None) -> None:
        rec = self.recs[slot]
        if rec.restore is not None:          # a prefetch started at tool end: let it finish
            await asyncio.gather(rec.restore, return_exceptions=True)
            rec.restore = None
        if self.policy.fork_prefix and prefix and (rec.tokens == 0 or rec.ctx_tokens == 0):
            # The slot holds none of this task's context: take the system prompt from a golden slot
            # instead of prefilling (and storing) another copy of it.
            await self._fork_prefix(rec, prefix)
        need = self.estimate(rec, prompt_chars) + n_predict
        async with self.cond:
            if self.policy.gate:
                await self._make_room(rec, need)
            offloaded = rec.state == "offloaded"
            self.decoding += 1
            self._set(rec, "active", tokens=need)
            if offloaded:
                await self._restore(rec, reason="on_demand")
            rec.compact_next = False
            self.cond.notify_all()

    async def after_generate(self, slot: int, prompt_chars: int, res: Any) -> None:
        rec = self.recs[slot]
        async with self.cond:
            self.decoding -= 1
            if res.ok:
                rec.ctx_tokens = res.prompt_n + res.cache_n + res.predicted_n
                rec.ctx_chars = prompt_chars + len(res.text)
                rec.max_ctx = max(rec.max_ctx, rec.ctx_tokens)
                if res.prompt_n >= 256 and res.prompt_ms > 0:
                    self.prefill_s_per_token += 0.2 * (res.prompt_ms / 1000 / res.prompt_n - self.prefill_s_per_token)
                rec.seq = next(self._seq)
                self._set(rec, "ready", tokens=rec.ctx_tokens)
            else:   # the server may have dropped the slot; assume nothing is left
                rec.ctx_tokens = 0
                rec.ctx_chars = 0
                self._set(rec, "ready", tokens=0)
            self.cond.notify_all()

    async def on_tool_start(self, slot: int, true_delay_s: float) -> None:
        rec = self.recs[slot]
        async with self.cond:
            rec.gave_up = False
            self._set(rec, "idle")
            self.cond.notify_all()     # agents waiting at the gate may now take this KV
        lc = self.policy.lifecycle
        if lc == "always":
            async with self.cond:
                await self._give_up(rec, "always")
                self.cond.notify_all()
        elif lc in ("pressure", "fixed_ttl", "oracle"):
            rec.timer = asyncio.create_task(self._tool_timer(rec, true_delay_s))

    async def on_tool_end(self, slot: int) -> None:
        rec = self.recs[slot]
        if rec.timer is not None:
            rec.timer.cancel()
            await asyncio.gather(rec.timer, return_exceptions=True)
            rec.timer = None
        async with self.cond:
            if rec.state == "idle":
                self._set(rec, "ready")
            elif rec.state == "offloaded" and self.policy.prefetch:
                rec.restore = asyncio.create_task(self._prefetch(rec))

    async def release(self, slot: int) -> None:
        """The task finished (or failed or timed out): keep its KV as stale for the next task's prefix reuse."""
        rec = self.recs[slot]
        for t in (rec.timer, rec.restore):
            if t is not None:
                t.cancel()
                await asyncio.gather(t, return_exceptions=True)
        rec.timer = rec.restore = None
        async with self.cond:
            if rec.state == "active":
                self.decoding -= 1
            if rec.max_ctx and rec.profile:
                self.peak_obs.setdefault(rec.profile, deque(maxlen=PEAK_WINDOW)).append(rec.max_ctx)
            rec.task_id, rec.peak, rec.ctx_tokens, rec.ctx_chars = None, 0, 0, 0
            if rec.slot in self.waiters:
                self.waiters.remove(rec.slot)
            self._set(rec, "stale" if rec.tokens else "free")
            self.cond.notify_all()

    # --- internals (callers hold self.cond) -------------------------------------------------

    async def _make_room(self, rec: SlotRec, need: int) -> None:
        limit = (1 - self.policy.margin) * self.pool
        if need > limit:
            raise RuntimeError(f"slot {rec.slot} needs {need} tokens, more than the whole pool ({self.pool})")
        if self.resident() - self.own(rec, rec.tokens) + self.own(rec, need) <= limit and not self.waiters:
            return
        t0 = time.monotonic()
        prev = rec.state
        self.waiters.append(rec.slot)
        self._set(rec, "waiting")
        try:
            while True:
                head = self.waiters[0] == rec.slot
                if head and self.resident() - self.own(rec, rec.tokens) + self.own(rec, need) <= limit:
                    break
                if head:
                    victim = self._pick_victim(rec)
                    if victim is not None:
                        await self._give_up(victim, "gate")
                        continue
                await self.cond.wait()
        finally:
            if rec.slot in self.waiters:
                self.waiters.remove(rec.slot)
        if rec.state == "waiting":
            self._set(rec, "ready" if prev != "offloaded" or rec.tokens else prev)
        wait = time.monotonic() - t0
        self.stats.gate_waits += 1
        self.stats.gate_wait_s += wait
        self.events.emit("kv_wait", task_id=rec.task_id, slot=rec.slot, wait_s=round(wait, 4), need=need)

    async def _golden_slot(self, prefix: str) -> tuple[int, int] | None:
        """(golden slot, prefix tokens) for this system prompt, priming a golden slot on first use."""
        key = hashlib.sha1(prefix.encode()).hexdigest()[:12]
        async with self._golden_lock:
            if key in self.golden:
                return self.golden[key]
            if not self.golden_free:
                return None
            g = self.golden_free.pop(0)
            res = await self.llm.generate(prefix, 1, [], g)
            if not res.ok:
                self.stats.fork_failures += 1
                return None
            n = res.prompt_n + res.cache_n
            async with self.cond:
                self.recs[g].prefix_key = key
                self._set(self.recs[g], "golden", tokens=n + res.predicted_n)
            self.golden[key] = (g, n)
            return self.golden[key]

    async def _fork_prefix(self, rec: SlotRec, prefix: str) -> None:
        got = await self._golden_slot(prefix)
        if got is None:
            return
        g, n = got
        key = self.recs[g].prefix_key
        if rec.shared and rec.prefix_key == key:
            return
        r = await self.llm.slot_fork(rec.slot, g, n)
        self.events.emit("slot_op", task_id=rec.task_id, **r.to_dict())
        self.stats.slot_ops["fork"] = self.stats.slot_ops.get("fork", 0) + 1
        async with self.cond:
            if r.ok:
                self.stats.forks += 1
                self.stats.forked_tokens += n
                self.family_prefix[rec.family] = n
                rec.shared, rec.prefix_key = n, key
                # whatever the slot held before is gone; it now holds exactly the shared prefix
                self._set(rec, rec.state if rec.state in RESIDENT else "ready", tokens=n)
                if rec.ctx_tokens == 0:
                    rec.ctx_tokens, rec.ctx_chars = n, len(prefix)
            else:
                self.stats.fork_failures += 1
            self.cond.notify_all()

    def _pick_victim(self, asker: SlotRec) -> SlotRec | None:
        """Finished agents' KV first, then idle agents' (cheapest to give up first, most recently
        placed on ties), then KV of agents queued behind the asker at the gate."""
        cands = [r for r in self.recs.values() if r is not asker and r.tokens > 0]
        stale = [r for r in cands if r.state == "stale"]
        if stale:
            return max(stale, key=lambda r: r.seq)
        idle = [r for r in cands if r.state == "idle"]
        if idle:
            return min(idle, key=lambda r: (self.give_up_cost(r.tokens), -r.seq))
        queued = [self.recs[s] for s in reversed(self.waiters) if s != asker.slot and self.recs[s].tokens > 0]
        return queued[0] if queued else None

    async def _give_up(self, rec: SlotRec, reason: str) -> None:
        if rec.tokens == 0 or rec.state not in ("idle", "waiting", "stale", "ready"):
            return
        if rec.state == "stale":
            await self._slot_op("erase", rec)
            self._set(rec, "free")
            kind = "stale"
        elif self.policy.give_up == "offload" and rec.state == "idle":
            r = await self._slot_op("save", rec)
            await self._slot_op("erase", rec)
            if r.ok:
                self._set(rec, "offloaded")
                kind = "offload"
            else:
                self._set(rec, "dropped")
                kind = "drop"
        else:
            await self._slot_op("erase", rec)
            self._set(rec, "dropped")
            kind = "drop"
        if rec.task_id and kind == "drop" and self.policy.compact == "on_drop":
            rec.compact_next = True
        rec.gave_up = True
        key = f"{reason}:{kind}"
        self.stats.give_ups[key] = self.stats.give_ups.get(key, 0) + 1

    async def _restore(self, rec: SlotRec, reason: str) -> bool:
        r = await self._slot_op("restore", rec)
        if reason == "prefetch" and r.ok:
            self.stats.restores_prefetched += 1
        return r.ok

    async def _prefetch(self, rec: SlotRec) -> None:
        """Restore offloaded KV as soon as the tool returns, if it fits without evicting anyone."""
        async with self.cond:
            if rec.state != "offloaded" or self.waiters:
                return
            if self.resident() + rec.ctx_tokens > (1 - self.policy.margin) * self.pool:
                return
            self._set(rec, "ready", tokens=rec.ctx_tokens)
            if not await self._restore(rec, "prefetch"):
                self._set(rec, "dropped")
            self.cond.notify_all()

    async def _slot_op(self, op: str, rec: SlotRec) -> Any:
        fname = f"kvm_slot{rec.slot}.bin"
        if op == "save":
            r = await self.llm.slot_save(rec.slot, fname)
        elif op == "restore":
            r = await self.llm.slot_restore(rec.slot, fname)
        else:
            r = await self.llm.slot_erase(rec.slot)
        self.stats.slot_ops[op] = self.stats.slot_ops.get(op, 0) + 1
        self.stats.slot_op_ms += r.wall_ms
        if not r.ok:
            self.stats.slot_op_failures += 1
        self.events.emit("slot_op", task_id=rec.task_id, **r.to_dict())
        return r

    async def _tool_timer(self, rec: SlotRec, true_delay_s: float) -> None:
        """Runs while the owner is in a tool call; gives the KV up when its costs say so."""
        lc, p = self.policy.lifecycle, self.policy
        rent = 0.0
        t_last = t0 = time.monotonic()
        while True:
            await asyncio.sleep(TICK_S if lc != "fixed_ttl" else p.ttl_s)
            async with self.cond:
                if rec.state != "idle" or rec.gave_up:
                    return
                now = time.monotonic()
                dt, t_last = now - t_last, now
                s = rec.tokens
                fire = None
                if lc == "fixed_ttl" and now - t0 >= p.ttl_s:
                    fire = "ttl"
                elif lc == "oracle":
                    if self.under_pressure() and s * max(0.0, true_delay_s - (now - t0)) >= self.give_up_cost(s):
                        fire = "oracle"
                elif lc == "pressure" and self.under_pressure():
                    rent += s * dt
                    if rent >= self.give_up_cost(s):
                        fire = "rent"
                if fire:
                    await self._give_up(rec, fire)
                    self.cond.notify_all()
                    return

    def summary(self) -> dict[str, Any]:
        s = self.stats
        return {"give_ups": s.give_ups, "gate_waits": s.gate_waits, "gate_wait_s": round(s.gate_wait_s, 3),
                "admit_wait_s": round(s.admit_wait_s, 3), "slot_ops": s.slot_ops,
                "slot_op_failures": s.slot_op_failures, "slot_op_ms": round(s.slot_op_ms, 1),
                "restores_prefetched": s.restores_prefetched, "peak_resident_tokens": s.peak_resident,
                "forks": s.forks, "fork_failures": s.fork_failures, "forked_tokens": s.forked_tokens,
                "golden_slots": {k: v[0] for k, v in self.golden.items()},
                "pool_tokens": self.pool, "policy": self.policy.__dict__}
