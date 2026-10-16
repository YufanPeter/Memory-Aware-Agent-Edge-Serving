"""L2 Scheduler interface and shared machinery.

A Scheduler sits between agents and a Backend. Agents call `generate()` (the
LLMClient protocol); the scheduler queues the request and decides when, on
which slot, and for how many tokens it runs.

Preemption is cooperative and chunk-based: with a `quantum`, each dispatch
generates at most `quantum` tokens, then the request returns to the queue and
the policy picks again. A request that loses its turn there is "preempted"; it
resumes later by continuing its partial output (`GenerationRequest.prefix`).
Whether its KV survives the pause is up to the KV retention policy.

Policies implement only small hooks:
  priority(item)    -> sort key, smallest first           (default: FIFO)
  quantum_for(item) -> tokens for the next chunk or None   (default: self.quantum)
  pick_next()       -> override for admission rules (e.g. memory-aware)
  admit_task / task_done -> task-level hooks
Everything else (queue, chunk loop, slot pinning, KV accounting, estimators,
events, overhead accounting) lives here so policies are compared on identical
machinery.
"""
from __future__ import annotations

import asyncio
import itertools
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

from agentserve.backends.base import Backend
from agentserve.core import registry
from agentserve.core.events import EventLog
from agentserve.core.types import GenerationRequest, GenerationResult, TaskResult, TaskSpec
from agentserve.schedulers.estimators import Estimators
from agentserve.schedulers.slots import SlotManager


@dataclass
class QueueItem:
    req: GenerationRequest
    t_enqueue: float
    seq: int
    future: asyncio.Future = field(repr=False)
    t_ready: float = 0.0               # when it (re-)entered the queue
    generated: str = ""
    gen_tokens: int = 0
    chunks: list[GenerationResult] = field(default_factory=list)
    service: float = 0.0               # seconds of backend execution so far
    preemptions: int = 0
    preempt_wait: float = 0.0
    recompute_tokens: int = 0
    resumed: bool = False
    dispatch_mark: int = 0             # global dispatch counter when requeued
    prompt_est: int = 0
    kv_proj: int = 0                   # projected KV while the current chunk runs
    chunk_tokens: int = 0
    slot: int | None = None


class Scheduler:
    registry_name: str = "?"

    def __init__(
        self,
        backend: Backend,
        events: EventLog,
        max_concurrency: int = 4,
        quantum: int | None = None,
        retention: dict[str, Any] | None = None,
        max_active_tasks: int | None = None,
        aging_s: float | None = None,
        **params: Any,
    ) -> None:
        self.backend = backend
        self.events = events
        self.clock = events.clock
        self.K = max_concurrency
        self.quantum = quantum
        self.max_active_tasks = max_active_tasks
        self.aging_s = aging_s
        self.params = dict(max_concurrency=max_concurrency, quantum=quantum, retention=retention,
                           max_active_tasks=max_active_tasks, aging_s=aging_s, **params)
        self.queue: list[QueueItem] = []
        self.inflight: dict[str, QueueItem] = {}
        self.active_tasks: set[str] = set()
        self.task_arrival: dict[str, float] = {}
        self.task_service: dict[str, float] = {}
        self.task_ctx_tokens: dict[str, int] = {}   # KV-able context after the task's last step
        self.tool_busy: dict[str, tuple[str, float]] = {}
        self.est = Estimators()
        self._seq = itertools.count()
        self._dispatches = 0
        self._wake = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self._running: set[asyncio.Task] = set()
        self._task_waiters: deque[asyncio.Future] = deque()
        self._reserved = 0
        self.overhead_s = 0.0
        self.decisions = 0

        caps = backend.capabilities()
        self.caps = caps
        self.retention = registry.build("retention", retention or {"type": "server"})
        if self.quantum and not caps.get("continuation"):
            raise ValueError(f"backend {backend.registry_name} cannot continue generation; quantum unsupported")
        self.slots: SlotManager | None = None
        if self.retention.managed:
            if not caps.get("slot_control"):
                raise ValueError(f"retention {self.retention.registry_name} needs a backend with slot_control")
            if self.retention.mode == "swap" and not caps.get("slot_save"):
                raise ValueError("swap retention needs slot_save")
            n_slots = int(caps["max_concurrency"])
            if self.K > n_slots:
                raise ValueError(f"max_concurrency {self.K} > backend slots {n_slots}")
            self.slots = SlotManager(backend, n_slots, int(caps["kv_budget_tokens"]),
                                     self.retention, events, self)
        events.subscribe(self._on_event)

    # -- LLMClient -----------------------------------------------------------

    async def generate(self, req: GenerationRequest) -> GenerationResult:
        fut = asyncio.get_running_loop().create_future()
        now = self.clock.now()
        item = QueueItem(req=req, t_enqueue=now, t_ready=now, seq=next(self._seq), future=fut)
        item.prompt_est = self.est.prompt_tokens(req)
        self.queue.append(item)
        self.events.emit(
            "request_enqueue", request_id=req.request_id, task_id=req.task_id, step=req.step,
            queue_len=len(self.queue), inflight=len(self.inflight), prompt_est=item.prompt_est,
        )
        self._wake.set()
        return await fut

    # -- policy hooks --------------------------------------------------------

    def priority(self, item: QueueItem) -> tuple:
        return (item.seq,)

    def quantum_for(self, item: QueueItem) -> int | None:
        return self.quantum

    def effective_priority(self, item: QueueItem, now: float) -> tuple:
        """priority() with optional aging: anything waiting > aging_s jumps ahead (FIFO among aged)."""
        if self.aging_s is not None and now - item.t_ready > self.aging_s:
            return (float("-inf"), item.t_ready)
        return self.priority(item)

    def ordered_queue(self) -> list[QueueItem]:
        now = self.clock.now()
        return sorted(self.queue, key=lambda i: self.effective_priority(i, now))

    def pick_next(self) -> QueueItem | None:
        if len(self.inflight) >= self.K or not self.queue:
            return None
        now = self.clock.now()
        return min(self.queue, key=lambda i: self.effective_priority(i, now))

    def on_complete(self, item: QueueItem, result: GenerationResult) -> None:
        pass

    async def admit_task(self, spec: TaskSpec) -> None:
        if self.max_active_tasks is not None and (
            self._occupied() >= self.max_active_tasks or self._task_waiters
        ):
            fut = asyncio.get_running_loop().create_future()
            self._task_waiters.append(fut)
            await fut  # task_done() reserved a seat for us before resolving
            self._reserved -= 1
        self.active_tasks.add(spec.task_id)
        self.task_arrival.setdefault(spec.task_id, self.clock.now())
        self.task_service.setdefault(spec.task_id, 0.0)

    def task_done(self, spec: TaskSpec, result: TaskResult) -> None:
        self.active_tasks.discard(spec.task_id)
        self.tool_busy.pop(spec.task_id, None)
        self.task_ctx_tokens.pop(spec.task_id, None)
        if self.slots:
            self.slots.task_done(spec.task_id)
        if self.max_active_tasks is not None:
            while self._task_waiters and self._occupied() < self.max_active_tasks:
                fut = self._task_waiters.popleft()
                if not fut.done():
                    self._reserved += 1
                    fut.set_result(None)
        self._wake.set()

    def _occupied(self) -> int:
        return len(self.active_tasks) + self._reserved

    # -- helpers for policies ----------------------------------------------------

    def next_chunk_tokens(self, item: QueueItem) -> int:
        remaining = max(1, item.req.max_tokens - item.gen_tokens)
        q = self.quantum_for(item)
        return min(q, remaining) if q else remaining

    def projected_kv(self, item: QueueItem) -> int:
        """KV tokens this item will hold while its next chunk runs."""
        return item.prompt_est + item.gen_tokens + self.next_chunk_tokens(item)

    def active_kv(self) -> int:
        return sum(i.kv_proj for i in self.inflight.values())

    def kv_budget(self) -> int | None:
        b = self.caps.get("kv_budget_tokens")
        return int(b) if b else None

    def state(self) -> dict[str, Any]:
        s = {
            "queue_len": len(self.queue),
            "inflight": len(self.inflight),
            "active_tasks": len(self.active_tasks),
            "paused": sum(1 for i in self.queue if i.resumed),
            "in_tool": len(self.tool_busy),
            "kv_active_proj": self.active_kv(),
        }
        if self.slots:
            s["kv_used_est"] = self.slots.used()
            s["swapped"] = len(self.slots.swapped)
        return s

    def stats(self) -> dict[str, Any]:
        out = {"policy_overhead_s": self.overhead_s, "decisions": self.decisions,
               "estimators": self.est.snapshot()}
        if self.slots:
            out.update(evictions=dict(self.slots.n_evictions), slot_ops_ms=self.slots.slot_ops_ms)
        return out

    # -- event observation -----------------------------------------------------

    def _on_event(self, kind: str, f: dict[str, Any]) -> None:
        if kind == "tool_start":
            self.tool_busy[f["task_id"]] = (f["tool"], self.clock.now())
        elif kind == "tool_end":
            if (busy := self.tool_busy.pop(f["task_id"], None)) is not None:
                self.est.tool_s[busy[0]].update(self.clock.now() - busy[1])

    # -- machinery -----------------------------------------------------------

    async def start(self) -> None:
        self._loop_task = asyncio.create_task(self._dispatch_loop(), name="scheduler-dispatch")

    async def stop(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
        for t in list(self._running):
            t.cancel()
        await asyncio.gather(*self._running, return_exceptions=True)
        self.events.emit("scheduler_stats", policy=self.registry_name, **self.stats())

    def notify(self) -> None:
        self._wake.set()

    async def _dispatch_loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            # Drop requests whose caller gave up (e.g. task timeout cancelled the await).
            self.queue = [i for i in self.queue if not i.future.done()]
            while self.queue:
                t0 = time.perf_counter()
                item = self.pick_next()
                self.overhead_s += time.perf_counter() - t0
                self.decisions += 1
                if item is None:
                    break
                self.queue.remove(item)
                self.inflight[item.req.request_id] = item
                item.chunk_tokens = self.next_chunk_tokens(item)
                item.kv_proj = self.projected_kv(item)
                if self.slots:
                    try:
                        item.slot = await self.slots.acquire(item.req.task_id, item.kv_proj)
                    except Exception as e:
                        self._finish(item, GenerationResult(request_id=item.req.request_id, status="error",
                                                            error=f"slot acquire: {e!r}", t_done=self.clock.now()))
                        continue
                self._dispatches += 1
                t = asyncio.create_task(self._run_chunk(item))
                self._running.add(t)
                t.add_done_callback(self._running.discard)

    async def _run_chunk(self, item: QueueItem) -> None:
        req = item.req
        now = self.clock.now()
        first = not item.chunks
        if item.resumed:
            wait = now - item.t_ready
            if self._dispatches - 1 > item.dispatch_mark:  # someone else ran while it waited
                item.preemptions += 1
                item.preempt_wait += wait
                self.events.emit("request_resume", request_id=req.request_id, task_id=req.task_id,
                                 wait=wait, chunk=len(item.chunks))
        else:
            self.events.emit(
                "request_dispatch", request_id=req.request_id, task_id=req.task_id, step=req.step,
                queue_wait=now - item.t_enqueue, inflight=len(self.inflight), queue_len=len(self.queue),
                slot=item.slot, kv_proj=item.kv_proj,
            )
        creq = replace(req, request_id=f"{req.request_id}#{len(item.chunks)}", max_tokens=item.chunk_tokens,
                       prefix=item.generated, slot=item.slot)
        try:
            res = await self.backend.generate(creq)
        except Exception as e:  # backends should not raise, but never lose a future
            res = GenerationResult(request_id=creq.request_id, status="error", error=repr(e), t_done=self.clock.now())
        if self.slots and item.slot is not None:
            self.slots.release(item.slot, res)
        exec_s = res.t_done - (res.t_send or now)
        item.service += exec_s
        self.task_service[req.task_id] = self.task_service.get(req.task_id, 0.0) + exec_s
        self.est.observe(creq, res)
        if not first and res.ok:
            item.recompute_tokens += max(0, res.prompt_tokens - res.cached_tokens - 1)
        item.chunks.append(res)
        item.generated += res.text
        item.gen_tokens += res.completion_tokens
        self.inflight.pop(req.request_id, None)
        self.events.emit("chunk_done", request_id=req.request_id, task_id=req.task_id, chunk=len(item.chunks) - 1,
                         slot=res.slot, status=res.status, prompt_tokens=res.prompt_tokens,
                         cached_tokens=res.cached_tokens, completion_tokens=res.completion_tokens,
                         exec_s=exec_s, finish_reason=res.finish_reason)

        stop_hit = next((s for s in req.stop if s in item.generated), None)
        if stop_hit is not None:
            item.generated = item.generated[: item.generated.index(stop_hit)]
        done = (not res.ok or stop_hit is not None or res.finish_reason != "length"
                or item.gen_tokens >= req.max_tokens)
        if done:
            self._finish(item, self._aggregate(item))
        else:
            item.resumed = True
            item.t_ready = self.clock.now()
            item.dispatch_mark = self._dispatches
            item.prompt_est = self.est.prompt_tokens(req)  # prefix is accounted via gen_tokens
            self.queue.append(item)
        self._wake.set()

    def _aggregate(self, item: QueueItem) -> GenerationResult:
        c0, cl = item.chunks[0], item.chunks[-1]
        return GenerationResult(
            request_id=item.req.request_id, text=item.generated, status=cl.status, error=cl.error,
            finish_reason=cl.finish_reason, prompt_tokens=c0.prompt_tokens, cached_tokens=c0.cached_tokens,
            completion_tokens=item.gen_tokens, t_send=c0.t_send,
            t_first_token=next((c.t_first_token for c in item.chunks if c.t_first_token is not None), None),
            t_done=cl.t_done,
            prefill_ms=sum(c.prefill_ms or 0 for c in item.chunks) or None,
            decode_ms=sum(c.decode_ms or 0 for c in item.chunks) or None,
            slot=cl.slot, n_chunks=len(item.chunks), preemptions=item.preemptions,
            preempt_wait=item.preempt_wait, recompute_tokens=item.recompute_tokens,
        )

    def _finish(self, item: QueueItem, res: GenerationResult) -> None:
        req = item.req
        self.inflight.pop(req.request_id, None)
        res.t_enqueue = item.t_enqueue
        first_dispatch = item.chunks[0].t_send if item.chunks else res.t_done
        res.t_dispatch = first_dispatch
        prev_ctx = self.task_ctx_tokens.get(req.task_id, 0)
        reuse_miss = max(0, min(prev_ctx, res.prompt_tokens) - res.cached_tokens) if res.ok and prev_ctx else 0
        if res.ok:
            self.task_ctx_tokens[req.task_id] = res.prompt_tokens + res.completion_tokens
            self.est.observe_output(req, res.completion_tokens)
        self.events.emit(
            "request_done", task_id=req.task_id, step=req.step, tags=req.tags,
            max_tokens=req.max_tokens, reuse_miss_tokens=reuse_miss,
            **{k: v for k, v in res.to_dict().items() if k != "text"}, text_len=len(res.text),
        )
        self.on_complete(item, res)
        if not item.future.done():
            item.future.set_result(res)
        self._wake.set()
