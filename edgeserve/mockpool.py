"""MockPoolLLM: a scripted llama-server with a bounded, unified KV cell pool and slot actions.

It implements both client interfaces the KV manager drives (spec §7):

- generate() / props() / close(), like MockLLM and LlamaServerClient, with the same scripted
  completions (llm.script_completion), so the agent behaves exactly as it does with MockLLM;
- tokenize() / slot_save() / slot_restore() / slot_erase() / slots(), like llm.SlotControl.

What it reproduces from llama.cpp b11960, because the manager's correctness depends on it
(sc = server-context.cpp, kv = llama-kv-cache.cpp; E0a/E0b = results/e0, results/e0b):

- A token is a 4-character chunk (the MockLLM count); token ids are interned chunks.
- Prefix reuse: n_past = LCP(slot tokens, prompt); a fully cached prompt re-evaluates one token
  (sc:4314); the tail beyond n_past is freed (seq_rm, sc:4355). The last sampled token is never
  decoded, so a slot holds s = prompt_n + cache_n + predicted_n - 1 cells. `n_predict` 0
  behaves like 1 (E0b truncate). Generation stops when prompt.n_tokens() + 1 >= n_ctx
  (sc:2188), and a prompt of >= n_ctx tokens is refused with HTTP 400 (sc:4081).
- Cells are placed like find_slot(cont=false) (kv:997-1060): one by one from the head, restarting
  at cell 0 when the head is far above the used count, so holes are reused and the high-water
  mark only falls when the top cells are freed. The attention span is pad256(used_max_p1)
  (kv:1255); it is tracked (max_span, mean_span) but only costs time if t_span > 0.
- When cells run out: purge the first idle slot holding tokens, in slot-id order, one per retry
  (sc:1911); with nothing to purge, shrink the batch to the free cells; with no free cell
  left, fail EVERY processing request with "Context size has been exceeded." and clear its
  slot (sc:4699-4745).
- Requests and slot actions on a busy slot wait until it is idle (deferred tasks). The main
  loop is one lock: prefill takes it per 512-token chunk, decode per token, save/restore for
  op_fixed_s + S*kappa, so slot ops stall every slot. /slots does not take it when
  slots_stale=True (SLOT_GET is answered while a decode runs, sc:3001-3005).
- restore clears the destination slot first; if fewer than n cells are free it answers 400
  and leaves the slot cleared (sc:3323). Restore into holes and into another slot works.
- /slots omits n_prompt_tokens for a slot that never ran a task (sc:814); a restore does not
  count as a task.
- Optional `cache_ram_mib` emulates `--cache-ram` with the default cache_idle_slots (B2): every
  launch saves and clears all idle slots (sc:3071-3086), and the prompt cache's save/load rules
  (server-task.cpp:1740-1900), including the pinned-empty-slot NaN that never reloads (sc:1853).
- Not modelled: n_cmpl children (cleared on release, sc:666; unused), per-slot regions
  (--no-kv-unified), checkpoints, real batching across slots (requests are serialized).

Fault injection: tok_noise (tokenize under- or over-counts), tokenize_delay_s, slots_stale
plus freeze_slots()/thaw_slots(), op_hang={op: seconds} (the op runs late, after the client
gave up at op_timeout_s), and inject_500() (fail only the requests processing now).
Instrumentation: purges, overflows, max_used, max_span, the purge log and `check`, a hook
called with the pool after every cell mutation; its exceptions land in check_errors.
"""
from __future__ import annotations

import asyncio
import bisect
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from edgeserve.llm import LLMResult, SlotOpResult, script_completion

EXCEEDED = "Context size has been exceeded."
RESTORE_FAILED = "Unable to restore slot: No available space in KV cache or invalid slot save file"
BYTES_PER_TOKEN = 57_360        # f16 KV of Qwen2.5-7B per token in a slot file (E0b: n_written)
FILE_HEADER_BYTES = 724         # E0b: 3,000 tokens -> 172,080,724 bytes
PREFILL_CHUNK = 512
SPAN_PAD = 256


def lcp(a: list[int], b: list[int]) -> int:
    """Longest common prefix length; slice comparisons keep it in C."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n            # a[:lo] == b[:lo], a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass
class _Req:
    failed: str | None = None   # set by a fail-all; the request notices at its next step


@dataclass
class _Slot:
    id: int
    tokens: list[int] = field(default_factory=list)
    cells: list[int] = field(default_factory=list)    # cell index of each token position
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    req: _Req | None = None    # is_processing
    id_task: int = -1          # -1: never ran a task
    n_prompt_processed: int = 0
    n_prompt_cached: int = 0
    t_last: float = 0.0

    def idle(self) -> bool:
        return not self.lock.locked() and not getattr(self.lock, "_waiters", None)


class MockPoolLLM:
    def __init__(self, script: Callable[[str], str], capacity: int | None = None, n_slots: int = 64,
                 kappa_s: float = 35e-6, kappa_r: float = 15e-6, op_fixed_s: float = 0.002,
                 t_prefill: float = 2e-5, t_decode: float = 1e-3, tok_noise: int = 0, *,
                 t_span: float = 0.0, cache_prompt: bool = True, return_progress: bool = False,
                 cache_ram_mib: float = 0.0, cache_idle_slots: bool = True, slot_prompt_similarity: float = 0.1,
                 slot_dir: str | Path | None = None, op_timeout_s: float = 5.0, tokenize_delay_s: float = 0.0,
                 slots_stale: bool = False, op_hang: dict[str, float] | None = None) -> None:
        self.script = script
        self.capacity = capacity              # None: cell-unbounded (n_ctx reported as 0)
        self.kappa_s, self.kappa_r, self.op_fixed_s = kappa_s, kappa_r, op_fixed_s
        self.t_prefill, self.t_decode, self.t_span = t_prefill, t_decode, t_span
        self.tok_noise = tok_noise            # > 0: /tokenize drops that many tokens; < 0: adds them
        self.cache_prompt, self.return_progress = cache_prompt, return_progress
        self.cache_ram_mib, self.cache_idle_slots = cache_ram_mib, cache_idle_slots
        self.slot_prompt_similarity = slot_prompt_similarity
        self.slot_dir = Path(slot_dir) if slot_dir is not None else None   # None: files kept in self.files
        self.op_timeout_s, self.tokenize_delay_s = op_timeout_s, tokenize_delay_s
        self.slots_stale = slots_stale
        self.op_hang = dict(op_hang or {})
        self.on_first: Callable[[int], None] | None = None
        self.check: Callable[[MockPoolLLM], None] | None = None

        self._slots = [_Slot(i) for i in range(n_slots)]
        self._loop = asyncio.Lock()
        self._free: list[int] = list(range(capacity)) if capacity else []   # sorted free cell indices
        self._size = capacity or 0
        self._head = 0
        self._used = 0
        self._vocab: dict[str, int] = {}
        self._chunks: list[str] = []
        self._task_id = 0
        self._waiters: list[asyncio.Future] = []
        self._op_tasks: set[asyncio.Task] = set()
        self._frozen: list[dict[str, Any]] | None = None
        self._host: list[list[int]] = []      # --cache-ram prompt cache entries, oldest first

        self.files: dict[str, list[int]] = {}
        self.calls: list[dict[str, Any]] = []
        self.ops: list[dict[str, Any]] = []
        self.log: list[str] = []
        self.purge_log: list[tuple[float, int, int]] = []   # (t, slot, tokens)
        self.check_errors: list[str] = []
        self.n_requests = 0
        self.purges = 0
        self.overflows = 0            # requests failed with "Context size has been exceeded."
        self.overflow_events = 0      # fail-all events
        self.http500 = 0              # requests failed by inject_500()
        self.max_used = 0
        self.max_span = 0
        self._span_sum = 0
        self._span_n = 0
        self.host_saves = self.host_loads = self.host_evictions = self.idle_clears = 0

    # --- inspection ------------------------------------------------------------------------

    @property
    def used(self) -> int:
        return self._used

    @property
    def n_slots(self) -> int:
        return len(self._slots)

    def cells(self, slot: int) -> int:
        return len(self._slots[slot].tokens)

    def tokens_of(self, slot: int) -> list[int]:
        return list(self._slots[slot].tokens)

    def snapshot(self) -> dict[int, int]:
        """Resident cells per non-empty slot."""
        return {s.id: len(s.tokens) for s in self._slots if s.tokens}

    def is_processing(self, slot: int) -> bool:
        return self._slots[slot].req is not None

    def used_max_p1(self) -> int:
        """One past the highest occupied cell: the start of the trailing run of free cells."""
        free, size = self._free, self._size
        target = size - len(free)      # free[i] - i is non-decreasing and at most this
        lo, hi = 0, len(free)
        while lo < hi:
            mid = (lo + hi) // 2
            if free[mid] - mid >= target:
                hi = mid
            else:
                lo = mid + 1
        return free[lo] if lo < len(free) else size

    def span(self) -> int:
        n = max(SPAN_PAD, -(-self.used_max_p1() // SPAN_PAD) * SPAN_PAD)
        return min(self.capacity, n) if self.capacity else n

    @property
    def mean_span(self) -> float:
        return self._span_sum / self._span_n if self._span_n else 0.0

    @property
    def host_tokens(self) -> int:
        return sum(len(e) for e in self._host)

    def counters(self) -> dict[str, Any]:
        return {"requests": self.n_requests, "purges": self.purges, "overflows": self.overflows,
                "overflow_events": self.overflow_events, "http500": self.http500, "max_used": self.max_used,
                "max_span": self.max_span, "mean_span": round(self.mean_span, 1),
                "host_saves": self.host_saves, "host_loads": self.host_loads,
                "host_evictions": self.host_evictions, "idle_clears": self.idle_clears}

    # --- tokens ----------------------------------------------------------------------------

    def _ids(self, text: str) -> list[int]:
        out = []
        for i in range(0, len(text) - len(text) % 4, 4):
            c = text[i:i + 4]
            t = self._vocab.get(c)
            if t is None:
                t = self._vocab[c] = len(self._chunks)
                self._chunks.append(c)
            out.append(t)
        return out

    def detokenize(self, ids: list[int]) -> str:
        return "".join(self._chunks[i] for i in ids)

    # --- cells -----------------------------------------------------------------------------

    def _place(self, n: int) -> list[int]:
        """Cell indices for n new tokens, like find_slot(cont=false). The caller checked room."""
        if n <= 0:
            return []
        if self._head > self._used + 2 * n:
            self._head = 0
        free = self._free
        j = bisect.bisect_left(free, self._head)
        out = free[j:j + n]
        if len(out) == n:
            del free[j:j + n]
        else:
            k = min(n - len(out), j)
            out += free[:k]
            del free[j:]
            del free[:k]
            if len(out) < n:          # cell-unbounded pool: grow
                assert self.capacity is None, "placement without room"
                m = n - len(out)
                out += range(self._size, self._size + m)
                self._size += m
        self._head = out[-1] + 1
        self._used += n
        return out

    def _append(self, sl: _Slot, toks: list[int]) -> None:
        sl.tokens.extend(toks)
        sl.cells.extend(self._place(len(toks)))
        self.max_used = max(self.max_used, self._used)
        sp = self.span()
        self.max_span = max(self.max_span, sp)
        self._span_sum += sp
        self._span_n += 1
        self._mutated()

    def _keep_first(self, sl: _Slot, n: int) -> None:
        """prompt.tokens.keep_first(n) + seq_rm(slot, n, -1)."""
        if n >= len(sl.tokens):
            return
        freed = sl.cells[n:]
        del sl.cells[n:]
        del sl.tokens[n:]
        self._free.extend(freed)
        self._free.sort()
        self._used -= len(freed)
        self._mutated()

    def _room(self, n: int) -> int:
        """Cells available for an n-token ubatch: purge idle slots first, then shrink the batch;
        0 means nothing fits and every processing request has been failed."""
        while True:
            free = n if self.capacity is None else self.capacity - self._used
            if free >= n:
                return n
            victim = next((s for s in self._slots if s.req is None and s.tokens), None)
            if victim is not None:
                self.purges += 1
                self.purge_log.append((time.monotonic(), victim.id, len(victim.tokens)))
                self.log.append(f"purging slot {victim.id} with {len(victim.tokens)} tokens")
                self._keep_first(victim, 0)
                continue
            if free > 0:
                return free
            self.overflow_events += 1
            self.overflows += len(self._fail_processing(EXCEEDED))
            return 0

    def _fail_processing(self, msg: str) -> list[int]:
        failed = []
        for s in self._slots:
            if s.req is not None:
                s.req.failed = msg
                s.req = None            # released at once, as slot.release() does
                self.log.append(f"send_error: slot {s.id}: {msg}")
                self._keep_first(s, 0)
                failed.append(s.id)
        return failed

    def inject_500(self, msg: str = EXCEEDED) -> list[int]:
        """HTTP-500 fault: fail the requests processing right now (not deferred ones)."""
        failed = self._fail_processing(msg)
        self.http500 += len(failed)
        return failed

    def _mutated(self) -> None:
        if self.check is not None:
            try:
                self.check(self)
            except Exception as e:
                self.check_errors.append(f"{type(e).__name__}: {e}"[:500])

    # --- requests --------------------------------------------------------------------------

    async def close(self) -> None:
        for t in list(self._op_tasks):
            t.cancel()
        await asyncio.gather(*self._op_tasks, return_exceptions=True)

    async def props(self) -> dict[str, Any]:
        return {"build_info": "mock-pool", "model_path": "mock", "total_slots": len(self._slots),
                "default_generation_settings": {"n_ctx": self.capacity or 0}}

    async def generate(self, prompt: str | list[int], n_predict: int, stop: list[str], slot: int = -1,
                       ignore_eos: bool = False, on_first: Callable[[int], None] | None = None) -> LLMResult:
        res = LLMResult(ok=False, t_start=time.monotonic())
        if isinstance(prompt, str):
            text_p, toks = prompt, self._ids(prompt)
        else:
            toks = list(prompt)
            text_p = self.detokenize(toks)
        self.n_requests += 1
        if slot >= len(self._slots):
            res.error, res.t_end = 'HTTP 400: {"error": {"code": 400, "message": "Invalid slot ID"}}', time.monotonic()
            return res
        sl = await self._take(slot, toks)
        req = _Req()
        try:
            await self._run(sl, req, res, toks, text_p, n_predict, stop, slot >= 0, ignore_eos,
                            on_first or self.on_first)
        finally:
            if sl.req is req:
                sl.req = None
            sl.t_last = time.monotonic()
            sl.lock.release()
            self._wake()
            res.t_end = time.monotonic()
            self.calls.append({"prompt": text_p, "slot": slot, "id_slot": sl.id, "text": res.text,
                               "n_predict": n_predict, "ids": not isinstance(prompt, str), "ok": res.ok})
        return res

    async def _take(self, slot: int, toks: list[int]) -> _Slot:
        if slot >= 0:
            sl = self._slots[slot]
            await sl.lock.acquire()
            return sl
        while True:
            sl = self._select(toks)
            if sl is not None:
                await sl.lock.acquire()   # idle and uncontended: returns without yielding
                return sl
            fut = asyncio.get_running_loop().create_future()
            self._waiters.append(fut)
            await fut

    def _wake(self) -> None:
        for f in self._waiters:
            if not f.done():
                f.set_result(None)
        self._waiters.clear()

    def _select(self, toks: list[int]) -> _Slot | None:
        """Unpinned slot choice: best LCP similarity above the threshold, else least recently used."""
        idle = [s for s in self._slots if s.idle()]
        if not idle:
            return None
        best, f_best = None, 0.0
        if self.slot_prompt_similarity != 0.0 and toks:
            for s in idle:
                if s.tokens:
                    f = lcp(s.tokens, toks) / len(toks)
                    if f > f_best and f > self.slot_prompt_similarity:
                        best, f_best = s, f
        return best or min(idle, key=lambda s: (s.t_last, s.id))

    def _launch(self, sl: _Slot, req: _Req, toks: list[int], pinned: bool) -> str | int:
        """Slot launch inside the main loop: prompt-cache update, n_ctx check, prefix reuse.
        Returns n_past, or an error string."""
        if self.cache_ram_mib > 0:
            self._cache_update(sl, toks, pinned)
        self._task_id += 1
        sl.id_task = self._task_id
        sl.req = req
        if self.cache_ram_mib > 0 and self.cache_idle_slots:
            for s in self._slots:     # [TAG_IDLE_SLOT_CLEAR]
                if s.req is None and s.tokens:
                    self._prompt_save(s)
                    self._keep_first(s, 0)
                    self.idle_clears += 1
        if self.capacity and len(toks) >= self.capacity:
            body = {"error": {"code": 400, "message": f"request ({len(toks)} tokens) exceeds the available context "
                              f"size ({self.capacity} tokens), try increasing it", "type": "exceed_context_size_error",
                              "n_prompt_tokens": len(toks), "n_ctx": self.capacity}}
            return "HTTP 400: " + json.dumps(body)
        n_past = lcp(sl.tokens, toks) if self.cache_prompt else 0
        if n_past == len(toks) and n_past > 0:
            n_past -= 1               # at least one token is evaluated
        sl.n_prompt_cached, sl.n_prompt_processed = n_past, 0
        self._keep_first(sl, n_past)
        return n_past

    async def _run(self, sl: _Slot, req: _Req, res: LLMResult, toks: list[int], text_p: str, n_predict: int,
                   stop: list[str], pinned: bool, ignore_eos: bool, on_first: Callable[[int], None] | None) -> None:
        async with self._loop:
            launched = self._launch(sl, req, toks, pinned)
        if isinstance(launched, str):
            res.error = launched
            return
        n_past = launched
        first_sent = False
        if self.return_progress:       # 0% progress message at prompt start
            first_sent = True
            res.slot = sl.id
            if on_first is not None:
                on_first(sl.id)

        text, stop_type = script_completion(self.script, text_p, n_predict, stop, ignore_eos)
        P = len(toks)
        full = self._ids(text_p + text)
        gen = full[P:]
        g = max(1, len(gen))
        truncated = False
        if self.capacity and P + g >= self.capacity:   # stop at prompt.n_tokens() + 1 >= n_ctx
            g, truncated, stop_type = self.capacity - P, True, "limit"
            text = self.detokenize(full[:P + g])[len(text_p):] if g < len(gen) else text

        pos = n_past
        while pos < P and not req.failed:
            async with self._loop:
                if req.failed:
                    break
                k = self._room(min(PREFILL_CHUNK, P - pos))
                if k == 0 or req.failed:
                    break
                self._append(sl, toks[pos:pos + k])
                sl.n_prompt_processed += k
                pos += k
                await asyncio.sleep(k * self._per_token(self.t_prefill))
        if not req.failed:
            res.t_first_token = time.monotonic()
            if not first_sent:
                first_sent = True
                res.slot = sl.id
                if on_first is not None:
                    on_first(sl.id)
        for i in range(g - 1):       # each decode stores the previously sampled token
            if req.failed:
                break
            async with self._loop:
                if req.failed or self._room(1) == 0 or req.failed:
                    break
                self._append(sl, [gen[i]])
                await asyncio.sleep(self._per_token(self.t_decode))
        if req.failed:
            res.slot = sl.id if first_sent else -1
            res.error = json.dumps({"code": 500, "message": req.failed, "type": "server_error"})
            return
        res.ok, res.text = True, text
        res.prompt_n, res.cache_n, res.predicted_n = P - n_past, n_past, g
        res.prompt_ms = round((P - n_past) * self.t_prefill * 1e3, 3)
        res.predicted_ms = round((g - 1) * self.t_decode * 1e3, 3)
        res.slot, res.stop_type, res.truncated = sl.id, stop_type, truncated

    def _per_token(self, t: float) -> float:
        return t + self.t_span * self.span() / 1000 if self.t_span else t

    # --- --cache-ram emulation (B2) ---------------------------------------------------------

    def _cache_update(self, sl: _Slot, toks: list[int], pinned: bool) -> None:
        """get_available_slot's update_cache rule (sc:1800-1900)."""
        if self.slot_prompt_similarity == 0.0:
            update = not pinned
        elif sl.tokens:
            n = lcp(sl.tokens, toks)
            f_sim = n / len(toks) if toks else 0.0
            # below the similarity threshold f_sim_best stays 0, so f_keep = 0 and the cache updates
            update = (n / len(sl.tokens) if f_sim > self.slot_prompt_similarity else 0.0) < 0.5
        else:
            update = not pinned       # pinned empty slot: f_keep = 0/0 = NaN, never reloads (sc:1853)
        if update:
            self._prompt_save(sl)
            if not self._prompt_load(sl, toks):
                self._keep_first(sl, 0)

    def _prompt_save(self, sl: _Slot) -> bool:
        toks = sl.tokens
        if not toks or any(lcp(e, toks) == len(toks) for e in self._host):
            return False
        limit = self.cache_ram_mib * 2**20
        new = len(toks) * BYTES_PER_TOKEN
        if new > limit:
            return False
        self._host = [e for e in self._host if lcp(e, toks) != len(e)]   # obsolete: contained in this one
        while self._host and (self.host_tokens * BYTES_PER_TOKEN) + new > limit:
            self.log.append(f"making room for prompt cache entry, removing oldest entry ({len(self._host[0])} tokens)")
            self._host.pop(0)
            self.host_evictions += 1
        self._host.append(list(toks))
        self.host_saves += 1
        return True

    def _prompt_load(self, sl: _Slot, toks: list[int]) -> bool:
        n = lcp(sl.tokens, toks)
        f_keep_best = n / len(sl.tokens) if sl.tokens else -1.0
        f_sim_best = n / len(toks) if toks else 0.0
        best = None
        for i, e in enumerate(self._host):
            m = lcp(e, toks)
            f_keep, f_sim = m / len(e), (m / len(toks) if toks else 0.0)
            if f_keep < 0.25:
                continue
            if f_keep_best < f_keep and f_sim_best < f_sim:
                best, f_keep_best, f_sim_best = i, f_keep, f_sim
        if best is None:
            return True
        entry = self._host[best]
        self._keep_first(sl, 0)
        if self.capacity and self.capacity - self._used < len(entry):
            return False
        self._append(sl, entry)
        del self._host[best]
        self.host_loads += 1
        return True

    # --- SlotControl interface -------------------------------------------------------------

    async def tokenize(self, text: str) -> list[int]:
        if self.tokenize_delay_s:
            await asyncio.sleep(self.tokenize_delay_s)
        ids = self._ids(text)
        if self.tok_noise > 0:
            return ids[:max(0, len(ids) - self.tok_noise)]
        if self.tok_noise < 0:
            return ids + [0] * (-self.tok_noise)
        return ids

    async def slots(self) -> list[dict[str, Any]]:
        if self._frozen is not None:
            return [dict(d) for d in self._frozen]
        if self.slots_stale:
            return self._slots_json()
        async with self._loop:
            return self._slots_json()

    def freeze_slots(self) -> None:
        """Answer /slots from the state now until thaw_slots() (a deterministic stale snapshot)."""
        self._frozen = self._slots_json()

    def thaw_slots(self) -> None:
        self._frozen = None

    def _slots_json(self) -> list[dict[str, Any]]:
        out = []
        for s in self._slots:
            d: dict[str, Any] = {"id": s.id, "n_ctx": self.capacity or 0, "speculative": False,
                                 "is_processing": s.req is not None}
            if s.id_task >= 0:
                d.update(id_task=s.id_task, n_prompt_tokens=len(s.tokens),
                         n_prompt_tokens_processed=s.n_prompt_processed, n_prompt_tokens_cache=s.n_prompt_cached)
            out.append(d)
        return out

    async def slot_save(self, slot: int, filename: str) -> SlotOpResult:
        return await self._op("save", slot, filename)

    async def slot_restore(self, slot: int, filename: str) -> SlotOpResult:
        return await self._op("restore", slot, filename)

    async def slot_erase(self, slot: int) -> SlotOpResult:
        return await self._op("erase", slot, None)

    async def _op(self, op: str, slot: int, filename: str | None) -> SlotOpResult:
        """The client side: the server-side op is its own task, so a client timeout or
        cancellation does not stop it (as with the real server once the task has started)."""
        t0 = time.monotonic()
        task = asyncio.ensure_future(self._server_op(op, slot, filename))
        self._op_tasks.add(task)
        task.add_done_callback(self._op_tasks.discard)
        try:
            res = replace(await asyncio.wait_for(asyncio.shield(task), self.op_timeout_s))
        except asyncio.TimeoutError:
            res = SlotOpResult(ok=False, op=op, slot=slot, filename=filename, timeout=True,
                               error=f"timeout after {self.op_timeout_s} s")
        res.wall_ms = round((time.monotonic() - t0) * 1000, 3)
        return res

    async def _server_op(self, op: str, slot: int, filename: str | None) -> SlotOpResult:
        res = SlotOpResult(ok=False, op=op, slot=slot, filename=filename, status=400)
        if not 0 <= slot < len(self._slots):
            res.error = "Invalid slot ID"
            return res
        if op != "erase" and (not filename or any(c in filename for c in "/\\") or filename.startswith(".")):
            res.error = "Invalid filename"
            return res
        sl = self._slots[slot]
        async with sl.lock:                       # deferred while the slot is processing
            if self.op_hang.get(op):
                await asyncio.sleep(self.op_hang[op])
            async with self._loop:                # declined during a decode, run between batches
                dur = self.op_fixed_s
                if op == "erase":
                    res.tokens = len(sl.tokens)
                    self._keep_first(sl, 0)
                elif op == "save":
                    toks = list(sl.tokens)
                    self._write_file(filename or "", toks)
                    dur += self.kappa_s * len(toks)
                    res.tokens, res.bytes = len(toks), len(toks) * BYTES_PER_TOKEN + FILE_HEADER_BYTES
                else:
                    toks_r = self._read_file(filename or "")
                    self._keep_first(sl, 0)
                    if toks_r is None or (self.capacity is not None and self.capacity - self._used < len(toks_r)):
                        res.error = RESTORE_FAILED
                    else:
                        self._append(sl, toks_r)
                        dur += self.kappa_r * len(toks_r)
                        res.tokens, res.bytes = len(toks_r), len(toks_r) * BYTES_PER_TOKEN + FILE_HEADER_BYTES
                await asyncio.sleep(dur)
        self._wake()                              # unpinned requests may take this slot now
        if res.error is None:
            res.ok, res.status = True, 200
            res.server_ms = None if op == "erase" else round(dur * 1e3, 3)
        self.ops.append({"t": time.monotonic(), "op": op, "slot": slot, "ok": res.ok, "tokens": res.tokens,
                         "server_ms": res.server_ms, "error": res.error})
        return res

    def _write_file(self, filename: str, toks: list[int]) -> None:
        if self.slot_dir is None:
            self.files[filename] = toks
        else:
            (self.slot_dir / filename).write_text(json.dumps(toks), encoding="utf-8", newline="\n")

    def _read_file(self, filename: str) -> list[int] | None:
        if self.slot_dir is None:
            toks = self.files.get(filename)
            return list(toks) if toks is not None else None
        path = self.slot_dir / filename
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
