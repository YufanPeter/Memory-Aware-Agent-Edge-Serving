"""Simulated backend for testing upper layers without a model.

Mimics the llama.cpp behaviour measured on the real server:
- `max_concurrency` slots; each slot keeps the token string it last processed,
  and a new prompt reuses its longest common prefix (prefix cache);
- all slots share a KV pool of `kv_budget_tokens`; idle slots are purged (LRU)
  when a request needs room; if the *active* requests still do not fit, every
  active request fails with `ctx_overflow` (like HTTP 500 on the real server);
- prefill/decode speed degrades linearly with the number of active requests;
- `GenerationRequest.prefix` is continued, so chunked generation works;
- slot erase/save/restore are supported.

Tokens are characters / `chars_per_token`. The reply for agent step k is
`script[k]`, padded with spaces to `completion_tokens` tokens.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from agentserve.backends.base import Backend
from agentserve.core.registry import register
from agentserve.core.types import GenerationRequest, GenerationResult


@dataclass
class _Slot:
    content: str = ""        # token string currently held (prompt + generated)
    active: bool = False
    last_used: float = 0.0
    failed: bool = False
    reserved: int = 0        # tokens an active request will occupy (prompt + output)


def _common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@register("backend", "fake")
class FakeBackend(Backend):
    def __init__(
        self,
        model: dict[str, Any] | None = None,
        max_concurrency: int = 4,
        kv_budget_tokens: int = 16384,
        prefill_tok_s: float = 2000.0,
        decode_tok_s: float = 50.0,
        contention: float = 0.15,
        chars_per_token: float = 4.0,
        script: list[str] | None = None,
        completion_tokens: int = 32,
        swap_tok_s: float = 200000.0,
        time_scale: float = 1.0,
        **kw: Any,
    ) -> None:
        super().__init__(model or {"name": "fake"}, **kw)
        self.params = dict(
            max_concurrency=max_concurrency, kv_budget_tokens=kv_budget_tokens,
            prefill_tok_s=prefill_tok_s, decode_tok_s=decode_tok_s, contention=contention,
            completion_tokens=completion_tokens, swap_tok_s=swap_tok_s, time_scale=time_scale,
        )
        self.script = script or [
            "Thought: I should look this up.\nAction: search[Example]",
            "Thought: I know the answer.\nAction: finish[yes]",
        ]
        self.cpt = chars_per_token
        self.slots = [_Slot() for _ in range(max_concurrency)]
        self.saved: dict[str, str] = {}
        self._pick_lock = asyncio.Lock()
        self.n_purged = 0

    # -- helpers ----------------------------------------------------------------

    def _tok(self, s: str) -> int:
        return int(len(s) / self.cpt)

    def _used(self) -> int:
        return sum(max(self._tok(s.content), s.reserved if s.active else 0) for s in self.slots)

    @staticmethod
    def _render(messages: list[dict[str, str]]) -> str:
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"

    def _choose_slot(self, prompt: str) -> int | None:
        idle = [i for i, s in enumerate(self.slots) if not s.active]
        if not idle:
            return None
        return max(idle, key=lambda i: (_common_prefix(self.slots[i].content, prompt), -self.slots[i].last_used))

    def _make_room(self, need: int, keep: int) -> bool:
        """Purge idle slots (LRU) until `need` more tokens fit; False if impossible."""
        budget = self.params["kv_budget_tokens"]
        while self._used() + need > budget:
            idle = [i for i, s in enumerate(self.slots) if not s.active and i != keep and s.content]
            if not idle:
                return False
            victim = min(idle, key=lambda i: self.slots[i].last_used)
            self.slots[victim].content = ""
            self.n_purged += 1
        return True

    # -- generation -----------------------------------------------------------------

    async def generate(self, req: GenerationRequest) -> GenerationResult:
        p = self.params
        res = GenerationResult(request_id=req.request_id, t_send=self.clock.now())
        prompt = self._render(req.messages) + req.prefix
        target = self.script[min(req.step, len(self.script) - 1)]
        total_chars = max(len(target), int(p["completion_tokens"] * self.cpt))
        full = target.ljust(total_chars)
        done_chars = len(req.prefix)
        n_out = max(0, min(req.max_tokens, self._tok(full[done_chars:]) or (1 if done_chars < len(full) else 0)))
        out_text = full[done_chars: done_chars + int(n_out * self.cpt)]
        finish = "stop" if done_chars + len(out_text) >= len(full) else "length"

        async with self._pick_lock:
            sid = req.slot if req.slot is not None else self._choose_slot(prompt)
            if sid is None or self.slots[sid].active:
                res.status, res.error = "error", "no idle slot"
                res.t_done = self.clock.now()
                return res
            slot = self.slots[sid]
            cached_chars = _common_prefix(slot.content, prompt)
            # the slot's old content beyond the shared prefix is dropped
            slot.content = slot.content[:cached_chars]
            need = self._tok(prompt) - self._tok(slot.content) + n_out
            ok = self._make_room(need, sid)
            slot.active, slot.failed = True, False
            slot.reserved = self._tok(prompt) + n_out
            if not ok:
                # Active requests do not fit: the real server fails every in-flight request.
                for s in self.slots:
                    if s.active:
                        s.failed = True
        res.slot = sid
        try:
            if slot.failed:
                raise _Overflow
            n_active = sum(s.active for s in self.slots)
            slow = 1.0 + p["contention"] * (n_active - 1)
            cached = self._tok(prompt[:cached_chars])
            new_prompt = self._tok(prompt) - cached
            prefill = new_prompt / p["prefill_tok_s"] * slow * p["time_scale"]
            await asyncio.sleep(prefill)
            if slot.failed:
                raise _Overflow
            slot.content = prompt
            res.t_first_token = self.clock.now()
            decode = n_out / p["decode_tok_s"] * slow * p["time_scale"]
            await asyncio.sleep(decode)
            if slot.failed:
                raise _Overflow
            slot.content = prompt + out_text
        except _Overflow:
            slot.content = ""
            res.status, res.error = "ctx_overflow", "Context size has been exceeded."
            res.t_done = self.clock.now()
            return res
        finally:
            slot.active, slot.reserved = False, 0
            slot.last_used = self.clock.now()
        res.text = out_text
        res.prompt_tokens, res.cached_tokens, res.completion_tokens = self._tok(prompt), cached, n_out
        res.prefill_ms, res.decode_ms = prefill * 1e3, decode * 1e3
        res.finish_reason = finish
        res.t_done = self.clock.now()
        return res

    # -- slot control ---------------------------------------------------------------

    async def _swap_sleep(self, tokens: int) -> float:
        d = tokens / self.params["swap_tok_s"] * self.params["time_scale"]
        await asyncio.sleep(d)
        return d * 1e3

    async def slot_erase(self, slot: int) -> dict[str, Any]:
        n = self._tok(self.slots[slot].content)
        self.slots[slot].content = ""
        return {"tokens": n, "ms": 0.0}

    async def slot_save(self, slot: int, name: str) -> dict[str, Any]:
        content = self.slots[slot].content
        self.saved[name] = content
        ms = await self._swap_sleep(self._tok(content))
        return {"tokens": self._tok(content), "ms": ms}

    async def slot_restore(self, slot: int, name: str) -> dict[str, Any]:
        content = self.saved.get(name, "")
        ms = await self._swap_sleep(self._tok(content))
        self._make_room(self._tok(content) - self._tok(self.slots[slot].content), slot)
        self.slots[slot].content = content
        return {"tokens": self._tok(content), "ms": ms}

    async def slot_discard(self, name: str) -> None:
        self.saved.pop(name, None)

    def capabilities(self) -> dict[str, Any]:
        return {
            "max_concurrency": self.params["max_concurrency"],
            "kv_budget_tokens": self.params["kv_budget_tokens"],
            "prefix_cache": True, "continuation": True, "slot_control": True, "slot_save": True,
            "kv_metrics": True, "pid": None,
        }

    async def probe(self) -> dict[str, Any]:
        return {
            "requests_processing": sum(s.active for s in self.slots),
            "kv_tokens_used": self._used(),
            "server_purges": self.n_purged,
        }


class _Overflow(Exception):
    pass
