"""LLM clients.

LlamaServerClient talks to llama.cpp's `llama-server` through its native
`/completion` endpoint (raw prompt, streamed). It pins a request to a slot with
`id_slot` and returns the server's own timing and cache counters, which are the
per-request measurements Phase 1 needs.

Slot control (save / restore / erase, and fork on our patched server) goes through a
second HTTP client without keep-alive: a control request must never wait behind, or race
with, the long-lived streaming connections of agents that are idle in tool calls.

MockLLM answers from a script and imitates the server's prefix cache per slot,
so the agent and runner can be tested without a model. Given a pool size it also
imitates how llama-server uses one shared KV pool (see MockLLM).
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx


@dataclass
class LLMResult:
    ok: bool
    text: str = ""
    error: str | None = None
    prompt_n: int = 0          # prompt tokens actually evaluated (not served from cache)
    cache_n: int = 0           # prompt tokens reused from the slot's KV cache
    predicted_n: int = 0       # generated tokens
    prompt_ms: float = 0.0
    predicted_ms: float = 0.0
    slot: int = -1
    stop_type: str = ""        # eos | word | limit
    truncated: bool = False
    t_start: float = 0.0       # monotonic
    t_first_token: float = 0.0
    t_end: float = 0.0

    @property
    def prompt_tokens(self) -> int:
        return self.prompt_n + self.cache_n

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "text"}
        d["ttft_s"] = self.t_first_token - self.t_start if self.t_first_token else None
        d["latency_s"] = self.t_end - self.t_start
        return d


@dataclass
class SlotOpResult:
    op: str                    # save | restore | erase | fork
    slot: int
    ok: bool
    wall_ms: float = 0.0       # client-side round trip, including any wait in the server's queue
    server_ms: float = 0.0     # save_ms / restore_ms reported by the server (0 for erase / fork)
    n_tokens: int = 0          # tokens saved / restored / erased / forked
    n_bytes: int = 0           # file bytes written or read
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _slot_result(op: str, slot: int, wall_ms: float, d: dict[str, Any]) -> SlotOpResult:
    t = d.get("timings", {})
    return SlotOpResult(
        op=op, slot=slot, ok=True, wall_ms=round(wall_ms, 2),
        server_ms=float(t.get("save_ms", t.get("restore_ms", t.get("fork_ms", t.get("trim_ms", 0.0))))),
        n_tokens=int(d.get("n_saved", d.get("n_restored", d.get("n_erased", d.get("n_forked",
                                                                                 d.get("n_trimmed", 0)))))),
        n_bytes=int(d.get("n_written", d.get("n_read", 0))))


class LlamaServerClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", seed: int = 0,
                 cache_prompt: bool = True, timeout_s: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.seed = seed
        self.cache_prompt = cache_prompt
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(timeout_s, connect=10.0),
                                         limits=httpx.Limits(max_connections=256, max_keepalive_connections=256))
        self._ctl = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(timeout_s, connect=10.0),
                                      limits=httpx.Limits(max_keepalive_connections=0))

    async def close(self) -> None:
        await self._client.aclose()
        await self._ctl.aclose()

    async def _slot_action(self, slot: int, op: str, body: dict[str, Any]) -> SlotOpResult:
        t0 = time.monotonic()
        try:
            r = await self._ctl.post(f"/slots/{slot}", params={"action": op}, json=body)
            wall = (time.monotonic() - t0) * 1000
            if r.status_code != 200:
                return SlotOpResult(op, slot, False, round(wall, 2), error=f"HTTP {r.status_code}: {r.text[:300]}")
            return _slot_result(op, slot, wall, r.json())
        except httpx.HTTPError as e:
            return SlotOpResult(op, slot, False, round((time.monotonic() - t0) * 1000, 2),
                                error=f"{type(e).__name__}: {e}")

    async def slot_save(self, slot: int, filename: str) -> SlotOpResult:
        """Write the slot's KV to <--slot-save-path>/filename. The slot keeps its KV."""
        return await self._slot_action(slot, "save", {"filename": filename})

    async def slot_restore(self, slot: int, filename: str) -> SlotOpResult:
        return await self._slot_action(slot, "restore", {"filename": filename})

    async def slot_erase(self, slot: int) -> SlotOpResult:
        return await self._slot_action(slot, "erase", {})

    async def slot_fork(self, slot: int, src: int, n_tokens: int) -> SlotOpResult:
        """Share the first n_tokens of slot `src` with `slot` without copying (patched server only)."""
        return await self._slot_action(slot, "fork", {"src": src, "n_tokens": n_tokens})

    async def slot_trim(self, slot: int, ranges: list[tuple[int, int]]) -> SlotOpResult:
        """Delete token ranges [p0, p1) from the slot's KV and shift later tokens down, without
        recomputing anything (patched server only). Indices refer to the slot's cached tokens."""
        return await self._slot_action(slot, "trim", {"ranges": [list(r) for r in ranges]})

    async def tokenize(self, text: str) -> list[tuple[int, bytes]]:
        """(token id, its bytes) for `text`, tokenized exactly as /completion does (special tokens
        parsed, model's BOS rule applied)."""
        r = await self._ctl.post("/tokenize", json={"content": text, "add_special": True,
                                                    "parse_special": True, "with_pieces": True})
        r.raise_for_status()
        out = []
        for t in r.json()["tokens"]:
            piece = t["piece"]
            out.append((int(t["id"]), piece.encode("utf-8") if isinstance(piece, str) else bytes(piece)))
        return out

    async def slots_state(self) -> list[dict[str, Any]]:
        r = await self._ctl.get("/slots")
        r.raise_for_status()
        return r.json()

    async def props(self) -> dict[str, Any]:
        r = await self._client.get("/props")
        r.raise_for_status()
        return r.json()

    async def generate(self, prompt: str, n_predict: int, stop: list[str], slot: int = -1,
                       ignore_eos: bool = False) -> LLMResult:
        """ignore_eos=True decodes exactly n_predict tokens (unless a stop string hits)."""
        body: dict[str, Any] = {
            "prompt": prompt, "n_predict": n_predict, "stop": stop, "stream": True,
            "cache_prompt": self.cache_prompt, "temperature": 0.0, "seed": self.seed,
        }
        if ignore_eos:
            body["ignore_eos"] = True
        if slot >= 0:
            body["id_slot"] = slot
        t_start = time.monotonic()
        for attempt in range(2):
            res, received = await self._generate_once(body, slot)
            # Keep-alive race (Phase 1: 9 of 5,890 tasks): the server closes an idle pooled
            # connection just as the client reuses it. Nothing was processed, so retry once.
            if res.ok or received or attempt == 1 or not (res.error or "").startswith("RemoteProtocolError"):
                break
        res.t_start = t_start
        return res

    async def _generate_once(self, body: dict[str, Any], slot: int) -> tuple[LLMResult, bool]:
        """One streamed request. Returns the result and whether any response bytes arrived."""
        res = LLMResult(ok=False, t_start=time.monotonic())
        chunks: list[str] = []
        received = False
        try:
            async with self._client.stream("POST", "/completion", json=body) as r:
                received = True
                if r.status_code != 200:
                    res.error = f"HTTP {r.status_code}: {(await r.aread()).decode(errors='replace')[:500]}"
                    res.t_end = time.monotonic()
                    return res, received
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    msg = json.loads(line[6:])
                    if "error" in msg:
                        res.error = json.dumps(msg["error"])[:500]
                        res.t_end = time.monotonic()
                        return res, received
                    if msg.get("content"):
                        if not res.t_first_token:
                            res.t_first_token = time.monotonic()
                        chunks.append(msg["content"])
                    if msg.get("stop"):
                        t = msg.get("timings", {})
                        res.prompt_n = int(t.get("prompt_n", 0))
                        res.cache_n = int(t.get("cache_n", msg.get("tokens_cached", 0)))
                        res.predicted_n = int(t.get("predicted_n", 0))
                        res.prompt_ms = float(t.get("prompt_ms", 0.0))
                        res.predicted_ms = float(t.get("predicted_ms", 0.0))
                        res.slot = int(msg.get("id_slot", slot))
                        res.stop_type = str(msg.get("stop_type", ""))
                        res.truncated = bool(msg.get("truncated", False))
                        res.ok = True
        except httpx.HTTPError as e:
            res.error = f"{type(e).__name__}: {e}"
        finally:
            res.t_end = time.monotonic()
            res.text = "".join(chunks)
        if not res.ok and res.error is None:
            res.error = "stream ended without a final message"
        return res, received


@dataclass
class MockLLM:
    """Scripted LLM. `script(prompt) -> completion`; completions are cut at stop strings
    like the server does. Token counts are len(text) // 4; the per-slot cache keeps
    prompt + completion, so cache_n is the common prefix with the previous request
    on that slot.

    With `pool_tokens > 0` the slots share one KV pool of that many tokens, and the mock
    reproduces what Phase 1 E0a measured on llama-server b11960:
    - a request that does not fit first makes room by purging idle slots, lowest slot id first;
    - if the requests in flight still exceed the pool, *every* in-flight request fails with
      "Context size has been exceeded".
    `tax_s_per_kv_token` adds decode time per generated token for every token resident in the
    pool (the unified-pool attention span, docs/phase2_notes.md). `slot_op_s_per_token` is
    the time a save or restore takes per token."""

    script: Callable[[str], str]
    prefill_s_per_token: float = 0.0
    decode_s_per_token: float = 0.0
    pool_tokens: int = 0
    tax_s_per_kv_token: float = 0.0
    slot_op_s_per_token: float = 0.0
    slots: dict[int, str] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    shared: dict[int, str] = field(default_factory=dict)   # slot -> prefix it shares with another slot (fork)
    purges: list[int] = field(default_factory=list)       # slots the "server" purged itself
    overflows: int = 0                                     # times the in-flight requests overflowed
    _active: dict[int, int] = field(default_factory=dict)  # slot -> tokens its request will hold
    _failed: set[int] = field(default_factory=set)

    async def close(self) -> None:
        pass

    async def props(self) -> dict[str, Any]:
        return {"build_info": "mock", "total_slots": 64, "model_path": "mock",
                "default_generation_settings": {"n_ctx": self.pool_tokens}}

    @staticmethod
    def tokens(text: str) -> int:
        return len(text) // 4

    def _own(self, slot: int, text: str) -> int:
        """Tokens of `text` in `slot` that are not shared with another slot."""
        pre = self.shared.get(slot, "")
        return self.tokens(text) - (self.tokens(pre) if pre and text.startswith(pre) else 0)

    def resident_tokens(self) -> int:
        """Tokens held in the pool: idle slots' KV plus what the requests in flight will hold.
        A forked prefix counts once (in the slot it came from)."""
        idle = sum(self._own(s, t) for s, t in self.slots.items() if s not in self._active)
        return idle + sum(self._active.values())

    def _admit(self, slot: int, need: int) -> bool:
        """Place a request of `need` tokens the way llama-server does; False on overflow."""
        self._active[slot] = need   # from now on the slot's old KV counts as part of `need`
        if not self.pool_tokens:
            return True
        for s in sorted(self.slots):
            if self.resident_tokens() <= self.pool_tokens:
                break
            if s not in self._active:
                del self.slots[s]
                self.shared.pop(s, None)
                self.purges.append(s)
        if sum(self._active.values()) > self.pool_tokens:
            self.overflows += 1
            self._failed |= set(self._active)
            return False
        return True

    async def generate(self, prompt: str, n_predict: int, stop: list[str], slot: int = -1,
                       ignore_eos: bool = False) -> LLMResult:
        t0 = time.monotonic()
        cached = self.slots.get(slot, "") if slot >= 0 else ""
        common = 0
        for a, b in zip(cached, prompt):
            if a != b:
                break
            common += 1
        text = self.script(prompt)
        stop_type = "eos"
        for s in stop:
            if s in text:
                text, stop_type = text[: text.index(s)], "word"
        if ignore_eos and stop_type == "eos":
            text, stop_type = (text or " x").ljust(n_predict * 4, "x"), "limit"
        text = text[: n_predict * 4]
        cache_n, prompt_n, predicted_n = common // 4, (len(prompt) - common) // 4, len(text) // 4
        key = slot if slot >= 0 else -1 - len(self.calls)   # unpinned requests get a private key
        try:
            if slot >= 0 and not prompt.startswith(self.shared.get(slot, "")):
                self.shared.pop(slot, None)   # the request diverges inside the shared prefix
            if not self._admit(key, self._own(slot, prompt + text)):
                return LLMResult(ok=False, error='{"message": "Context size has been exceeded."}',
                                 slot=slot, t_start=t0, t_end=time.monotonic())
            await asyncio.sleep(prompt_n * self.prefill_s_per_token)
            t_first = time.monotonic()
            per_token = self.decode_s_per_token + self.tax_s_per_kv_token * self.resident_tokens()
            await asyncio.sleep(predicted_n * per_token)
            if key in self._failed:
                self.slots.pop(slot, None)
                self.shared.pop(slot, None)
                return LLMResult(ok=False, error='{"message": "Context size has been exceeded."}',
                                 slot=slot, t_start=t0, t_end=time.monotonic())
            if slot >= 0:
                self.slots[slot] = prompt + text
        finally:
            self._active.pop(key, None)
            self._failed.discard(key)
        self.calls.append({"prompt": prompt, "slot": slot, "text": text})
        return LLMResult(ok=True, text=text, prompt_n=prompt_n, cache_n=cache_n, predicted_n=predicted_n,
                         slot=slot, stop_type=stop_type, t_start=t0, t_first_token=t_first,
                         t_end=time.monotonic())

    async def slot_save(self, slot: int, filename: str) -> SlotOpResult:
        text = self.slots.get(slot, "")
        await asyncio.sleep(self.tokens(text) * self.slot_op_s_per_token)
        self.files[filename] = text
        return SlotOpResult("save", slot, True, n_tokens=self.tokens(text), n_bytes=len(text))

    async def slot_restore(self, slot: int, filename: str) -> SlotOpResult:
        if filename not in self.files:
            return SlotOpResult("restore", slot, False, error="file not found")
        text = self.files[filename]
        self.shared.pop(slot, None)
        others = self.resident_tokens() - (self._own(slot, self.slots.get(slot, "")) if slot not in self._active else 0)
        if self.pool_tokens and others + self.tokens(text) > self.pool_tokens:
            return SlotOpResult("restore", slot, False, error="failed to find free space in the KV cache")
        await asyncio.sleep(self.tokens(text) * self.slot_op_s_per_token)
        self.slots[slot] = text
        return SlotOpResult("restore", slot, True, n_tokens=self.tokens(text), n_bytes=len(text))

    async def slot_erase(self, slot: int) -> SlotOpResult:
        text = self.slots.pop(slot, "")
        self.shared.pop(slot, None)
        return SlotOpResult("erase", slot, True, n_tokens=self.tokens(text))

    _MOCK_TOKEN = re.compile(r" ?[^\s]+|\s")

    async def tokenize(self, text: str) -> list[tuple[int, bytes]]:
        """Toy tokenizer for the mock: a token is an optional space plus a run of non-spaces."""
        return [(hash(p) % 100_000, p.encode("utf-8")) for p in self._MOCK_TOKEN.findall(text)]

    async def slot_trim(self, slot: int, ranges: list[tuple[int, int]]) -> SlotOpResult:
        """Like the patched server: delete token ranges (mock tokenization) from the slot."""
        if slot in self._active:
            return SlotOpResult("trim", slot, False, error="slot busy")
        text = self.slots.get(slot, "")
        pieces = self._MOCK_TOKEN.findall(text)
        starts = [0]
        for piece in pieces:
            starts.append(starts[-1] + len(piece))
        out, last, n = [], 0, 0
        for p0, p1 in sorted(ranges):
            if not 0 <= p0 < p1 <= len(pieces):
                return SlotOpResult("trim", slot, False, error="ranges must be within the slot's tokens")
            out.append(text[last:starts[p0]])
            last, n = starts[p1], n + p1 - p0
        out.append(text[last:])
        self.slots[slot] = "".join(out)
        return SlotOpResult("trim", slot, True, n_tokens=n)

    async def slot_fork(self, slot: int, src: int, n_tokens: int) -> SlotOpResult:
        """Like the patched server: `slot` shares the first n_tokens of `src` without a copy."""
        text = self.slots.get(src, "")
        if n_tokens > self.tokens(text):
            return SlotOpResult("fork", slot, False, error="Source slot holds fewer tokens than requested")
        pre = text[: n_tokens * 4]
        self.slots[slot] = pre
        self.shared[slot] = pre
        return SlotOpResult("fork", slot, True, n_tokens=n_tokens)

    async def slots_state(self) -> list[dict[str, Any]]:
        return [{"id": s, "n_tokens": self.tokens(t), "is_processing": s in self._active}
                for s, t in sorted(self.slots.items())]
