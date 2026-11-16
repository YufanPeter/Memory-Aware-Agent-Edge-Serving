"""LLM clients.

LlamaServerClient talks to llama.cpp's `llama-server` through its native
`/completion` endpoint (raw prompt, streamed). It pins a request to a slot with
`id_slot` and returns the server's own timing and cache counters, which are the
per-request measurements Phase 1 needs.

SlotControl is the KV manager's side channel to the same server: `/tokenize` (exact prompt
sizes, answered on the HTTP thread so it never stalls decoding), the slot actions
save / restore / erase, and `/slots` for reconciliation. Response field names follow
llama.cpp b11960 (server-context.cpp handlers, server-task.cpp to_json).

MockLLM answers from a script and imitates the server's prefix cache per slot,
so the agent and runner can be tested without a model. mockpool.MockPoolLLM adds a
bounded cell pool and the slot actions.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

# llama-server closes idle keep-alive connections after 5 s. A pooled connection reused at
# that moment fails with RemoteProtocolError before any response byte (24 Phase 1 failures),
# so the client drops idle connections first, and retries once if it still loses that race.
KEEPALIVE_EXPIRY_S = 2.0
_RETRYABLE = (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError, httpx.ConnectError)


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
    slot: int = -1             # id_slot from the stream (first data message or final message)
    stop_type: str = ""        # eos | word | limit
    truncated: bool = False
    t_start: float = 0.0       # monotonic
    t_first_token: float = 0.0
    t_end: float = 0.0
    retries: int = 0           # connection-level retries before any response byte

    @property
    def prompt_tokens(self) -> int:
        return self.prompt_n + self.cache_n

    @property
    def kv_tokens(self) -> int:
        """Cells the slot holds after this request: the last sampled token is never decoded."""
        return self.prompt_n + self.cache_n + self.predicted_n - 1 if self.ok else 0

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "text"}
        d["ttft_s"] = self.t_first_token - self.t_start if self.t_first_token else None
        d["latency_s"] = self.t_end - self.t_start
        return d


class LlamaServerClient:
    """`on_first(id_slot)` (attribute, or per call) is called on the first data message of a
    stream, which is how an unpinned request learns its slot. `return_progress=True` makes the
    server send that first message when prompt processing starts instead of at the first token
    (B2 observe mode); it changes no server state."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", seed: int = 0,
                 cache_prompt: bool = True, timeout_s: float = 600.0, return_progress: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.seed = seed
        self.cache_prompt = cache_prompt
        self.return_progress = return_progress
        self.on_first: Callable[[int], None] | None = None
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(timeout_s, connect=10.0),
                                         limits=httpx.Limits(max_connections=256, max_keepalive_connections=256,
                                                             keepalive_expiry=KEEPALIVE_EXPIRY_S))

    async def close(self) -> None:
        await self._client.aclose()

    async def props(self) -> dict[str, Any]:
        r = await self._client.get("/props")
        r.raise_for_status()
        return r.json()

    async def generate(self, prompt: str | list[int], n_predict: int, stop: list[str], slot: int = -1,
                       ignore_eos: bool = False, on_first: Callable[[int], None] | None = None) -> LLMResult:
        """ignore_eos=True decodes exactly n_predict tokens (unless a stop string hits).
        `prompt` may be token ids (drop-to-prefix sends the family prefix ids)."""
        body: dict[str, Any] = {
            "prompt": prompt, "n_predict": n_predict, "stop": stop, "stream": True,
            "cache_prompt": self.cache_prompt, "temperature": 0.0, "seed": self.seed,
        }
        if ignore_eos:
            body["ignore_eos"] = True
        if slot >= 0:
            body["id_slot"] = slot
        if self.return_progress:
            body["return_progress"] = True
        first = on_first or self.on_first
        res = LLMResult(ok=False, t_start=time.monotonic())
        chunks: list[str] = []
        try:
            for attempt in range(2):
                try:
                    await self._stream(body, slot, res, chunks, first)
                    break
                except _NoResponse as e:
                    if attempt == 1:
                        raise e.cause from None
                    res.retries += 1
        except httpx.HTTPError as e:
            res.error = f"{type(e).__name__}: {e}"
        finally:
            res.t_end = time.monotonic()
            res.text = "".join(chunks)
        if not res.ok and res.error is None:
            res.error = "stream ended without a final message"
        return res

    async def _stream(self, body: dict[str, Any], slot: int, res: LLMResult, chunks: list[str],
                      on_first: Callable[[int], None] | None) -> None:
        try:
            cm = self._client.stream("POST", "/completion", json=body)
            r = await cm.__aenter__()
        except _RETRYABLE as e:   # nothing of the response arrived: safe to send again
            raise _NoResponse(e) from e
        try:
            if r.status_code != 200:
                res.error = f"HTTP {r.status_code}: {(await r.aread()).decode(errors='replace')[:500]}"
                return
            seen_first = False
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                msg = json.loads(line[6:])
                if "error" in msg:
                    res.error = json.dumps(msg["error"])[:500]
                    return
                if not seen_first and "id_slot" in msg:
                    seen_first = True
                    res.slot = int(msg["id_slot"])
                    if on_first is not None:
                        on_first(res.slot)
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
        finally:
            await cm.__aexit__(None, None, None)


class _NoResponse(Exception):
    def __init__(self, cause: httpx.HTTPError) -> None:
        super().__init__(str(cause))
        self.cause = cause


@dataclass
class SlotOpResult:
    ok: bool
    op: str                    # save | restore | erase
    slot: int
    tokens: int = 0            # n_saved / n_restored / n_erased
    bytes: int = 0             # n_written / n_read (0 for erase)
    server_ms: float | None = None   # timings.save_ms / restore_ms; erase reports none
    wall_ms: float = 0.0
    error: str | None = None
    status: int = 0            # HTTP status, 0 if no response
    timeout: bool = False      # no answer within op_timeout_s: the server may or may not have run it
    filename: str | None = None


class SlotControl:
    """KV-manager control channel. Its own client without keep-alive (ops are rare, and a
    stale pooled connection must never turn an op into an unknown outcome). Every slot op
    has a wall timeout; on timeout the result says so and the caller treats the slot as
    UNKNOWN (spec §2.4 I4). The server needs `--slot-save-path` for all three actions,
    erase included (post_slots refuses otherwise)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", op_timeout_s: float = 5.0,
                 timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.op_timeout_s = op_timeout_s
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(timeout_s, connect=10.0),
                                         limits=httpx.Limits(max_connections=64, max_keepalive_connections=0))

    async def close(self) -> None:
        await self._client.aclose()

    async def props(self) -> dict[str, Any]:
        r = await self._client.get("/props")
        r.raise_for_status()
        return r.json()

    async def tokenize(self, text: str) -> list[int]:
        """The server's own tokenizer, exactly as /completion tokenizes a string prompt."""
        r = await self._client.post("/tokenize", json={"content": text, "add_special": True,
                                                       "parse_special": True})
        r.raise_for_status()
        return [int(t) for t in r.json()["tokens"]]

    async def slots(self) -> list[dict[str, Any]]:
        """Per-slot `n_prompt_tokens` (= prompt.tokens.size()); absent for a slot that never ran a task."""
        r = await self._client.get("/slots")
        r.raise_for_status()
        return r.json()

    async def slot_save(self, slot: int, filename: str) -> SlotOpResult:
        return await self._op("save", slot, filename)

    async def slot_restore(self, slot: int, filename: str) -> SlotOpResult:
        return await self._op("restore", slot, filename)

    async def slot_erase(self, slot: int) -> SlotOpResult:
        return await self._op("erase", slot, None)

    async def _op(self, op: str, slot: int, filename: str | None) -> SlotOpResult:
        t0 = time.monotonic()
        res = SlotOpResult(ok=False, op=op, slot=slot, filename=filename)
        try:
            r = await asyncio.wait_for(
                self._client.post(f"/slots/{slot}", params={"action": op},
                                  json={"filename": filename} if filename is not None else {}),
                self.op_timeout_s)
        except asyncio.TimeoutError:
            res.timeout, res.error = True, f"timeout after {self.op_timeout_s} s"
        except httpx.HTTPError as e:
            res.error = f"{type(e).__name__}: {e}"
        else:
            res.status = r.status_code
            try:
                d = r.json()
            except ValueError:
                d = {"error": {"message": r.text[:300]}}
            parse_slot_op(res, d)
        res.wall_ms = round((time.monotonic() - t0) * 1000, 3)
        return res


def parse_slot_op(res: SlotOpResult, d: dict[str, Any]) -> SlotOpResult:
    """Fill `res` from a slot-action response body (server-task.cpp:1667–1699)."""
    if res.status != 200 or "error" in d:
        err = d.get("error", d)
        res.error = str(err.get("message", err) if isinstance(err, dict) else err)[:500]
        return res
    t = d.get("timings", {})
    if res.op == "save":
        res.tokens, res.bytes, res.server_ms = int(d["n_saved"]), int(d["n_written"]), float(t.get("save_ms", 0.0))
    elif res.op == "restore":
        res.tokens, res.bytes = int(d["n_restored"]), int(d["n_read"])
        res.server_ms = float(t.get("restore_ms", 0.0))
    else:
        res.tokens = int(d["n_erased"])
    res.ok = True
    return res


def script_completion(script: Callable[[str], str], prompt: str, n_predict: int, stop: list[str],
                      ignore_eos: bool) -> tuple[str, str]:
    """The scripted completion, cut at stop strings and at n_predict (4 chars per token) the way
    the server cuts it. Shared by MockLLM and MockPoolLLM so both drive the agent identically."""
    text = script(prompt)
    stop_type = "eos"
    for s in stop:
        if s in text:
            text, stop_type = text[: text.index(s)], "word"
    if ignore_eos and stop_type == "eos":
        text, stop_type = (text or " x").ljust(n_predict * 4, "x"), "limit"
    return text[: n_predict * 4], stop_type


@dataclass
class MockLLM:
    """Scripted LLM. `script(prompt) -> completion`; completions are cut at stop strings
    like the server does. Token counts are len(text) // 4; the per-slot cache keeps
    prompt + completion, so cache_n is the common prefix with the previous request
    on that slot."""

    script: Callable[[str], str]
    prefill_s_per_token: float = 0.0
    decode_s_per_token: float = 0.0
    slots: dict[int, str] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def close(self) -> None:
        pass

    async def props(self) -> dict[str, Any]:
        return {"build_info": "mock", "total_slots": 64, "model_path": "mock"}

    async def generate(self, prompt: str, n_predict: int, stop: list[str], slot: int = -1,
                       ignore_eos: bool = False, on_first: Callable[[int], None] | None = None) -> LLMResult:
        t0 = time.monotonic()
        cached = self.slots.get(slot, "") if slot >= 0 else ""
        common = 0
        for a, b in zip(cached, prompt):
            if a != b:
                break
            common += 1
        text, stop_type = script_completion(self.script, prompt, n_predict, stop, ignore_eos)
        cache_n, prompt_n, predicted_n = common // 4, (len(prompt) - common) // 4, len(text) // 4
        await asyncio.sleep(prompt_n * self.prefill_s_per_token)
        t_first = time.monotonic()
        if on_first is not None:
            on_first(slot)
        await asyncio.sleep(predicted_n * self.decode_s_per_token)
        if slot >= 0:
            self.slots[slot] = prompt + text
        self.calls.append({"prompt": prompt, "slot": slot, "text": text})
        return LLMResult(ok=True, text=text, prompt_n=prompt_n, cache_n=cache_n, predicted_n=predicted_n,
                         slot=slot, stop_type=stop_type, t_start=t0, t_first_token=t_first,
                         t_end=time.monotonic())
