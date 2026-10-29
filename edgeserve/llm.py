"""LLM clients.

LlamaServerClient talks to llama.cpp's `llama-server` through its native
`/completion` endpoint (raw prompt, streamed). It pins a request to a slot with
`id_slot` and returns the server's own timing and cache counters, which are the
per-request measurements Phase 1 needs.

MockLLM answers from a script and imitates the server's prefix cache per slot,
so the agent and runner can be tested without a model.
"""
from __future__ import annotations

import asyncio
import json
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


class LlamaServerClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", seed: int = 0,
                 cache_prompt: bool = True, timeout_s: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.seed = seed
        self.cache_prompt = cache_prompt
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(timeout_s, connect=10.0),
                                         limits=httpx.Limits(max_connections=256, max_keepalive_connections=256))

    async def close(self) -> None:
        await self._client.aclose()

    async def props(self) -> dict[str, Any]:
        r = await self._client.get("/props")
        r.raise_for_status()
        return r.json()

    async def generate(self, prompt: str, n_predict: int, stop: list[str], slot: int = -1) -> LLMResult:
        body: dict[str, Any] = {
            "prompt": prompt, "n_predict": n_predict, "stop": stop, "stream": True,
            "cache_prompt": self.cache_prompt, "temperature": 0.0, "seed": self.seed,
        }
        if slot >= 0:
            body["id_slot"] = slot
        res = LLMResult(ok=False, t_start=time.monotonic())
        chunks: list[str] = []
        try:
            async with self._client.stream("POST", "/completion", json=body) as r:
                if r.status_code != 200:
                    res.error = f"HTTP {r.status_code}: {(await r.aread()).decode(errors='replace')[:500]}"
                    return res
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    msg = json.loads(line[6:])
                    if "error" in msg:
                        res.error = json.dumps(msg["error"])[:500]
                        return res
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
        return res


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

    async def generate(self, prompt: str, n_predict: int, stop: list[str], slot: int = -1) -> LLMResult:
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
        text = text[: n_predict * 4]
        cache_n, prompt_n, predicted_n = common // 4, (len(prompt) - common) // 4, len(text) // 4
        await asyncio.sleep(prompt_n * self.prefill_s_per_token)
        t_first = time.monotonic()
        await asyncio.sleep(predicted_n * self.decode_s_per_token)
        if slot >= 0:
            self.slots[slot] = prompt + text
        self.calls.append({"prompt": prompt, "slot": slot, "text": text})
        return LLMResult(ok=True, text=text, prompt_n=prompt_n, cache_n=cache_n, predicted_n=predicted_n,
                         slot=slot, stop_type=stop_type, t_start=t0, t_first_token=t_first,
                         t_end=time.monotonic())
