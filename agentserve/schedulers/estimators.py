"""Online estimators shared by scheduling policies. Everything is learned from
observed results, so no policy needs runtime- or model-specific constants."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from agentserve.core.types import GenerationRequest, GenerationResult


class EMA:
    def __init__(self, init: float, alpha: float = 0.1) -> None:
        self.value, self.alpha, self.n = init, alpha, 0

    def update(self, x: float) -> None:
        self.n += 1
        # plain mean for the first few samples, then exponential smoothing
        a = max(self.alpha, 1.0 / self.n)
        self.value += a * (x - self.value)


class Estimators:
    def __init__(self, chars_per_token: float = 3.6, msg_overhead_tokens: int = 6,
                 decode_ms_per_tok: float = 15.0, prefill_ms_per_tok: float = 1.0) -> None:
        self.chars_per_token = EMA(chars_per_token)
        self.msg_overhead = msg_overhead_tokens
        self.decode_ms = EMA(decode_ms_per_tok)
        self.prefill_ms = EMA(prefill_ms_per_tok)
        self.out_tokens: dict[str, EMA] = {}
        self.tool_s: dict[str, EMA] = defaultdict(lambda: EMA(1.0, 0.2))

    # -- tokens ------------------------------------------------------------------

    @staticmethod
    def _chars(req: GenerationRequest) -> int:
        return sum(len(m["content"]) for m in req.messages) + len(req.prefix)

    def prompt_tokens(self, req: GenerationRequest) -> int:
        return int(self._chars(req) / self.chars_per_token.value) + self.msg_overhead * len(req.messages)

    def observe(self, req: GenerationRequest, res: GenerationResult) -> None:
        if not res.ok or res.prompt_tokens <= 0:
            return
        body = res.prompt_tokens - self.msg_overhead * len(req.messages)
        if body > 0:
            self.chars_per_token.update(self._chars(req) / body)
        new_prompt = res.prompt_tokens - res.cached_tokens
        if res.prefill_ms and new_prompt > 16:
            self.prefill_ms.update(res.prefill_ms / new_prompt)
        if res.decode_ms and res.completion_tokens > 1:
            self.decode_ms.update(res.decode_ms / res.completion_tokens)

    # -- output length -----------------------------------------------------------

    @staticmethod
    def _key(req: GenerationRequest) -> str:
        return str(req.tags.get("mix", "_"))

    def predicted_output(self, req: GenerationRequest) -> float:
        e = self.out_tokens.get(self._key(req))
        return e.value if e else req.max_tokens / 2

    def observe_output(self, req: GenerationRequest, completion_tokens: int) -> None:
        k = self._key(req)
        if k not in self.out_tokens:
            self.out_tokens[k] = EMA(completion_tokens, 0.1)
        else:
            self.out_tokens[k].update(completion_tokens)

    # -- tools ---------------------------------------------------------------------

    def tool_mean(self, tool: str) -> float:
        return self.tool_s[tool].value

    def snapshot(self) -> dict[str, Any]:
        return {
            "chars_per_token": self.chars_per_token.value,
            "decode_ms_per_tok": self.decode_ms.value,
            "prefill_ms_per_tok": self.prefill_ms.value,
            "out_tokens": {k: v.value for k, v in self.out_tokens.items()},
            "tool_s": {k: v.value for k, v in self.tool_s.items()},
        }
