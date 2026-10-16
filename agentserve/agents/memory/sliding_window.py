"""Keep system + query + the last `window` (thought/action, observation) turns.
Bounds context size, at the cost of breaking prefix reuse once the window slides."""
from __future__ import annotations

from typing import Any

from agentserve.agents.memory.base import Memory, MemoryEntry
from agentserve.core.registry import register


@register("memory", "sliding_window")
class SlidingWindowMemory(Memory):
    def __init__(self, window: int = 3, **params: Any) -> None:
        super().__init__(window=window, **params)
        self.window = window

    def select(self) -> list[MemoryEntry]:
        head = [e for e in self.entries if e.kind in ("system", "query")]
        turns = [e for e in self.entries if e.kind not in ("system", "query")]
        return head + turns[-2 * self.window:] if self.window > 0 else head
