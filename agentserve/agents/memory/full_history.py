"""Baseline memory: send the whole trajectory every step (context grows linearly,
but each step's prompt is a strict prefix-extension of the last — ideal for
prefix caching)."""
from __future__ import annotations

from agentserve.agents.memory.base import Memory, MemoryEntry
from agentserve.core.registry import register


@register("memory", "full_history")
class FullHistoryMemory(Memory):
    def select(self) -> list[MemoryEntry]:
        return list(self.entries)
