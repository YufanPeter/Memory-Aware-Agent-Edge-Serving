"""Agent memory interface: decides what context is sent to the LLM at each step.

This is where per-step context growth (and thus KV usage) is determined, so it
is kept independent from orchestration. Orchestrators only append entries and
ask for `render()`.

Entries are role-tagged messages plus a `kind` label so policies can treat
them differently (e.g. drop old observations, keep thoughts).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class MemoryEntry:
    role: str      # system | user | assistant
    content: str
    kind: str      # system | query | thought_action | observation | other


class Memory(ABC):
    registry_name: str = "?"

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.entries: list[MemoryEntry] = []

    def reset(self, system: str, query: str) -> None:
        self.entries = [MemoryEntry("system", system, "system"), MemoryEntry("user", query, "query")]

    def add(self, role: str, content: str, kind: str) -> None:
        self.entries.append(MemoryEntry(role, content, kind))

    @abstractmethod
    def select(self) -> list[MemoryEntry]:
        """Entries to send at this step, in order."""

    def render(self) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        for e in self.select():
            # Merge consecutive same-role entries; chat templates expect alternation.
            if msgs and msgs[-1]["role"] == e.role:
                msgs[-1]["content"] += "\n" + e.content
            else:
                msgs.append({"role": e.role, "content": e.content})
        return msgs

    def stats(self) -> dict[str, Any]:
        sel = self.select()
        return {
            "entries_total": len(self.entries),
            "entries_sent": len(sel),
            "chars_total": sum(len(e.content) for e in self.entries),
            "chars_sent": sum(len(e.content) for e in sel),
        }
