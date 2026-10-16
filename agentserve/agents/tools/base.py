"""Tool interface. Tasks provide concrete Tool instances; agents only see this API."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Tool(ABC):
    name: str = "?"
    description: str = ""  # one line, shown to the model, e.g. "search[entity]: ..."

    @abstractmethod
    async def call(self, arg: str) -> str:
        """Execute the tool and return the observation text."""


class ToolRegistry:
    """Per-task set of tools, addressed by name."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        return "\n".join(f"- {t.description}" for t in self._tools.values())
