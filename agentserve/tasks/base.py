"""L3a Task interface: *what* to solve (data, initial context, tools, gold).

A Task contains no agent loop logic; any orchestrator/memory can be paired
with it. Workload tags (e.g. `context: long`) are interpreted here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentserve.agents.tools.base import Tool
from agentserve.core.types import AgentContext, TaskInstance, TaskSpec


class Task(ABC):
    registry_name: str = "?"
    # Evaluators used when the experiment does not list any.
    default_evaluators: list[dict[str, Any]] = []

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def load(self) -> list[TaskInstance]:
        """All candidate instances, in a deterministic order."""

    @abstractmethod
    def context(self, spec: TaskSpec) -> AgentContext:
        """Initial agent context for this spec (honouring spec.tags)."""

    def tools(self, spec: TaskSpec) -> list[Tool]:
        """Fresh tool instances for one task execution (tools may be stateful)."""
        return []

    @abstractmethod
    def gold(self, instance: TaskInstance) -> Any:
        """Reference answer(s) handed to evaluators."""

    def describe(self, instance: TaskInstance) -> dict[str, Any]:
        """Small, JSON-able metadata recorded per task (e.g. question type)."""
        return {}
