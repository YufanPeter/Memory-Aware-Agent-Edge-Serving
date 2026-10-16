"""Task-quality evaluator interface. Consumes only (prediction, gold, TaskResult),
so external evaluation frameworks / LLM judges can be wrapped as new Evaluators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentserve.core.types import TaskResult


class Evaluator(ABC):
    registry_name: str = "?"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def score(self, result: TaskResult, gold: Any) -> dict[str, float]:
        """Return metric_name -> value; names should be globally unique (prefix them)."""
