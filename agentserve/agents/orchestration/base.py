"""Orchestrator interface: the control loop of an agent (when to call the LLM,
when to call tools, when to stop). Independent of task, tools and memory policy."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentserve.agents.memory.base import Memory
from agentserve.agents.tools.executor import ToolExecutor
from agentserve.core.events import EventLog
from agentserve.core.types import AgentContext, GenerationRequest, LLMClient, TaskResult


class Orchestrator(ABC):
    registry_name: str = "?"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    async def run(
        self, ctx: AgentContext, llm: LLMClient, tools: ToolExecutor, memory: Memory, events: EventLog,
    ) -> TaskResult: ...

    @staticmethod
    def make_request(ctx: AgentContext, step: int, memory: Memory, max_tokens: int, stop: list[str]) -> GenerationRequest:
        return GenerationRequest(
            request_id=f"{ctx.task_id}:{step}",
            task_id=ctx.task_id,
            step=step,
            messages=memory.render(),
            max_tokens=max_tokens,
            seed=ctx.seed,
            stop=stop,
            cache_hint=ctx.task_id,
            tags=ctx.tags,
        )
