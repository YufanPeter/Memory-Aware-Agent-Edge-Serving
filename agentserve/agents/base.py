"""Agent = Orchestrator x Tools x Memory, each built independently from config:

    agent:
      orchestration: {type: react, max_steps: 7}
      memory:        {type: full_history}
      tools:         {delay: {type: exponential, mean: 1.0}, per_tool_delay: {}, timeout: 30}
"""
from __future__ import annotations

import random
from typing import Any

from agentserve.agents.tools.base import Tool, ToolRegistry
from agentserve.agents.tools.executor import ToolExecutor
from agentserve.core import registry
from agentserve.core.events import EventLog
from agentserve.core.types import AgentContext, LLMClient, TaskResult


class Agent:
    def __init__(self, orchestration: dict[str, Any], memory: dict[str, Any] | None = None,
                 tools: dict[str, Any] | None = None) -> None:
        self.orchestrator = registry.build("orchestrator", orchestration)
        self.memory_cfg = memory or {"type": "full_history"}
        self.tools_cfg = tools or {}

    async def run(self, ctx: AgentContext, tools: list[Tool], llm: LLMClient, events: EventLog) -> TaskResult:
        memory = registry.build("memory", self.memory_cfg)  # fresh per task
        rng = random.Random(f"{ctx.seed}:{ctx.task_id}")     # order-independent tool delays
        executor = ToolExecutor(ToolRegistry(tools), events, rng, ctx.task_id, **self.tools_cfg)
        return await self.orchestrator.run(ctx, llm, executor, memory, events)
