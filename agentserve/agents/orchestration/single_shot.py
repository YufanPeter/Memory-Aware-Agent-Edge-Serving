"""Single LLM call, no tools: a one-step control for multi-step agents."""
from __future__ import annotations

from typing import Any

from agentserve.agents.memory.base import Memory
from agentserve.agents.orchestration.base import Orchestrator
from agentserve.agents.orchestration.parsers import parse_answer
from agentserve.agents.tools.executor import ToolExecutor
from agentserve.core.events import EventLog
from agentserve.core.registry import register
from agentserve.core.types import AgentContext, LLMClient, StepRecord, TaskResult


@register("orchestrator", "single_shot")
class SingleShotOrchestrator(Orchestrator):
    def __init__(self, reasoning: bool = False, **params: Any) -> None:
        super().__init__(reasoning=reasoning, **params)
        self.reasoning = reasoning

    async def run(self, ctx: AgentContext, llm: LLMClient, tools: ToolExecutor, memory: Memory, events: EventLog) -> TaskResult:
        fmt = ("Think step by step, then give the final line as 'Answer: <answer>'."
               if self.reasoning else "Reply with only 'Answer: <answer>'.")
        docs = "\n\n".join(ctx.documents)
        query = (f"Context:\n{docs}\n\n" if docs else "") + f"Question: {ctx.query}"
        memory.reset(f"{ctx.instruction}\n{fmt}".strip(), query)
        rec = StepRecord(step=0, action="answer", memory_stats=memory.stats())
        req = self.make_request(ctx, 0, memory, ctx.max_tokens, [])
        res = await llm.generate(req)
        rec.request_id, rec.llm_status = req.request_id, res.status
        result = TaskResult(task_id=ctx.task_id, steps=[rec])
        if not res.ok:
            result.status, result.error = "llm_error", f"{res.status}: {res.error}"
        else:
            result.prediction = parse_answer(res.text)
        return result
