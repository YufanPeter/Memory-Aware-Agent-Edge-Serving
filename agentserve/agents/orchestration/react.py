"""ReAct (Thought / Action / Observation) loop."""
from __future__ import annotations

from typing import Any

from agentserve.agents.memory.base import Memory
from agentserve.agents.orchestration.base import Orchestrator
from agentserve.agents.orchestration.parsers import parse_answer, parse_react
from agentserve.agents.tools.executor import ToolExecutor
from agentserve.core.events import EventLog
from agentserve.core.registry import register
from agentserve.core.types import AgentContext, LLMClient, StepRecord, TaskResult

SYSTEM_TEMPLATE = """Solve the task by interleaving Thought, Action and Observation steps.
Each reply is exactly one Thought line followed by exactly one Action line, then stop and wait for the Observation.

Available actions:
{tools}
- finish: give the final answer and end the task (a short phrase or yes/no). Example: Action: finish[1844]

{instruction}"""


@register("orchestrator", "react")
class ReActOrchestrator(Orchestrator):
    def __init__(self, max_steps: int = 7, force_finish: bool = True, max_parse_errors: int = 2, **params: Any) -> None:
        super().__init__(max_steps=max_steps, force_finish=force_finish, max_parse_errors=max_parse_errors, **params)
        self.max_steps = max_steps
        self.force_finish = force_finish
        self.max_parse_errors = max_parse_errors

    def build_query(self, ctx: AgentContext) -> str:
        docs = "\n\n".join(ctx.documents)
        return (f"Context:\n{docs}\n\n" if docs else "") + f"Question: {ctx.query}"

    async def run(self, ctx: AgentContext, llm: LLMClient, tools: ToolExecutor, memory: Memory, events: EventLog) -> TaskResult:
        result = TaskResult(task_id=ctx.task_id)
        max_steps = ctx.max_steps or self.max_steps
        system = SYSTEM_TEMPLATE.format(tools=tools.describe(), instruction=ctx.instruction).strip()
        memory.reset(system, self.build_query(ctx))
        parse_errors = 0
        stop = ["\nObservation", "Observation:"]

        for step in range(max_steps):
            rec = StepRecord(step=step)
            result.steps.append(rec)
            req = self.make_request(ctx, step, memory, ctx.max_tokens, stop)
            rec.memory_stats = memory.stats()
            res = await llm.generate(req)
            rec.request_id, rec.llm_status = req.request_id, res.status
            if not res.ok:
                result.status, result.error = "llm_error", f"{res.status}: {res.error}"
                return result
            text = res.text.strip()
            memory.add("assistant", text, "thought_action")
            parsed = parse_react(text)
            if parsed is None:
                parse_errors += 1
                rec.action = "parse_error"
                obs = "Invalid format. Reply with 'Thought: ...' then 'Action: tool[argument]'."
                if parse_errors > self.max_parse_errors:
                    break
            else:
                rec.action, rec.action_input = parsed
                if rec.action == "finish":
                    result.prediction = rec.action_input
                    return result
                obs, rec.tool_delay_s = await tools.execute(rec.action, rec.action_input, step)
            rec.observation = obs
            memory.add("user", f"Observation: {obs}", "observation")

        # Ran out of steps (or too many parse errors).
        result.status = "max_steps"
        if self.force_finish:
            step = len(result.steps)
            memory.add("user", "You must answer now. Reply with: Action: finish[answer]", "other")
            req = self.make_request(ctx, step, memory, min(ctx.max_tokens, 64), stop)
            rec = StepRecord(step=step, action="forced_finish", memory_stats=memory.stats())
            result.steps.append(rec)
            res = await llm.generate(req)
            rec.request_id, rec.llm_status = req.request_id, res.status
            if res.ok:
                result.prediction = parse_answer(res.text)
        return result
