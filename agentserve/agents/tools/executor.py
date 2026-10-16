"""Uniform wrapper around tool calls: injected latency, timeout, events.

Keeps latency modelling and instrumentation independent of any concrete tool.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any

from agentserve.agents.tools.base import ToolRegistry
from agentserve.agents.tools.delay import DelayModel, NoDelay
from agentserve.core import registry
from agentserve.core.events import EventLog


class ToolExecutor:
    def __init__(
        self,
        tools: ToolRegistry,
        events: EventLog,
        rng: random.Random,
        task_id: str,
        delay: dict[str, Any] | None = None,
        per_tool_delay: dict[str, dict[str, Any]] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.tools = tools
        self.events = events
        self.rng = rng
        self.task_id = task_id
        self.timeout = timeout
        self.default_delay: DelayModel = registry.build("delay", delay) if delay else NoDelay()
        self.per_tool: dict[str, DelayModel] = {
            name: registry.build("delay", cfg) for name, cfg in (per_tool_delay or {}).items()
        }

    def describe(self) -> str:
        return self.tools.describe()

    def names(self) -> list[str]:
        return self.tools.names()

    async def execute(self, name: str, arg: str, step: int) -> tuple[str, float]:
        """Returns (observation, injected_delay_seconds)."""
        tool = self.tools.get(name)
        delay = (self.per_tool.get(name) or self.default_delay).sample(self.rng)
        self.events.emit("tool_start", task_id=self.task_id, step=step, tool=name, delay=delay)
        status = "ok"
        if tool is None:
            obs, status = f"Unknown tool '{name}'. Available: {', '.join(self.tools.names())}.", "unknown_tool"
        else:
            try:
                # Delay models the tool's external latency; the tool body itself is local and fast.
                await asyncio.sleep(delay)
                obs = await asyncio.wait_for(tool.call(arg), self.timeout)
            except asyncio.TimeoutError:
                obs, status = "Tool timed out.", "timeout"
            except Exception as e:  # tool bugs become observations, never crash the agent
                obs, status = f"Tool error: {e}", "error"
        self.events.emit("tool_end", task_id=self.task_id, step=step, tool=name, status=status, obs_len=len(obs))
        return obs, delay
