"""Data contracts shared by all layers.

These dataclasses are the only things layers exchange. A layer may depend on
this module and on the abstract interface of the layer directly below it, but
never on a concrete implementation in another layer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


# --- L1/L2: generation -------------------------------------------------------

@dataclass
class GenerationRequest:
    request_id: str
    task_id: str
    step: int
    messages: list[dict[str, str]]
    max_tokens: int = 256
    temperature: float = 0.0
    seed: int = 0
    stop: list[str] = field(default_factory=list)
    # Optional hints a scheduler/backend may use (e.g. prefix-cache identity,
    # estimated prompt size). Implementations must work when they are absent.
    cache_hint: str | None = None
    est_prompt_tokens: int | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    # Set by the scheduler, not by agents:
    prefix: str = ""          # already-generated assistant text to continue from (chunked preemption)
    slot: int | None = None   # pin to a backend slot (None = backend chooses)


@dataclass
class GenerationResult:
    request_id: str
    text: str = ""
    status: str = "ok"  # ok | timeout | ctx_overflow | error
    error: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    # Wall-clock seconds measured by the client; server-side timings as reported.
    t_send: float = 0.0
    t_first_token: float | None = None
    t_done: float = 0.0
    prefill_ms: float | None = None
    decode_ms: float | None = None
    slot: int | None = None
    # Filled in by the scheduler.
    t_enqueue: float = 0.0
    t_dispatch: float = 0.0
    n_chunks: int = 1
    preemptions: int = 0          # times the request lost its turn at a chunk boundary
    preempt_wait: float = 0.0     # seconds spent paused after being preempted
    recompute_tokens: int = 0     # prompt tokens re-prefilled in continuation chunks (lost KV)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMClient(Protocol):
    """What agents see. Implemented by every Scheduler."""

    async def generate(self, req: GenerationRequest) -> GenerationResult: ...


# --- L3/L4: tasks ------------------------------------------------------------

@dataclass
class TaskInstance:
    """One dataset item, opaque to everything except its Task."""
    instance_id: str
    payload: Any


@dataclass
class TaskSpec:
    """A unit of work produced by the workload layer."""
    task_id: str
    task_name: str           # key into the experiment's `tasks:` section
    instance: TaskInstance
    arrival_time: float = 0.0  # seconds since run start (open-loop only)
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContext:
    """Task-provided starting point for an agent; agent-agnostic."""
    task_id: str
    instruction: str                  # task-specific guidance (goes into system prompt)
    query: str                        # the question / goal
    documents: list[str] = field(default_factory=list)  # pre-loaded context (may be empty)
    max_tokens: int = 256
    max_steps: int | None = None
    seed: int = 0
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepRecord:
    step: int
    request_id: str | None = None
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None
    llm_status: str | None = None
    tool_delay_s: float = 0.0
    memory_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    status: str = "ok"  # ok | max_steps | llm_error | timeout | error
    prediction: str | None = None
    steps: list[StepRecord] = field(default_factory=list)
    error: str | None = None

    @property
    def n_steps(self) -> int:
        return len(self.steps)
