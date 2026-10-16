"""Layer contract tests. Everything runs on the fake backend and an in-test task,
which also demonstrates registering plugins from outside the package."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agentserve.agents.memory.base import Memory
from agentserve.agents.orchestration.parsers import parse_answer, parse_react
from agentserve.agents.tools.base import Tool
from agentserve.backends.fake import FakeBackend
from agentserve.core import registry
from agentserve.core.config import expand_sweep, load_config
from agentserve.core.events import EventLog
from agentserve.core.types import AgentContext, GenerationRequest, TaskInstance, TaskSpec
from agentserve.evaluators.qa_em_f1 import QAEmF1
from agentserve.core.types import TaskResult
from agentserve.runner import run_point
from agentserve.schedulers.fcfs import FCFSScheduler
from agentserve.tasks.base import Task


# --- an external task plugin --------------------------------------------------

class EchoTool(Tool):
    name = "search"
    description = "search: echo. Example: Action: search[x]"

    async def call(self, arg: str) -> str:
        return f"echo {arg}"


@registry.register("task", "toy")
class ToyTask(Task):
    default_evaluators = [{"type": "qa_em_f1"}]

    def __init__(self, n: int = 20, **kw):
        super().__init__(n=n, **kw)
        self.n = n

    def load(self):
        return [TaskInstance(f"q{i}", {"q": f"question {i}", "a": "yes"}) for i in range(self.n)]

    def context(self, spec: TaskSpec) -> AgentContext:
        return AgentContext(task_id=spec.task_id, instruction="", query=spec.instance.payload["q"],
                            max_tokens=int(spec.tags.get("max_tokens", 32)))

    def tools(self, spec):
        return [EchoTool()]

    def gold(self, inst):
        return inst.payload["a"]


def toy_config(**over) -> dict:
    cfg = {
        "model": {"name": "fake"},
        "backend": {"type": "fake", "max_concurrency": 8, "time_scale": 0.02},
        "scheduler": {"type": "fcfs", "max_concurrency": 2},
        "tasks": {"toy": {"type": "toy"}},
        "agent": {"orchestration": {"type": "react", "max_steps": 4},
                  "memory": {"type": "full_history"},
                  "tools": {"delay": {"type": "fixed", "seconds": 0.01}}},
        "workload": {"n_tasks": 10, "arrival": {"type": "burst"},
                     "mix": [{"name": "a", "task": "toy", "weight": 1, "tags": {"max_tokens": 16}},
                             {"name": "b", "task": "toy", "weight": 1, "tags": {"max_tokens": 64}}]},
        "monitors": [{"type": "scheduler_state", "interval": 0.01}],
        "run": {"warmup_requests": 0},
        "seed": 0,
    }
    cfg.update(over)
    return cfg


def read_events(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)]


# --- end to end ---------------------------------------------------------------

async def test_end_to_end_fake(tmp_path):
    s = await run_point(toy_config(), tmp_path)
    assert s["n_tasks"] == 10 and s["n_finished"] == 10 and s["n_failed"] == 0
    assert s["em"] == 1.0  # fake script finishes with "yes"
    assert set(s["by_mix"]) == {"a", "b"}
    for f in ("events.jsonl", "tasks.jsonl", "meta.json", "summary.json", "config.yaml"):
        assert (tmp_path / f).exists()


async def test_fcfs_respects_concurrency_and_order(tmp_path):
    await run_point(toy_config(), tmp_path)
    ev = read_events(tmp_path / "events.jsonl")
    # in-flight (including the request being dispatched) never exceeds K=2, and reaches it
    assert max(e["inflight"] for e in ev if e["kind"] == "request_dispatch") == 2
    # dispatch order == enqueue order
    enq = [e["request_id"] for e in ev if e["kind"] == "request_enqueue"]
    dis = [e["request_id"] for e in ev if e["kind"] == "request_dispatch"]
    assert dis == enq


async def test_task_admission_cap(tmp_path):
    cfg = toy_config(scheduler={"type": "fcfs", "max_concurrency": 4, "max_active_tasks": 3})
    await run_point(cfg, tmp_path)
    ev = read_events(tmp_path / "events.jsonl")
    active = 0
    for e in ev:
        if e["kind"] == "task_start":
            active += 1
            assert active <= 3
        elif e["kind"] == "task_done":
            active -= 1


@pytest.mark.parametrize("orch,memory", [
    ({"type": "react", "max_steps": 3}, {"type": "sliding_window", "window": 1}),
    ({"type": "single_shot"}, {"type": "full_history"}),
])
async def test_agent_dimensions_swap(tmp_path, orch, memory):
    cfg = toy_config()
    cfg["agent"] = {"orchestration": orch, "memory": memory, "tools": {}}
    s = await run_point(cfg, tmp_path)
    assert s["n_finished"] == 10


# --- units --------------------------------------------------------------------

async def test_fake_backend_overflow():
    b = FakeBackend(kv_budget_tokens=10)
    r = await b.generate(GenerationRequest("r", "t", 0, [{"role": "user", "content": "x" * 400}]))
    assert r.status == "ctx_overflow"


async def test_scheduler_queues_beyond_k():
    b = FakeBackend(max_concurrency=8, time_scale=0.05)
    ev = EventLog(None)
    s = FCFSScheduler(b, ev, max_concurrency=1)
    await s.start()
    reqs = [GenerationRequest(f"r{i}", f"t{i}", 0, [{"role": "user", "content": "hi"}]) for i in range(3)]
    res = await asyncio.gather(*(s.generate(r) for r in reqs))
    await s.stop()
    starts = sorted(r.t_dispatch for r in res)
    dones = sorted(r.t_done for r in res)
    assert starts[1] >= dones[0] - 1e-6 and starts[2] >= dones[1] - 1e-6


def test_parsers():
    assert parse_react("Thought: x\nAction: search[Ed Wood]") == ("search", "Ed Wood")
    assert parse_react("Thought: x\nAction: Finish[yes]") == ("finish", "yes")
    assert parse_react("Final Answer: Paris") == ("finish", "Paris")
    assert parse_react("no idea") is None
    assert parse_answer("blah\nAnswer: 1844") == "1844"


def test_em_f1():
    ev = QAEmF1()
    assert ev.score(TaskResult("t", prediction="The Beatles"), "beatles")["em"] == 1.0
    assert ev.score(TaskResult("t", prediction="yes"), "no")["f1"] == 0.0
    assert 0 < ev.score(TaskResult("t", prediction="Arthur Magazine"), "Arthur's Magazine")["f1"] < 1


def test_memory_render_merges_roles():
    m = registry.build("memory", {"type": "full_history"})
    m.reset("sys", "q")
    m.add("user", "Observation: a", "observation")
    assert [x["role"] for x in m.render()] == ["system", "user"]


def test_config_refs_and_sweep():
    cfg = load_config(Path(__file__).parents[1] / "configs/experiments/smoke_fake.yaml")
    assert cfg["backend"]["type"] == "fake" and "ctx_size" not in cfg["backend"]
    assert cfg["workload"]["arrival"] == {"type": "burst"}
    pts = list(expand_sweep(cfg))
    assert [p["backend"]["max_concurrency"] for _, p in pts] == [1, 4]
    assert [p["scheduler"]["max_concurrency"] for _, p in pts] == [1, 4]


def test_all_experiment_point_ids_unique():
    root = Path(__file__).parents[1] / "configs/experiments"
    for f in root.glob("*.yaml"):
        if f.name.startswith("_"):
            continue
        ids = [p for p, _ in expand_sweep(load_config(f))]
        assert len(ids) == len(set(ids)), f.name
