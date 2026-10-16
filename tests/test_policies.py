"""Behavioural tests of the preemptive policies and KV retention on the fake backend."""
from __future__ import annotations

import asyncio

import pytest

from agentserve.backends.fake import FakeBackend
from agentserve.core import registry
from agentserve.core.events import EventLog
from agentserve.core.types import GenerationRequest
from agentserve.runner import run_point

from tests.test_layers import toy_config


def req(rid: str, task: str, max_tokens: int, chars: int = 40, step: int = 0) -> GenerationRequest:
    return GenerationRequest(rid, task, step, [{"role": "user", "content": f"{task}:" + "x" * chars}],
                             max_tokens=max_tokens)


class Recorder(EventLog):
    def __init__(self):
        super().__init__(None)
        self.log: list[tuple[str, dict]] = []

    def emit(self, kind, **f):
        super().emit(kind, **f)
        self.log.append((kind, f))

    def kinds(self, kind):
        return [f for k, f in self.log if k == kind]


def make(sched: dict, **backend_kw):
    backend = FakeBackend(**{"max_concurrency": 4, "completion_tokens": 64, "time_scale": 0.05, **backend_kw})
    ev = Recorder()
    s = registry.build("scheduler", sched, backend=backend, events=ev)
    return backend, ev, s


async def run_reqs(s, reqs, delays=None):
    await s.start()
    async def one(r, d):
        await asyncio.sleep(d)
        return await s.generate(r)
    res = await asyncio.gather(*(one(r, (delays or [0] * len(reqs))[i]) for i, r in enumerate(reqs)))
    await s.stop()
    return res


# --- chunked continuation ---------------------------------------------------------

async def test_chunked_output_identical_to_unchunked():
    _, _, s1 = make({"type": "fcfs", "max_concurrency": 1})
    _, _, s2 = make({"type": "rr", "max_concurrency": 1, "quantum": 7})
    (a,) = await run_reqs(s1, [req("r", "t", 64)])
    (b,) = await run_reqs(s2, [req("r", "t", 64)])
    assert a.ok and b.ok and a.text == b.text and a.completion_tokens == b.completion_tokens == 64
    assert a.n_chunks == 1 and b.n_chunks == 10


async def test_stop_sequence_across_chunks():
    script = ["Thought: x\nAction: search[abc]\nObservation: hallucinated"]
    _, _, s = make({"type": "rr", "max_concurrency": 1, "quantum": 3}, script=script, completion_tokens=8)
    r = req("r", "t", 64)
    r.stop = ["\nObservation"]
    (res,) = await run_reqs(s, [r])
    assert res.text.strip() == "Thought: x\nAction: search[abc]"


# --- preemptive orderings -----------------------------------------------------------

async def test_rr_interleaves_and_counts_preemptions():
    _, ev, s = make({"type": "rr", "max_concurrency": 1, "quantum": 16})
    res = await run_reqs(s, [req("a", "A", 64), req("b", "B", 64)])
    order = [f["task_id"] for f in ev.kinds("chunk_done")]
    assert order[:4] == ["A", "B", "A", "B"]
    assert all(r.preemptions >= 3 for r in res)


@pytest.mark.parametrize("policy", [
    {"type": "fcfs", "max_concurrency": 1},
    {"type": "srpt", "max_concurrency": 1, "quantum": 8},
    {"type": "mlfq", "max_concurrency": 1, "quantum": 8, "skip_join": False},
])
async def test_short_request_overtakes_long(policy):
    _, _, s = make(policy)
    s.est.observe_output(req("x", "x", 1), 64)  # prior: long outputs
    long_r, short_r = req("long", "L", 64), req("short", "S", 4)
    short_r.tags = {"mix": "short"}
    s.est.observe_output(short_r, 4)
    lr, sr = await run_reqs(s, [long_r, short_r], delays=[0, 0.02])
    if policy["type"] == "fcfs":
        assert sr.t_done > lr.t_done          # head-of-line blocking
    else:
        assert sr.t_done < lr.t_done          # preempted the long one
        assert lr.preemptions >= 1


async def test_las_prefers_task_with_less_service():
    _, ev, s = make({"type": "las", "max_concurrency": 1, "quantum": 16, "base_s": 0.001})
    s.task_service.update({"OLD": 10.0, "NEW": 0.0})
    s.task_arrival.update({"OLD": 0.0, "NEW": 1.0})  # OLD arrived first but used more service
    await s.start()
    s.K = 0  # hold dispatch until both are queued
    t1 = asyncio.create_task(s.generate(req("o", "OLD", 16)))
    t2 = asyncio.create_task(s.generate(req("n", "NEW", 16)))
    await asyncio.sleep(0.01)
    s.K = 1
    s.notify()
    await asyncio.gather(t1, t2)
    await s.stop()
    assert [f["task_id"] for f in ev.kinds("chunk_done")][0] == "NEW"


# --- memory pressure --------------------------------------------------------------------

async def test_overflow_cliff_vs_mem_aware():
    big = [req(f"r{i}", f"T{i}", 64, chars=1600) for i in range(4)]  # ~400 prompt tokens each
    _, _, fc = make({"type": "fcfs", "max_concurrency": 4}, kv_budget_tokens=1200)
    res = await run_reqs(fc, big)
    assert any(r.status == "ctx_overflow" for r in res)
    big = [req(f"r{i}", f"T{i}", 64, chars=1600) for i in range(4)]
    _, _, ma = make({"type": "mem_aware", "order": "fcfs", "retention": {"type": "lru"}}, kv_budget_tokens=1200)
    res = await run_reqs(ma, big)
    assert all(r.ok for r in res)
    assert ma.blocked_decisions > 0


# --- KV retention -----------------------------------------------------------------------

async def _agent_steps(s, tasks, steps=3, gap=0.0):
    """Each task issues `steps` sequential requests whose prompt extends the previous one."""
    async def agent(t):
        hist = f"{t}:" + "y" * 200
        out = []
        for k in range(steps):
            r = GenerationRequest(f"{t}:{k}", t, k, [{"role": "user", "content": hist}], max_tokens=16)
            res = await s.generate(r)
            out.append(res)
            hist += res.text + "obs" * 10
            await asyncio.sleep(gap)
        return out
    await s.start()
    out = await asyncio.gather(*(agent(t) for t in tasks))
    await s.stop()
    return out


async def test_pinning_gives_prefix_hits():
    _, _, s = make({"type": "fcfs", "max_concurrency": 2, "retention": {"type": "lru"}}, max_concurrency=4)
    out = await _agent_steps(s, ["A", "B"])
    later = [r for steps in out for r in steps[1:]]
    assert all(r.cached_tokens > 0 for r in later)


@pytest.mark.parametrize("mode", ["erase", "swap"])
async def test_eviction_modes(mode):
    # 3 agents share 2 slots -> evictions; swap should restore KV on return, erase should not
    _, ev, s = make({"type": "fcfs", "max_concurrency": 2, "retention": {"type": "lru", "mode": mode}},
                    max_concurrency=2)
    await _agent_steps(s, ["A", "B", "C"], gap=0.02)
    ev_kinds = [f["mode"] for f in ev.kinds("slot_evict")]
    assert mode in ev_kinds
    misses = sum(f.get("reuse_miss_tokens", 0) for f in ev.kinds("request_done"))
    if mode == "swap":
        assert ev.kinds("slot_restore")
    else:
        assert misses > 0


async def test_tool_aware_evicts_agent_in_long_tool_call():
    _, ev, s = make({"type": "fcfs", "max_concurrency": 1, "retention": {"type": "tool_aware"}}, max_concurrency=2)
    s.est.tool_s["slow"].update(10.0)
    s.est.tool_s["fast"].update(0.01)
    await s.start()
    await s.generate(req("a0", "A", 8))
    await s.generate(req("b0", "B", 8))
    ev.emit("tool_start", task_id="A", tool="slow", step=0)
    ev.emit("tool_start", task_id="B", tool="fast", step=0)
    await s.generate(req("c0", "C", 8))  # needs a slot: should evict A (long tool call)
    await s.stop()
    assert [f["owner"] for f in ev.kinds("slot_evict")] == ["A"]


# --- every policy end to end -----------------------------------------------------------

@pytest.mark.parametrize("sched", [
    {"type": "rr", "max_concurrency": 2, "quantum": 8},
    {"type": "srpt", "max_concurrency": 2, "quantum": 8},
    {"type": "mlfq", "max_concurrency": 2, "quantum": 8},
    {"type": "las", "max_concurrency": 2, "quantum": 8},
    {"type": "mem_aware", "order": "las", "retention": {"type": "lru", "mode": "swap"}},
    {"type": "rr", "max_concurrency": 2, "quantum": 8, "retention": {"type": "tool_aware"}},
])
async def test_policy_end_to_end(tmp_path, sched):
    cfg = toy_config(scheduler=sched)
    cfg["backend"]["completion_tokens"] = 40
    s = await run_point(cfg, tmp_path)
    assert s["n_finished"] == 10 and s["n_failed"] == 0 and s["em"] == 1.0


async def test_swap_restore_survives_full_pool():
    # tight pool: restoring an image needs room, which must be made before restore
    _, ev, s = make({"type": "fcfs", "max_concurrency": 2, "retention": {"type": "lru", "mode": "swap"}},
                    max_concurrency=3, kv_budget_tokens=420)
    out = await _agent_steps(s, ["A", "B", "C", "D"], steps=3, gap=0.02)
    assert all(r.ok for steps in out for r in steps)
    assert ev.kinds("slot_restore") and not ev.kinds("slot_op_error")
