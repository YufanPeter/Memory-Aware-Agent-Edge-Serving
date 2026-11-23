"""KV manager (edgeserve/kvm.py) against MockLLM's shared pool."""
import asyncio

from edgeserve.agent import Agent
from edgeserve.events import EventLog
from edgeserve.kvm import KVManager
from edgeserve.llm import MockLLM
from edgeserve.policies import get_policy
from edgeserve.profiles import Profile

DELAY = {"type": "fixed", "seconds": 0.05}


def two_searches(prompt: str) -> str:
    """Search twice, then finish (each completion is ~15 mock tokens)."""
    if prompt.endswith("finish["):
        return "x]"
    n = prompt.rsplit("Question: ", 1)[-1].count("\nObservation ")
    if n == 0:
        return " First Scott Derrickson.\nAction 1: search[Scott Derrickson]"
    if n == 1:
        return " Now Ed Wood.\nAction 2: search[Ed Wood]"
    return " Both are American.\nAction 3: finish[yes]"


async def run_agents(corpus, item, n_agents: int, pool: int, policy: str | None):
    """n_agents run concurrently, worker k on slot k, like the runner's fixed-K mode."""
    llm = MockLLM(two_searches, decode_s_per_token=2e-4, pool_tokens=pool)
    events = EventLog()
    kvm = KVManager(llm, events, get_policy(policy), pool, n_agents) if policy else None
    agent = Agent(llm, corpus, events, kvm=kvm)
    profile = Profile(name="react_short", few_shot=2, delay=DELAY)

    async def one(k: int):
        if kvm:
            await kvm.admit(f"t{k}", profile.name, k)
        try:
            return await agent.run(f"t{k}", item, profile, slot=k)
        finally:
            if kvm:
                await kvm.release(k)

    traces = await asyncio.gather(*(one(k) for k in range(n_agents)))
    return traces, llm, kvm


async def footprint(corpus, item) -> int:
    """Tokens one agent holds at its peak, measured alone."""
    traces, llm, _ = await run_agents(corpus, item, 1, 0, None)
    return max(s.llm["prompt_n"] + s.llm["cache_n"] + s.llm["predicted_n"] for s in traces[0].steps)


async def test_fixed_k_overflows_where_the_gate_does_not(corpus, item):
    peak = await footprint(corpus, item)
    pool = int(peak * 2.5)                      # room for two agents, not four
    traces, llm, _ = await run_agents(corpus, item, 4, pool, None)
    assert llm.overflows > 0 and any(t.status == "ctx_overflow" for t in traces)

    traces, llm, kvm = await run_agents(corpus, item, 4, pool, "ttl_fixed_k")
    assert llm.overflows == 0
    assert all(t.status == "ok" and t.prediction == "yes" for t in traces)
    assert kvm.stats.peak_resident <= pool
    assert sum(kvm.stats.give_ups.values()) > 0          # it had to take KV from idle agents


async def test_admission_is_fifo_and_bounded(corpus, item):
    peak = await footprint(corpus, item)
    pool = int(peak * 2.5)
    llm = MockLLM(two_searches, decode_s_per_token=2e-4, pool_tokens=pool)
    events = EventLog()
    pol = get_policy("admit_only")
    kvm = KVManager(llm, events, pol, pool, 6)
    kvm_peak = {"react_short": peak + 50}
    import edgeserve.kvm as kvm_mod
    old = dict(kvm_mod.PEAK_P95)
    kvm_mod.PEAK_P95.update(kvm_peak)
    agent = Agent(llm, corpus, events, kvm=kvm)
    profile = Profile(name="react_short", few_shot=2, delay=DELAY)
    order: list[str] = []
    running = {"now": 0, "max": 0}

    async def one(k: int):
        await kvm.admit(f"t{k}", profile.name, k)
        order.append(f"t{k}")
        running["now"] += 1
        running["max"] = max(running["max"], running["now"])
        try:
            return await agent.run(f"t{k}", item, profile, slot=k)
        finally:
            running["now"] -= 1
            await kvm.release(k)

    try:
        traces = await asyncio.gather(*(one(k) for k in range(6)))
    finally:
        kvm_mod.PEAK_P95.clear()
        kvm_mod.PEAK_P95.update(old)
    assert order == [f"t{k}" for k in range(6)]
    assert running["max"] == 2                            # 2 x (peak + 50) fits in 2.5 x peak, 3 do not
    assert llm.overflows == 0 and all(t.status == "ok" for t in traces)


async def test_offload_restores_the_cache(corpus, item):
    llm = MockLLM(two_searches, decode_s_per_token=2e-4, pool_tokens=100_000)
    events = EventLog()
    kvm = KVManager(llm, events, get_policy("always_offload"), 100_000, 1)
    agent = Agent(llm, corpus, events, kvm=kvm)
    await kvm.admit("t0", "react_short", 0)
    trace = await agent.run("t0", item, Profile(name="react_short", few_shot=2, delay=DELAY), slot=0)
    await kvm.release(0)
    assert trace.status == "ok"
    assert kvm.stats.slot_ops.get("save") == 2 and kvm.stats.slot_ops.get("restore") == 2
    # Each step after a tool call reuses the restored KV instead of re-prefilling it.
    for prev, step in zip(trace.steps, trace.steps[1:]):
        assert step.llm["cache_n"] >= prev.llm["prompt_n"] + prev.llm["cache_n"]


async def test_admission_reuses_a_stale_slot_of_the_same_family(corpus, item):
    llm = MockLLM(two_searches, decode_s_per_token=2e-4, pool_tokens=100_000)
    events = EventLog()
    kvm = KVManager(llm, events, get_policy("admit_only"), 100_000, 8)
    agent = Agent(llm, corpus, events, kvm=kvm)
    profile = Profile(name="react_short", few_shot=2, delay=DELAY)
    s1 = await kvm.admit("a", "react_short")
    await agent.run("a", item, profile, slot=s1)
    await kvm.release(s1)
    s_one = await kvm.admit("b", "oneshot_long")          # another family: does not take a's slot
    s2 = await kvm.admit("c", "react_short")              # same family: takes a's stale slot
    assert s2 == s1 and s_one != s1
    trace = await agent.run("c", item, profile, slot=s2)
    assert trace.steps[0].llm["cache_n"] > 0              # the shared prefix was reused


async def test_fork_prefix_shares_the_system_prompt(corpus, item):
    llm = MockLLM(two_searches, decode_s_per_token=2e-4, pool_tokens=100_000)
    events = EventLog()
    kvm = KVManager(llm, events, get_policy("base_fork"), 100_000, 8)
    agent = Agent(llm, corpus, events, kvm=kvm)
    profile = Profile(name="react_short", few_shot=2, delay=DELAY)
    slots = [await kvm.admit(t, "react_short") for t in ("a", "b")]
    assert not set(slots) & {5, 6, 7}                     # the top 3 slots are reserved for golden prefixes
    traces = await asyncio.gather(*(agent.run(t, item, profile, slot=s) for t, s in zip("ab", slots)))
    prefix_tokens = len(traces[0].segments[0][1]) // 4
    for t in traces:
        assert t.status == "ok"
        assert t.steps[0].llm["cache_n"] >= prefix_tokens - 1   # first request reuses the forked prefix
    assert kvm.stats.forks == 2 and len(kvm.golden) == 1
    # The manager's accounting matches the pool (both count the shared prefix once).
    assert kvm.resident() == llm.resident_tokens()


async def test_fork_keeps_a_tight_pool_safe(corpus, item):
    peak = await footprint(corpus, item)
    pool = int(peak * 2.5)
    llm = MockLLM(two_searches, decode_s_per_token=2e-4, pool_tokens=pool)
    events = EventLog()
    kvm = KVManager(llm, events, get_policy("base_fork", overcommit=3.0), pool, 9)
    agent = Agent(llm, corpus, events, kvm=kvm)
    profile = Profile(name="react_short", few_shot=2, delay=DELAY)

    async def one(t):
        s = await kvm.admit(t, "react_short")
        try:
            return await agent.run(t, item, profile, slot=s)
        finally:
            await kvm.release(s)

    traces = await asyncio.gather(*(one(f"t{k}") for k in range(6)))
    assert llm.overflows == 0 and not llm.purges
    assert all(t.status == "ok" for t in traces)


async def test_a_request_that_can_never_fit_is_refused_without_blocking_others(corpus, item):
    """An agent whose context outgrows the pool fails alone (ctx_overflow); it must not sit at the
    head of the gate queue and block the agent behind it (the base_r1.0 timeout of 2026-11-23)."""
    peak = await footprint(corpus, item)
    pool = int(peak * 0.9)                         # the agent's last request cannot fit at all
    llm = MockLLM(two_searches, decode_s_per_token=2e-4, pool_tokens=pool)
    events = EventLog()
    kvm = KVManager(llm, events, get_policy("ttl_fixed_k"), pool, 2)
    agent = Agent(llm, corpus, events, kvm=kvm)
    profile = Profile(name="react_short", few_shot=2, delay=DELAY)

    async def one(k: int):
        await kvm.admit(f"t{k}", profile.name, k)
        try:
            return await asyncio.wait_for(agent.run(f"t{k}", item, profile, slot=k), timeout=10)
        finally:
            await kvm.release(k)

    traces = await asyncio.gather(*(one(k) for k in range(2)))
    assert all(t.status == "ctx_overflow" for t in traces)     # each fails on its own, nothing hangs
    assert kvm.stats.refused == 2 and not kvm.waiters
    assert llm.overflows == 0                                    # the server never saw an oversize request


async def test_golden_slots_count_as_unfreeable_room(corpus, item):
    """The 2026-11-23 deadlock: the request fits the pool on its own, but not next to the golden
    slots (never evicted). It used to wait at the head of the gate forever."""
    from edgeserve.prompt import CHATML, react_system
    peak = await footprint(corpus, item)                        # react agent alone, no fork
    explain_prefix = CHATML["system"].format(content=react_system(2, True)) + CHATML["user_open"]
    p_explain = MockLLM.tokens(explain_prefix)
    pool = int((peak + p_explain // 2) / 0.97)                  # fits alone; not with the explain golden too
    llm = MockLLM(two_searches, decode_s_per_token=2e-4, pool_tokens=pool)
    events = EventLog()
    kvm = KVManager(llm, events, get_policy("base_fork", overcommit=100.0), pool, 8)
    agent = Agent(llm, corpus, events, kvm=kvm)

    async def one(task: str, profile: Profile):
        s = await kvm.admit(task, profile.name)
        try:
            return await asyncio.wait_for(agent.run(task, item, profile, slot=s), timeout=10)
        finally:
            await kvm.release(s)

    explain = await one("e", Profile(name="react_explain", few_shot=2, explain_tokens=4, delay=DELAY))
    assert explain.status == "ok" and len(kvm.golden) == 1      # its golden slot stays
    react = await one("r", Profile(name="react_short", few_shot=2, delay=DELAY))
    assert react.status == "ctx_overflow" and kvm.stats.refused >= 1 and not kvm.waiters
