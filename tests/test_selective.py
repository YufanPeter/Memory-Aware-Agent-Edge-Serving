"""Selective KV: sentence selection (edgeserve/selection.py) and the agent loop with
edgeserve/kvtrim.SelectiveKV against MockLLM."""
from edgeserve.agent import Agent
from edgeserve.events import EventLog
from edgeserve.kvm import KVManager
from edgeserve.kvtrim import TRIMMED, SelectiveKV
from edgeserve.llm import MockLLM
from edgeserve.policies import get_policy
from edgeserve.profiles import Profile
from edgeserve.prompt import Prompt
from edgeserve.selection import select_drop

DELAY = {"type": "fixed", "seconds": 0.01}


def three_searches(prompt: str) -> str:
    if prompt.endswith("finish["):
        return "x]"
    n = prompt.rsplit("Question: ", 1)[-1].count("\nObservation ")
    actions = ["search[Scott Derrickson]", "search[Ed Wood]", "lookup[American]"]
    if n < 3:
        return f" Step {n + 1}.\nAction {n + 1}: {actions[n]}"
    return " Both are American.\nAction 4: finish[yes]"


async def run(corpus, item, mechanism: str | None):
    llm = MockLLM(three_searches)
    sel = SelectiveKV(llm, mechanism, scorer="bm25", keep=0.3, keep_last=1) if mechanism else None
    agent = Agent(llm, corpus, EventLog(), selective=sel)
    trace = await agent.run("t0", item, Profile(name="react_short", few_shot=2, delay=DELAY), slot=0)
    return trace, llm


async def test_kv_trim_deletes_old_sentences_and_keeps_the_cache(corpus, item):
    base, _ = await run(corpus, item, None)
    trace, llm = await run(corpus, item, "kv")
    assert trace.status == "ok" and trace.prediction == "yes"
    trims = [s.trim for s in trace.steps if s.trim]
    # tool call 3 is the first with an observation older than the latest one (keep_last=1)
    assert len(trims) == 1 and trace.steps[2].trim is not None
    assert trims[0]["ok"] and trims[0]["tokens"] > 0 and trims[0]["sentences"] == 2
    text = "".join(t for _, t in trace.segments)
    assert len(text) < len("".join(t for _, t in base.segments))
    assert sum(k == TRIMMED for k, _ in trace.segments) == 1     # the observation is marked as processed
    # The request after the trim (step 4) reuses everything the slot still holds: only the
    # new observation and its framing are prefilled.
    new_part = llm.calls[3]["prompt"].rsplit("\nObservation 3: ", 1)[1]
    assert trace.steps[3].llm["prompt_n"] <= (len("\nObservation 3: ") + len(new_part)) // 4 + 1


async def test_text_trim_forces_a_re_prefill(corpus, item):
    kv, _ = await run(corpus, item, "kv")
    text, _ = await run(corpus, item, "text")
    assert text.status == "ok" and text.prediction == "yes"
    # same sentences chosen, but without the KV operation the prompt diverges from the slot's
    # cache at the first removed sentence, so the next request prefills more
    assert [s.trim["sentences"] for s in text.steps if s.trim] == [s.trim["sentences"] for s in kv.steps if s.trim]
    assert text.steps[3].llm["prompt_n"] > kv.steps[3].llm["prompt_n"]
    assert all(s.trim["tokens"] == 0 for s in text.steps if s.trim)


async def run_with_trigger(corpus, item, pressure: float):
    """One agent under a KV manager whose pressure threshold is `pressure` (0: always under pressure,
    1 with a large pool: never), selective KV triggered by pressure, trimming right after use."""
    llm = MockLLM(three_searches)
    events = EventLog()
    kvm = KVManager(llm, events, get_policy("ttl_fixed_k", pressure=pressure), 100_000, 1)
    sel = SelectiveKV(llm, "kv", scorer="bm25", keep=0.3, keep_last=0, trigger="pressure")
    agent = Agent(llm, corpus, events, kvm=kvm, selective=sel)
    await kvm.admit("t0", "react_short", 0)
    trace = await agent.run("t0", item, Profile(name="react_short", few_shot=2, delay=DELAY), slot=0)
    await kvm.release(0)
    return trace


async def test_pressure_trigger_waits_while_memory_is_free(corpus, item):
    trace = await run_with_trigger(corpus, item, pressure=1.0)
    assert trace.status == "ok"
    assert not any(s.trim for s in trace.steps)
    assert sum(s.trim_deferred for s in trace.steps) == 3          # every tool call postponed trimming
    assert not any(k == TRIMMED for k, _ in trace.segments)        # nothing marked: still eligible later


async def test_pressure_trigger_trims_under_pressure(corpus, item):
    trace = await run_with_trigger(corpus, item, pressure=0.0)
    trims = [s.trim for s in trace.steps if s.trim]
    assert trace.status == "ok" and trims and all(t["ok"] for t in trims)
    assert sum(t["tokens"] for t in trims) > 0 and not any(s.trim_deferred for s in trace.steps)


# --- sentence selection ----------------------------------------------------------------------

Q = "Were Scott Derrickson and Ed Wood of the same nationality?"


def react_prompt() -> Prompt:
    p = Prompt()
    p.add("shared_prefix", "SYSTEM Question: ")
    p.add("question", Q)
    p.add("control", "<|im_end|><|im_start|>assistant\nThought 1:")
    p.add("thought_action", " Search Scott Derrickson.\nAction 1: search[Scott Derrickson]")
    p.add("control", "\nObservation 1: ")
    p.extend([("obs_meta", "[Scott Derrickson] "), ("obs_sf", "Scott Derrickson is an American director."),
              ("obs_other", " He lives in Los Angeles."), ("obs_other", " He directs horror films.")])
    p.add("control", "\nThought 2:")
    p.add("thought_action", " He is American. Now Ed Wood.\nAction 2: search[Ed Wood]")
    p.add("control", "\nObservation 2: ")
    p.extend([("obs_meta", "[Ed Wood] "), ("obs_sf", "Ed Wood was an American filmmaker."),
              ("obs_other", " He was born in Poughkeepsie.")])
    p.add("control", "\nThought 3:")
    return p


def texts(p: Prompt, idx: set[int]) -> set[str]:
    return {p.segments[i][1] for i in idx}


def test_oracle_keeps_facts_and_never_touches_the_latest_observation():
    p = react_prompt()
    drop, obs = select_drop(p, Q, "oracle")
    assert len(obs) == 1                                   # only observation 1 (keep_last=1)
    assert texts(p, drop) == {" He lives in Los Angeles.", " He directs horror films."}
    drop, obs = select_drop(p, Q, "oracle", keep_last=0)
    assert len(obs) == 2 and " He was born in Poughkeepsie." in texts(p, drop)


def test_bm25_prefers_relevant_sentences():
    p = react_prompt()
    drop, _ = select_drop(p, Q, "bm25", keep=0.3)          # keeps 1 of 3
    assert texts(p, drop) == {" He lives in Los Angeles.", " He directs horror films."}


def test_marked_observations_are_not_selected_again():
    p = react_prompt()
    _, obs = select_drop(p, Q, "bm25", keep=0.3)
    p.segments.insert(obs[0][1], (TRIMMED, ""))            # what SelectiveKV records after processing
    drop, obs = select_drop(p, Q, "bm25", keep=0.3)
    assert drop == set() and obs == []
