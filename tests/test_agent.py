from edgeserve.agent import Agent, parse_action
from edgeserve.events import EventLog
from edgeserve.llm import MockLLM
from edgeserve.profiles import Profile


def scripted(replies):
    it = iter(replies)
    return lambda prompt: next(it)


def test_parse_action():
    assert parse_action(" I should search.\nAction 1: search[Ed Wood]") == ("search", "Ed Wood")
    assert parse_action("Thought.\nAction 3: finish[Harvard [MA]]") == ("finish", "Harvard [MA]")
    assert parse_action("Thought.\nAction: finish Peter Jacobson") == ("finish", "Peter Jacobson")
    assert parse_action("Thought.\nAction: search Peter Jacobson") is None
    assert parse_action("no action here") is None
    assert parse_action("Thought.\nAction  finish[Riders of the Purple Sage]") == ("finish", "Riders of the Purple Sage")


async def test_react_finishes_and_prompt_is_append_only(corpus, item):
    llm = MockLLM(scripted([
        " I need Scott Derrickson's nationality.\nAction 1: search[Scott Derrickson]",
        " He is American. Now Ed Wood.\nAction 2: search[Ed Wood]",
        " Both are American.\nAction 3: finish[yes]",
    ]))
    agent = Agent(llm, corpus, EventLog())
    trace = await agent.run("t0", item, Profile(name="p", few_shot=2), slot=0)

    assert trace.status == "ok" and trace.prediction == "yes"
    assert [s.action for s in trace.steps] == ["search", "search", "finish"]
    assert trace.sf_seen == item.sf_set
    prompts = [c["prompt"] for c in llm.calls]
    for prev, cur, call in zip(prompts, prompts[1:], llm.calls):
        assert cur.startswith(prev + call["text"])          # each prompt extends the last one
    # Step >= 2 reuses everything up to the previous prompt + output (mock counts 4 chars/token).
    assert all(s.llm["cache_n"] >= len(p) // 4 for s, p in zip(trace.steps[1:], prompts))
    # The segment ledger reproduces the full context exactly.
    text = "".join(t for _, t in trace.segments)
    assert text == prompts[-1] + llm.calls[-1]["text"]
    kinds = {k for k, _ in trace.segments}
    assert {"shared_prefix", "question", "thought_action", "obs_sf", "obs_other", "obs_meta", "control"} <= kinds


async def test_shared_prefix_identical_across_tasks(corpus, item):
    llm = MockLLM(lambda p: " Done.\nAction 1: finish[x]")
    agent = Agent(llm, corpus, EventLog())
    a = await agent.run("a", item, Profile(name="p"), slot=0)
    b = await agent.run("b", item, Profile(name="p"), slot=0)
    assert a.segments[0] == b.segments[0] and a.segments[0][0] == "shared_prefix"
    assert b.steps[0].llm["cache_n"] > 0


async def test_forced_finish_after_max_steps(corpus, item):
    def script(prompt):
        return "Ed Wood]" if prompt.endswith("finish[") else " Search again.\nAction: search[Ed Wood]"
    agent = Agent(MockLLM(script), corpus, EventLog())
    trace = await agent.run("t", item, Profile(name="p", max_steps=2), slot=0)
    assert trace.status == "forced" and trace.prediction == "Ed Wood"
    assert [s.kind for s in trace.steps] == ["react", "react", "forced_finish"]


async def test_parse_failures(corpus, item):
    def script(prompt):
        return "nothing]" if prompt.endswith("finish[") else " I am confused."
    agent = Agent(MockLLM(script), corpus, EventLog())
    trace = await agent.run("t", item, Profile(name="p", max_parse_errors=1), slot=0)
    assert trace.status == "parse_fail"
    assert len(trace.steps) == 3                             # 2 failed steps + forced finish


async def test_oneshot_uses_items_paragraphs(corpus, item):
    llm = MockLLM(lambda p: " Both are American.\nAction 1: finish[yes]")
    trace = await Agent(llm, corpus, EventLog()).run("t", item, Profile(name="o", mode="oneshot"), slot=0)
    assert trace.prediction == "yes" and len(trace.steps) == 1
    assert trace.sf_seen == item.sf_set
    assert "[Ed Wood (film)]" in llm.calls[0]["prompt"]


async def test_explain_is_a_fixed_length_step(corpus, item):
    llm = MockLLM(scripted([" Done.\nAction 1: finish[yes]\nExplanation: short", " Because."]))
    trace = await Agent(llm, corpus, EventLog()).run("t", item, Profile(name="e", explain_tokens=10), slot=0)
    assert trace.status == "ok" and trace.prediction == "yes"
    assert [s.kind for s in trace.steps] == ["react", "explain"]
    assert trace.steps[-1].llm["predicted_n"] == 10
    assert llm.calls[1]["prompt"].endswith("finish[yes]\nExplanation:")


async def test_oneshot_extra_paragraphs(corpus, item):
    corpus.text_search = lambda q, n: ["Ed Wood", "Doctor Strange (2016 film)"]
    llm = MockLLM(lambda p: " Yes.\nAction 1: finish[yes]")
    await Agent(llm, corpus, EventLog()).run("t", item, Profile(name="o", mode="oneshot", extra_paragraphs=5), slot=0)
    prompt = llm.calls[0]["prompt"]
    assert prompt.count("[Ed Wood]") == 1 and "[Doctor Strange (2016 film)]" in prompt


async def test_llm_error_is_classified(corpus, item):
    class Failing(MockLLM):
        async def generate(self, prompt, n_predict, stop, slot=-1, ignore_eos=False):
            res = await super().generate(prompt, n_predict, stop, slot, ignore_eos)
            res.ok, res.error = False, "HTTP 400: the request exceeds the available context size"
            return res
    trace = await Agent(Failing(lambda p: "x"), corpus, EventLog()).run("t", item, Profile(name="p"), slot=0)
    assert trace.status == "ctx_overflow"
