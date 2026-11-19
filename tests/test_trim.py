"""Phase 3 TRIM: selectors, compaction whitespace, the agent's lazy and eager hooks, and the
runner / analysis / sweep plumbing. All offline (MockLLM, MockPoolLLM, the conftest corpus)."""
import asyncio
import itertools
import json

import pytest

from analysis import aggregate, kv_timeline
from edgeserve.agent import TRIM_N_PREDICT, Agent
from edgeserve.events import EventLog
from edgeserve.kvm import KVManager, ManagedLLM
from edgeserve.mockpool import MockPoolLLM
from edgeserve.policies import load_policy
from edgeserve.profiles import Profile
from edgeserve.prompt import Prompt
from edgeserve.runner import mock_script, n_steps, summarize, trim_summary
from edgeserve.trim import (MARKER, TrimContext, TrimMode, compact_observation, compact_pieces, idf_table,
                            parse_mode, reference_tokens, salient, sentence_titles, trim_pieces)
from scripts import sweep
from tests.agent_scenarios import GOLDEN, SCENARIOS, SEARCHES, make_item, plan_script, run_scenario

MODES = ["oracle", "drop", "referenced", "relevance:1"]

# Two search paragraphs and a "Similar" tail, as tools.ToolBox builds them.
OBS = [("obs_meta", "[Scott Derrickson] "),
       ("obs_sf", "Scott Derrickson (born July 16, 1966) is an American director."),
       ("obs_other", " Scott Derrickson lives in Los Angeles, California."),
       ("obs_other", " He is best known for directing horror films."),
       ("obs_meta", "\n"),
       ("obs_meta", "[Ed Wood] "),
       ("obs_other", "Edward Davis Wood Jr. was a filmmaker."),
       ("obs_sf", " He was born in Poughkeepsie, New York."),
       ("obs_meta", "\nSimilar: ['Ed Wood (film)'].")]


def text(pieces):
    return "".join(t for _, t in pieces)


# --- parsing and tokens ---------------------------------------------------------------------

def test_parse_mode():
    assert parse_mode("none") == TrimMode("none") and not parse_mode("none").active
    assert parse_mode(" Oracle ") == TrimMode("oracle")
    assert parse_mode("relevance:2") == TrimMode("relevance", 2) and str(parse_mode("relevance:2")) == "relevance:2"
    assert parse_mode("relevance:3").needs_idf and not parse_mode("referenced").needs_idf
    assert parse_mode(TrimMode("drop")) == TrimMode("drop")
    for bad in ("summary", "relevance", "relevance:0", "relevance:x", "oracle:2", ""):
        with pytest.raises(ValueError):
            parse_mode(bad)


def test_salient_tokens():
    assert salient("The García 2nd of an AI in 1966, he said") == {"garcia", "2nd", "1966", "said"}
    assert salient("Scott Derrickson's films", exclude={"scott", "derrickson"}) == {"films"}
    # the ReAct format's own words and step labels carry no content
    assert salient("I need to search Ed Wood.") == {"wood"}
    assert reference_tokens(" He was born in 1966.\nAction 2: search[Ed Wood]") == {"born", "1966", "wood"}


def test_sentence_titles_follow_title_pieces_and_fall_back_to_page():
    t = sentence_titles(OBS)
    assert t[1] == t[3] == "Scott Derrickson" and t[6] == t[7] == "Ed Wood"
    look = [("obs_meta", "(Result 1 / 2) "), ("obs_other", "He was born in Denver.")]
    assert sentence_titles(look, page="Scott Derrickson") == ["Scott Derrickson", "Scott Derrickson"]


# --- compaction whitespace --------------------------------------------------------------------

def test_oracle_keeps_framing_and_marks_each_dropped_run():
    new, st = trim_pieces(OBS, TrimMode("oracle"), TrimContext())
    assert text(new) == ("[Scott Derrickson] Scott Derrickson (born July 16, 1966) is an American director. …\n"
                         "[Ed Wood] … He was born in Poughkeepsie, New York.\nSimilar: ['Ed Wood (film)'].")
    assert [k for k, _ in new].count("obs_trim") == 2
    assert [p for p in new if p[0] == "obs_meta"] == [p for p in OBS if p[0] == "obs_meta"]
    assert (st.sents_kept, st.sents_dropped, st.sf_kept, st.sf_dropped) == (2, 3, 2, 0)
    assert st.chars_before == len(text(OBS)) and st.chars_after == len(text(new)) and st.changed


def test_drop_leaves_titles_and_one_marker_per_paragraph():
    new, st = trim_pieces(OBS, TrimMode("drop"), TrimContext())
    assert text(new) == "[Scott Derrickson] …\n[Ed Wood] …\nSimilar: ['Ed Wood (film)']."
    assert st.sf_dropped == 2 and st.sents_kept == 0


def test_marker_whitespace_cases():
    # a middle run: one marker, the kept sentence brings its own space
    mid = [("obs_meta", "[A] "), ("obs_sf", "S0."), ("obs_other", " S1."), ("obs_other", " S2."), ("obs_sf", " S3.")]
    assert text(compact_pieces(mid, {1, 4})) == "[A] S0. … S3."
    # a lookup result has no leading space and follows "(Result i / n) "
    look = [("obs_meta", "(Result 1 / 2) "), ("obs_other", "He was born in Denver.")]
    assert text(compact_pieces(look, set())) == "(Result 1 / 2) …"
    # a kept sentence that starts mid-quote (no leading space) gets one after the marker
    quote = [("obs_meta", "[A] "), ("obs_other", "First."), ("obs_other", " Second."), ("obs_sf", '", third.')]
    assert text(compact_pieces(quote, {3})) == '[A] … ", third.'
    # a dropped run after a kept sentence that already ends in whitespace
    trail = [("obs_meta", "[A] "), ("obs_sf", "S0. "), ("obs_other", "S1.")]
    assert text(compact_pieces(trail, {1})) == "[A] S0. …"
    assert compact_pieces(mid, {1, 2, 3, 4}) == mid
    assert MARKER == "…"


def test_none_and_meta_only_observations_are_unchanged():
    new, st = trim_pieces(OBS, TrimMode("none"), TrimContext())
    assert new == OBS and not st.changed and st.sents_dropped == 0
    meta = [("obs_meta", "Invalid format. Write one Thought line.")]
    for mode in MODES:
        new, st = trim_pieces(meta, parse_mode(mode), TrimContext(question="q", generated="g"))
        assert new == meta and not st.changed


# --- selectors ------------------------------------------------------------------------------

def test_referenced_ignores_the_pages_own_title_tokens():
    ctx = TrimContext(generated=" Scott Derrickson was born in 1966. Now Ed Wood.\nAction 2: search[Ed Wood]")
    kept = TrimMode("referenced").kept(OBS, ctx)
    # S0 shares born/1966; S1 shares only "scott derrickson" (its page title); Ed Wood's sentences
    # share "born" (U1) and only the title token "wood" (U0)
    assert kept == {1, 7}


def test_referenced_lookup_uses_the_searched_page():
    look = [("obs_meta", "(Result 1 / 1) "), ("obs_other", "Scott Derrickson was born in Denver.")]
    page = "Scott Derrickson"
    assert TrimMode("referenced").kept(look, TrimContext(generated="Scott Derrickson", page=page)) == set()
    assert TrimMode("referenced").kept(look, TrimContext(generated="He is from Denver", page=page)) == {1}


def test_relevance_ranks_by_idf_and_never_keeps_zero_scores(corpus):
    q = "Which American director lives in Los Angeles and made horror films?"
    idf = {"american": 0.5, "director": 2.0, "lives": 0.1, "los": 1.0, "angeles": 1.0, "made": 0.1, "horror": 4.0,
           "films": 1.0, "": 9.0}
    ctx = TrimContext(question=q, idf=idf)
    assert TrimMode("relevance", 1).kept(OBS, ctx) == {3}             # horror + films = 5.0
    assert TrimMode("relevance", 2).kept(OBS, ctx) == {3, 1}          # american + director 2.5 > lives los angeles 2.1
    assert TrimMode("relevance", 9).kept(OBS, ctx) == {1, 2, 3}       # Ed Wood's sentences share nothing
    assert TrimMode("relevance", 1).kept(OBS, TrimContext(question="Who lives in Los Angeles?", idf=idf)) == {2}
    tie = [("obs_meta", "[A] "), ("obs_other", "Horror one."), ("obs_other", " Horror two.")]
    assert TrimMode("relevance", 1).kept(tie, TrimContext(question="horror", idf=idf)) == {1}
    assert TrimMode("relevance", 1).kept(tie, TrimContext(question="horror")) == {1}   # no IDF: plain overlap


def test_idf_table_is_cached_per_corpus(corpus):
    t = idf_table(corpus)
    assert idf_table(corpus) is t
    assert t["poughkeepsie"] > t["american"] > 0 and t[""] >= t["poughkeepsie"]


def test_compact_observation_rewrites_the_slice_in_place():
    p = Prompt()
    p.add("shared_prefix", "SYS ")
    p.add("control", "\nObservation 1: ")
    p.extend(OBS)
    p.add("control", "\nThought 2:")
    alias = p.segments
    st = compact_observation(p, 2, 2 + len(OBS), TrimMode("oracle"), TrimContext())
    assert p.segments is alias and st.pieces_after == len(OBS) - 1
    assert p.text.startswith("SYS \nObservation 1: [Scott Derrickson] Scott") and p.text.endswith("York.\nSimilar:"
                                                                                                   " ['Ed Wood (film)']."
                                                                                                   "\nThought 2:")
    assert set(st.to_dict()) == {"chars_before", "chars_after", "sents_kept", "sents_dropped", "sf_kept", "sf_dropped"}
    with pytest.raises(ValueError):
        compact_observation(p, 0, 3, TrimMode("drop"), TrimContext())


# --- the agent: trim 'none' is Phase 2 --------------------------------------------------------

def _norm(x):
    return json.loads(json.dumps(x, ensure_ascii=False, sort_keys=True))


@pytest.mark.parametrize("kwargs", [{}, {"trim": "none"}, {"trim": "none", "trim_eager": True}],
                         ids=["default", "none", "none_eager"])
async def test_trim_none_is_byte_identical_to_phase2(kwargs):
    """tests/golden/agent_phase2.json was written by the agent before TRIM existed: same
    prompts (sha256), same events and traces, wall-clock fields aside."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert set(golden) == set(SCENARIOS)
    for name in SCENARIOS:
        r = await run_scenario(name, **kwargs)
        r.pop("_calls")
        assert _norm(r) == golden[name], name
        assert "trim" not in r["trace"] and all("trimmed_obs" not in s for s in r["trace"]["steps"])
        assert not any(e["type"] == "trim" or "trimmed_obs" in e for e in r["events"])


# --- the agent: lazy ----------------------------------------------------------------------------

def trims(r):
    return [(e["step"], e["obs"], e["eager"]) for e in r["events"] if e["type"] == "trim"]


async def test_lazy_compacts_each_consumed_observation_once_and_keeps_the_newest():
    r = await run_scenario("searches_finish", trim="drop")
    steps = r["trace"]["steps"]
    assert [s["kind"] for s in steps] == ["react"] * 5 and r["trace"]["status"] == "ok"
    assert trims(r) == [(3, 1, False), (4, 2, False), (5, 3, False)]   # obs 4 is the newest at the end
    assert [s.get("trimmed_obs") for s in steps] == [None, None, 1, 2, 3]
    prompts = [c["prompt"] for c in r["_calls"]]
    for k, prompt in enumerate(prompts[1:], 2):      # request k: obs k-1 full, all earlier ones compacted
        task = prompt.rsplit("Question: ", 1)[-1]
        assert task.count(f"Observation {k - 1}: [") == 1 and f"Observation {k - 1}: [" + "…" not in task
        for j in range(1, k - 1):
            line = task.split(f"Observation {j}: ", 1)[1].split("\n", 1)[0]
            assert line.endswith("] …"), (k, j, line)
    final = r["trace"]["segments"]
    assert ("obs_meta", "[Ed Wood (film)] ") in final and any(k == "obs_other" and "Tim Burton" in t
                                                              for k, t in final)
    t = r["trace"]["trim"]
    ev = [e for e in r["events"] if e["type"] == "trim"]
    assert t["obs"] == 3 and t["chars_removed"] == sum(e["chars_before"] - e["chars_after"] for e in ev) > 0
    assert t["requests"] == 0 and t["mode"] == "drop" and t["eager"] is False


async def test_lazy_compacts_before_forced_finish_and_explanation():
    r = await run_scenario("whole_paragraphs_forced_explain", trim="oracle")
    assert [s["kind"] for s in r["trace"]["steps"]] == ["react", "react", "forced_finish", "explain"]
    assert trims(r) == [(3, 1, False), (4, 2, False)]
    r = await run_scenario("mixed_explain", trim="oracle")
    assert [o for _, o, _ in trims(r)] == [1, 2, 3, 4, 5, 6]          # obs 6 before the Explanation


def _first_diff(a: str, b: str) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


@pytest.mark.parametrize("eager", [False, True], ids=["lazy", "eager"])
@pytest.mark.parametrize("mode", MODES)
async def test_prompts_are_append_only_except_inside_the_compacted_observation(mode, eager):
    for name in SCENARIOS:
        r = await run_scenario(name, trim=mode, trim_eager=eager)
        calls, steps = r["_calls"], r["trace"]["steps"]
        assert len(calls) == len(steps)
        ev = {(e["step"], e["obs"]): e for e in r["events"] if e["type"] == "trim"}
        for prev, cur, pstep, step in zip(calls, calls[1:], steps, steps[1:]):
            base = prev["prompt"] + ("" if pstep["kind"] == "trim" else prev["text"])
            k = step.get("trimmed_obs")
            if k is None:
                assert cur["prompt"].startswith(base), (name, step["step"], step["kind"])
                continue
            e = ev[(step["step"], k)]
            d = _first_diff(base, cur["prompt"])
            assert e["at_char"] <= d < e["at_char"] + e["chars_before"], (name, step["step"])
            assert cur["prompt"][:e["at_char"]] == base[:e["at_char"]]
        # every observation is compacted at most once
        obs = [o for _, o, _ in trims(r)]
        assert len(obs) == len(set(obs)), (name, obs)
        # the segment ledger still reproduces the last context (a forced finish adds its "]" after)
        assert "".join(t for _, t in r["trace"]["segments"]).startswith(calls[-1]["prompt"] + calls[-1]["text"])


async def test_oracle_keeps_every_supporting_fact_sentence():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for name in ("mixed_explain", "forced", "searches_finish"):
        r = await run_scenario(name, trim="oracle")
        sf = lambda segs: [t for k, t in segs if k == "obs_sf"]  # noqa: E731
        assert sf(r["trace"]["segments"]) == sf(golden[name]["trace"]["segments"]) != []
        assert r["trace"]["trim"]["sf_dropped"] == 0 and r["trace"]["trim"]["sf_kept"] > 0
        segs = r["trace"]["segments"]
        newest = max(i for i, (k, t) in enumerate(segs) if k == "control" and t.startswith("\nObservation "))
        if r["trace"]["trim"]["obs"] == sum(1 for k, t in segs if k == "control" and t.startswith("\nObservation ")):
            newest = len(segs)   # the newest was consumed and compacted too (before the Explanation)
        assert not any(k == "obs_other" for k, _ in segs[:newest]), name


# --- the agent: eager ------------------------------------------------------------------------

async def test_eager_sends_one_trim_request_per_tool_call_then_appends():
    r = await run_scenario("searches_finish", trim="drop", trim_eager=True)
    steps, calls = r["trace"]["steps"], r["_calls"]
    kinds = [(s["kind"], s["step"]) for s in steps]
    assert kinds == [("react", 1), ("react", 2), ("trim", 2), ("react", 3), ("trim", 3), ("react", 4), ("trim", 4),
                     ("react", 5)]
    assert trims(r) == [(2, 1, True), (3, 2, True), (4, 3, True)]
    assert r["trace"]["trim"]["requests"] == 3
    for i, s in enumerate(steps):
        if s["kind"] == "trim":
            assert s["trimmed_obs"] == s["step"] - 1 and s["llm"]["predicted_n"] == TRIM_N_PREDICT
            assert calls[i]["prompt"].endswith(steps[i - 1]["output"])   # ends with the new Thought/Action
            assert calls[i + 1]["prompt"].startswith(calls[i]["prompt"])  # the next request only appends
            assert steps[i + 1].get("trimmed_obs") is None
    # the trim request finishes before the tool starts, so the slot is compacted for the wait
    order = [(e["type"], e.get("kind"), e.get("step")) for e in r["events"] if e["type"] in ("req_end", "tool_start")]
    for st in (2, 3, 4):
        assert order.index(("req_end", "trim", st)) < order.index(("tool_start", None, st))


async def test_eager_trim_under_the_kv_manager_keeps_the_ledger_exact(corpus, item, tmp_path):
    """A compacting request is shorter than the slot's content: the full gate charges it and
    the pool never overflows; after each trim request the slot holds exactly its prompt."""
    prof = Profile(name="react_short", few_shot=1, max_steps=6)
    pool = MockPoolLLM(plan_script(SEARCHES), capacity=4096, n_slots=4, slot_dir=tmp_path, t_prefill=0.0,
                       t_decode=0.0, op_fixed_s=0.0)
    events = EventLog()
    cfg = load_policy("ours", {"ctx": 4096, "k": 2, "slot_dir": str(tmp_path)})
    kvm = KVManager(pool, events, cfg, pool, profiles={prof.name: prof}, priors={})
    events.listeners.append(kvm.on_event)
    agent = Agent(ManagedLLM(pool, kvm), corpus, events, trim="referenced", trim_eager=True)
    await kvm.start()
    tickets = itertools.count()
    traces = {}

    async def one(tid: str, w: int) -> None:
        slot = await kvm.admit(tid, prof.name, next(tickets), worker=w)
        trace = None
        try:
            trace = await agent.run(tid, item, prof, slot)
        finally:
            await asyncio.shield(kvm.release(tid, trace.status if trace else "error"))
        traces[tid] = trace
    await asyncio.gather(one("a", 0), one("b", 1))
    await kvm.close()
    assert {t.status for t in traces.values()} == {"ok"}
    assert pool.overflows == 0 and pool.purges == 0 and not kvm.violations
    for tr in traces.values():
        kinds = [s.kind for s in tr.steps]
        assert kinds.count("trim") == tr.trim["requests"] >= 1
        for s, nxt in zip(tr.steps, tr.steps[1:]):
            if s.kind == "trim":   # the next request reuses the whole compacted prompt
                assert nxt.llm["cache_n"] == s.llm["prompt_n"] + s.llm["cache_n"] + s.llm["predicted_n"] - 1


@pytest.mark.parametrize("shrink_fast", [True, False])
async def test_eager_trim_under_pressure_is_never_evicted_while_it_waits(corpus, item, tmp_path, shrink_fast):
    """Four agents in a pool that holds about three: with shrink_fast every compacting request is
    dispatched at once (it only shrinks its own resident slot), and no slot is given up while its
    owner's compaction is pending. Off, the first Phase 3 behaviour: they queue like any request."""
    prof = Profile(name="react_short", few_shot=1, max_steps=6)
    pool = MockPoolLLM(plan_script(SEARCHES), capacity=1100, n_slots=4, slot_dir=tmp_path, t_prefill=1e-5,
                       t_decode=2e-3, op_fixed_s=0.0)
    events = EventLog()
    # managed_k: full gate and managed lifecycle at a fixed K (no KV admission), so all four compete
    cfg = load_policy("managed_k", {"ctx": 1100, "k": 4, "slot_dir": str(tmp_path), "shrink_fast": shrink_fast})
    kvm = KVManager(pool, events, cfg, pool, profiles={prof.name: prof}, priors={})
    events.listeners.append(kvm.on_event)
    agent = Agent(ManagedLLM(pool, kvm), corpus, events, trim="referenced", trim_eager=True, delay_scale=0.02)
    await kvm.start()
    tickets = itertools.count()
    traces = {}

    async def one(tid: str, w: int) -> None:
        slot = await kvm.admit(tid, prof.name, next(tickets), worker=w)
        trace = None
        try:
            trace = await agent.run(tid, item, prof, slot)
        finally:
            await asyncio.shield(kvm.release(tid, trace.status if trace else "error"))
        traces[tid] = trace
    await asyncio.wait_for(asyncio.gather(*(one(f"t{i}", i) for i in range(4))), 60)
    await kvm.close()
    assert {t.status for t in traces.values()} == {"ok"}
    assert pool.overflows == 0 and pool.purges == 0
    n_trim = sum(t.trim["requests"] for t in traces.values())
    assert n_trim >= 4
    assert kvm.summary()["give_ups"] > 0   # the pool is really under pressure
    if not shrink_fast:
        assert kvm.summary()["shrink_fast"] == 0
        return
    assert not kvm.violations
    assert kvm.summary()["shrink_fast"] == n_trim
    # no give-up of a slot whose owner had a compaction pending (a trim event that changed the
    # prompt, until its request's kv_gate; an unchanged observation sends no request)
    pending: set[str] = set()
    for e in events.events:
        if e["type"] == "trim" and e.get("eager") and e["chars_after"] < e["chars_before"]:
            pending.add(e["task_id"])
        elif e["type"] in ("kv_gate", "tool_start"):
            pending.discard(e["task_id"])
        elif e["type"] == "kv_op" and e.get("owner") in pending:
            raise AssertionError(f"slot of {e['owner']} given up while its compaction waited: {e}")


# --- runner, analysis, sweep --------------------------------------------------------------------

def test_mock_script_searches():
    item = make_item()
    prompt = "SYS Question: x\nQuestion: " + item.question + "<|im_end|>\nThought 1:"
    assert mock_script(prompt) == mock_script(prompt, 1)
    assert "Action 1: search[Were Scott Derrickson]" in mock_script(prompt, 3)   # capitalized phrases in turn
    second = prompt + " a\nObservation 1: b\nThought 2:"
    assert "Action 2: search[Ed Wood]" in mock_script(second, 3) and "finish[" in mock_script(second, 1)


def _row(steps, trim=None):
    trace = {"steps": steps, **({"trim": trim} if trim else {})}
    return {"status": "ok", "em": 1.0, "f1": 1.0, "sf_recall": 1.0, "arrival_s": 0.0, "latency_s": 1.0,
            "queue_wait_s": 0.0, "n_llm_calls": n_steps(trace), "trace": trace}


def _step(kind, prompt_n, cache_n, predicted_n, trimmed=None, ttft=0.1):
    s = {"kind": kind, "step": 1, "llm": {"ok": True, "prompt_n": prompt_n, "cache_n": cache_n,
                                          "predicted_n": predicted_n, "ttft_s": ttft}}
    if trimmed is not None:
        s["trimmed_obs"] = trimmed
    return s


def test_summary_counts_trim_requests_apart_from_agent_steps():
    tr = {"mode": "drop", "eager": True, "obs": 1, "obs_changed": 1, "chars_before": 100, "chars_after": 20,
          "sents_kept": 0, "sents_dropped": 3, "sf_kept": 0, "sf_dropped": 1, "requests": 1, "chars_removed": 80}
    rows = [_row([_step("react", 1000, 0, 30), _step("react", 60, 1029, 20), _step("trim", 40, 1000, 1, trimmed=1),
                  _step("react", 50, 1040, 10)], tr)]
    s = summarize(rows, 60.0)
    assert s["steps_mean"] == 3 and s["decode_tokens_mean"] == 60          # the trim request is not a step
    assert s["prefill_tokens_total"] == 1150                                # but its prefill is server work
    t = trim_summary(rows)
    assert t["requests"] == 1 and t["trim_prefill_tokens_total"] == 40 and t["trim_cached_tokens_total"] == 1000
    assert t["kept_char_share"] == 0.2 and t["sf_kept_share"] == 0.0 and t["trim_requests_per_task"] == 1
    assert trim_summary([_row([_step("react", 10, 0, 1)])]) is None


def test_aggregate_separates_trim_losses_and_requests():
    tasks = [{"trace": {"steps": [_step("react", 1000, 0, 30), _step("react", 60, 1029, 20),
                                  _step("trim", 40, 1000, 1, trimmed=1, ttft=9.0),
                                  _step("react", 50, 1040, 10), _step("react", 500, 600, 10)]}}]
    rec, trim_lost = aggregate.recompute_split(tasks)
    # the trim request lost 1108 - 1000 cells on purpose; the last request lost 1099 - 600 to an eviction
    assert trim_lost == 1108 - 1000 and rec == 1099 - 600 and aggregate.recompute_tokens(tasks) == rec
    w = aggregate.work_row(tasks, 10.0)
    assert w["trim_requests_per_task"] == 1 and w["trim_prefill_per_task"] == 40
    assert w["decode_per_task"] == 70 and w["prefill_per_task"] == 1650
    assert w["final_context_mean"] == 1110 and w["context_peak_mean"] == 1109
    old = [{"trace": {"steps": [_step("react", 1000, 0, 30), _step("react", 60, 1029, 20)]}}]
    assert aggregate.recompute_split(old) == (0, None)


def test_kv_timeline_does_not_count_a_compaction_as_an_eviction():
    def req(t, step, kind, prompt_n, cache_n, trimmed=None):
        e = {"t": t, "type": "req_end", "task_id": "a", "slot": 0, "ok": True, "kind": kind, "step": step,
             "t_start": t, "prompt_n": prompt_n, "cache_n": cache_n, "predicted_n": 5}
        if trimmed is not None:
            e["trimmed_obs"] = trimmed
        return e
    ev = [req(0.0, 1, "react", 1000, 0), req(1.0, 2, "react", 100, 1000), req(1.5, 2, "trim", 50, 600, trimmed=1),
          req(2.0, 3, "react", 80, 650)]
    st = kv_timeline.recompute_stats(ev)
    assert st["evictions"] == 0 and st["followup_requests"] == 2 and st["trim_followups"] == 1
    assert "trim_followups" not in kv_timeline.recompute_stats(ev[:2])


def test_sweep_trim_keys(tmp_path):
    spec = {"name": "s", "defaults": {"ctx": 8192},
            "points": [{"name": "a", "workload": "w.json", "k": 4},
                       {"name": "b", "workload": "w.json", "k": 4, "policy": "ours", "trim": "relevance:2",
                        "trim_eager": True},
                       {"name": "c", "workload": "w.json", "k": 4, "trim": "oracle"}]}
    (_, a), (_, b), (_, c) = sweep.points(spec)
    assert "--trim" not in sweep.runner_args(a, tmp_path) and "--trim-eager" not in sweep.runner_args(a, tmp_path)
    args = sweep.runner_args(b, tmp_path, mock=True)
    assert args[args.index("--trim") + 1] == "relevance:2" and "--trim-eager" in args
    args = sweep.runner_args(c, tmp_path)
    assert args[-2:] == ["--trim", "oracle"] and "--trim-eager" not in args
    for bad in ({"trim": "summarise"}, {"trim_eager": True}, {"trim": "oracle", "trim_eager": "yes"}):
        with pytest.raises(SystemExit):
            sweep.points({"name": "s", "points": [{"name": "x", "workload": "w.json", "k": 1, **bad}]})
