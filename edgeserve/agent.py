"""The ReAct agent loop (and the one-shot variant) for one HotpotQA task.

The prompt only ever grows: the model's output is appended verbatim, then the
observation, then the next "Thought i:" label. See prompt.py for why.

Phase 3 (TRIM, edgeserve/trim.py) relaxes that in one controlled way. Once the model has
written the Thought/Action after an observation, the observation is "consumed" and is
rewritten once to the sentences a selector keeps. Lazy (default): the rewrite happens just
before the next request is built, so every request carries every consumed observation in
compacted form and only the newest at full length; the server re-prefills from the first
changed byte. Eager (--trim-eager): right after a tool-calling Thought/Action, before
tool_start, one prefill-only request (kind 'trim') sends the compacted prompt, so the slot
holds the compacted context during the tool wait and the next request is a pure append.
With trim 'none' none of this runs and prompts, events and traces are Phase 2's exactly.
"""
from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from edgeserve.corpus import Corpus
from edgeserve.data import HotpotItem
from edgeserve.delays import DelaySampler
from edgeserve.events import EventLog
from edgeserve.llm import LLMResult
from edgeserve.profiles import Profile
from edgeserve.prompt import CHATML, CLOSEDBOOK_RULES, ONESHOT_RULES, Prompt, react_system
from edgeserve.tools import ToolBox
from edgeserve.trim import TrimContext, TrimMode, TrimStats, compact_observation, idf_table, parse_mode

STOP = ["\nObservation", "\nExplanation"] + CHATML["stop"]   # the Explanation is its own request
_ACTION_LINE = re.compile(r"^\s*Action\s*\d*\s*:?\s*(.*)$", re.M)   # the colon is sometimes dropped
_ACTION = re.compile(r"([A-Za-z_]+)\s*\[(.*)\]")
_FINISH_LOOSE = re.compile(r"finish\s*[:\[]?\s*(.+?)\]?\s*$", re.I)
# The eager trim request only prefills. llama-server b11960 treats n_predict 0 like 1 (E0b,
# docs/phase2_spec.md §4.1), and ManagedLLM reads an effective n_predict <= 0 as kv_infeasible,
# so it asks for 1 token: the sampled token is never decoded, the slot holds exactly the prompt.
TRIM_N_PREDICT = 1
TRIM_TOTALS = ("obs", "obs_changed", "chars_before", "chars_after", "sents_kept", "sents_dropped", "sf_kept",
               "sf_dropped", "requests")


def parse_action(text: str) -> tuple[str, str] | None:
    """First `Action i: name[arg]` line in the model output (`finish answer` without
    brackets is accepted too)."""
    for line in _ACTION_LINE.findall(text):
        if m := _ACTION.search(line):
            return m.group(1).lower(), m.group(2).strip()
        if m := _FINISH_LOOSE.match(line.strip()):
            return "finish", m.group(1).strip()
    return None


@dataclass
class StepRecord:
    step: int
    kind: str                       # react | forced_finish | oneshot
    llm: dict[str, Any]
    output: str = ""
    action: str | None = None
    arg: str | None = None
    tool_delay_s: float = 0.0
    obs_chars: int = 0
    trimmed_obs: int | None = None  # TRIM: this request's prompt first differs inside this observation


@dataclass
class TaskTrace:
    task_id: str
    qid: str
    profile: str
    qtype: str
    slot: int
    status: str = "ok"              # ok | forced | parse_fail | ctx_overflow | llm_error | timeout | error
    prediction: str = ""
    error: str | None = None
    steps: list[StepRecord] = field(default_factory=list)
    segments: list[tuple[str, str]] = field(default_factory=list)
    sf_seen: set[tuple[str, int]] = field(default_factory=set)
    trim: dict[str, Any] | None = None   # TRIM per-task totals (None: trim off)

    def to_dict(self) -> dict[str, Any]:
        """Without TRIM the dict is exactly Phase 2's: the TRIM fields appear only when set."""
        d = asdict(self)
        d["sf_seen"] = sorted(self.sf_seen)
        if d["trim"] is None:
            del d["trim"]
        for s in d["steps"]:
            if s["trimmed_obs"] is None:
                del s["trimmed_obs"]
        return d


@dataclass
class _ObsSpan:
    """Where observation n sits in the Prompt's segments, and what consumed it."""
    n: int
    start: int
    end: int
    page: str | None = None          # the ToolBox page after the call (a lookup's sentences carry no title)
    generated: str | None = None     # the model's text right after it; None: not consumed yet
    trimmed: bool = False


@dataclass
class _TaskTrim:
    question: str
    spans: list[_ObsSpan] = field(default_factory=list)

    def add(self, n: int, start: int, end: int, page: str | None = None) -> None:
        self.spans.append(_ObsSpan(n, start, end, page))

    def consumed(self, text: str) -> None:
        """The model has written `text` after reading every observation so far."""
        for sp in self.spans:
            if sp.generated is None:
                sp.generated = text


class Agent:
    def __init__(self, llm: Any, corpus: Corpus, events: EventLog, seed: int = 0,
                 delay_scale: float = 1.0, trim: str | TrimMode = "none", trim_eager: bool = False) -> None:
        self.llm, self.corpus, self.events = llm, corpus, events
        self.seed, self.delay_scale = seed, delay_scale
        self.trim = parse_mode(trim)
        self.trim_eager = trim_eager and self.trim.active
        self._idf = idf_table(corpus) if self.trim.needs_idf else None

    async def _generate(self, trace: TaskTrace, prompt: Prompt, step: int, kind: str,
                        n_predict: int, stop: list[str], ignore_eos: bool = False,
                        trimmed_obs: int | None = None) -> tuple[LLMResult, StepRecord]:
        extra = {} if trimmed_obs is None else {"trimmed_obs": trimmed_obs}   # TRIM only
        self.events.emit("req_start", task_id=trace.task_id, step=step, kind=kind, slot=trace.slot,
                         prompt_chars=len(prompt.text), **extra)
        res = await self.llm.generate(prompt.text, n_predict, stop, trace.slot, ignore_eos)
        info = res.to_dict()
        for k in ("t_start", "t_first_token", "t_end"):
            info[k] = self.events.rel(info[k]) if info[k] else None
        self.events.emit("req_end", task_id=trace.task_id, step=step, kind=kind, **extra, **info)
        rec = StepRecord(step=step, kind=kind, llm=info, output=res.text, trimmed_obs=trimmed_obs)
        trace.steps.append(rec)
        if not res.ok:
            err = res.error or ""
            trace.status = "ctx_overflow" if "context" in err.lower() else "llm_error"
            trace.error = err
        return res, rec

    def _compact(self, trace: TaskTrace, p: Prompt, tt: _TaskTrim | None, step: int, eager: bool) -> int | None:
        """Compact every consumed observation not compacted yet (in the loop that is the one
        before the newest). Returns the first observation whose text changed: the request
        built next differs from the previous prompt from inside it. One 'trim' event each."""
        if tt is None:
            return None
        first: int | None = None
        for i, sp in enumerate(tt.spans):
            if sp.trimmed or sp.generated is None:
                continue
            at_char = sum(len(t) for _, t in p.segments[:sp.start])
            ctx = TrimContext(question=tt.question, generated=sp.generated, page=sp.page, idf=self._idf)
            st = compact_observation(p, sp.start, sp.end, self.trim, ctx)
            sp.trimmed = True
            if delta := st.pieces_after - st.pieces_before:
                sp.end += delta
                for later in tt.spans[i + 1:]:
                    later.start += delta
                    later.end += delta
            self._account(trace, st)
            self.events.emit("trim", task_id=trace.task_id, step=step, obs=sp.n, eager=eager, at_char=at_char,
                             **st.to_dict())
            if st.changed and first is None:
                first = sp.n
        return first

    @staticmethod
    def _account(trace: TaskTrace, st: TrimStats) -> None:
        t = trace.trim
        if t is None:
            return
        t["obs"] += 1
        t["obs_changed"] += int(st.changed)
        for k in ("chars_before", "chars_after", "sents_kept", "sents_dropped", "sf_kept", "sf_dropped"):
            t[k] += getattr(st, k)
        t["chars_removed"] = t["chars_before"] - t["chars_after"]

    async def run(self, task_id: str, item: HotpotItem, profile: Profile, slot: int = -1) -> TaskTrace:
        trace = TaskTrace(task_id=task_id, qid=item.id, profile=profile.name, qtype=item.qtype, slot=slot)
        if self.trim.active:
            trace.trim = {"mode": str(self.trim), "eager": self.trim_eager, **{k: 0 for k in TRIM_TOTALS},
                          "chars_removed": 0}
        tools = ToolBox(self.corpus, item.sf_set, profile.search_sentences, profile.neighbors)
        if profile.mode in ("oneshot", "closedbook"):
            await self._run_oneshot(trace, item, tools, profile)
        else:
            await self._run_react(trace, item, tools, profile)
        return trace

    async def _run_oneshot(self, trace: TaskTrace, item: HotpotItem, tools: ToolBox, profile: Profile) -> None:
        p = Prompt()
        if profile.mode == "closedbook":   # quality baseline: no retrieval at all
            p.add("shared_prefix", CHATML["system"].format(content=CLOSEDBOOK_RULES) + CHATML["user_open"])
            p.add("control", "Question: ")
        else:
            p.add("shared_prefix", CHATML["system"].format(content=ONESHOT_RULES) + CHATML["user_open"] + "Context:\n")
            titles = [t for t in item.titles if t in self.corpus.docs]
            if profile.extra_paragraphs > 0:   # longer context: BM25 distractors, order fixed per question
                hits = await asyncio.to_thread(self.corpus.text_search, item.question,
                                               profile.extra_paragraphs + len(titles))
                titles += [t for t in hits if t not in titles][:profile.extra_paragraphs]
                random.Random(item.id).shuffle(titles)
            ctx = tools.paragraphs(titles)
            p.extend(ctx.pieces)
            trace.sf_seen |= ctx.sf_hits
            p.add("control", "\nQuestion: ")
        p.add("question", item.question)
        p.add("control", CHATML["user_close_assistant_open"] + "Thought 1:")
        res, rec = await self._generate(trace, p, 1, "oneshot", profile.n_predict, STOP)
        if res.ok:
            p.add("thought_action", res.text)
            parsed = parse_action(res.text)
            if parsed and parsed[0] == "finish":
                rec.action, rec.arg = parsed
                trace.prediction = parsed[1]
            else:
                trace.status = "parse_fail"
        trace.segments = p.segments

    async def _run_react(self, trace: TaskTrace, item: HotpotItem, tools: ToolBox, profile: Profile) -> None:
        delays = DelaySampler(profile.delay, profile.per_tool_delay, self.delay_scale)
        p = Prompt()
        system = react_system(profile.few_shot, profile.explain_tokens > 0)
        p.add("shared_prefix", CHATML["system"].format(content=system) + CHATML["user_open"] + "Question: ")
        p.add("question", item.question)
        p.add("control", CHATML["user_close_assistant_open"] + "Thought 1:")
        trace.segments = p.segments
        tt = _TaskTrim(item.question) if self.trim.active else None
        parse_errors = 0
        n = 0
        for n in range(1, profile.max_steps + 1):
            if n > 1:
                p.add("control", f"\nThought {n}:")
            changed = self._compact(trace, p, tt, n, eager=False)
            res, rec = await self._generate(trace, p, n, "react", profile.n_predict, STOP, trimmed_obs=changed)
            if not res.ok:
                return
            p.add("thought_action", res.text)
            if tt is not None:
                tt.consumed(res.text)
            parsed = parse_action(res.text)
            if parsed is None:
                parse_errors += 1
                p.add("control", f"\nObservation {n}: ")
                start = len(p.segments)
                p.add("obs_meta", "Invalid format. Write one Thought line, then one Action line such as "
                                  "Action: search[entity], lookup[keyword] or finish[answer].")
                if tt is not None:
                    tt.add(n, start, len(p.segments))
                if parse_errors > profile.max_parse_errors:
                    trace.status = "parse_fail"
                    break
                continue
            rec.action, rec.arg = parsed
            if rec.action == "finish":
                trace.prediction = rec.arg
                await self._explain(trace, p, n + 1, profile, tt)
                return
            if self.trim_eager and (k := self._compact(trace, p, tt, n, eager=True)) is not None:
                # Before tool_start, so the manager's decision at the tool boundary sees the
                # slot's compacted size; the next request then appends to this prompt.
                res_t, _ = await self._generate(trace, p, n, "trim", TRIM_N_PREDICT, [], trimmed_obs=k)
                trace.trim["requests"] += 1
                if not res_t.ok:
                    return
            rec.tool_delay_s = delays.sample(rec.action, self.seed, trace.task_id, n)
            self.events.emit("tool_start", task_id=trace.task_id, step=n, tool=rec.action, delay_s=rec.tool_delay_s)
            # The tool's own compute (BM25, fuzzy title match) runs in a thread, overlapped with
            # the seeded delay: it does not block other agents, and the idle gap is
            # max(delay, compute) instead of delay + compute.
            t_tool = time.monotonic()
            obs, _ = await asyncio.gather(asyncio.to_thread(tools.call, rec.action, rec.arg),
                                          asyncio.sleep(rec.tool_delay_s))
            rec.obs_chars = len(obs.text)
            trace.sf_seen |= obs.sf_hits
            self.events.emit("tool_end", task_id=trace.task_id, step=n, tool=rec.action, obs_chars=rec.obs_chars,
                             idle_s=round(time.monotonic() - t_tool, 4))
            p.add("control", f"\nObservation {n}: ")
            start = len(p.segments)
            p.extend(obs.pieces)
            if tt is not None:
                tt.add(n, start, len(p.segments), tools.current)
        else:
            trace.status = "forced"

        # Out of steps (or too many format errors): ask for the answer directly.
        n += 1
        p.add("control", f"\nThought {n}: I must give my final answer now.\nAction {n}: finish[")
        changed = self._compact(trace, p, tt, n, eager=False)
        res, rec = await self._generate(trace, p, n, "forced_finish", profile.forced_finish_tokens,
                                        ["]", "\n"] + CHATML["stop"], trimmed_obs=changed)
        if res.ok:
            p.add("thought_action", res.text)
            if tt is not None:
                tt.consumed(res.text)
            rec.action, rec.arg = "finish", res.text.strip()
            trace.prediction = rec.arg
            p.add("control", "]")
            await self._explain(trace, p, n + 1, profile, tt)

    async def _explain(self, trace: TaskTrace, p: Prompt, step: int, profile: Profile,
                       tt: _TaskTrim | None = None) -> None:
        """Long-output profiles: after the answer, decode a fixed-length Explanation. With TRIM
        the observation the answer was written from is consumed too, so it is compacted here."""
        if profile.explain_tokens <= 0:
            return
        p.add("control", "\nExplanation:")
        changed = self._compact(trace, p, tt, step, eager=False)
        res, _ = await self._generate(trace, p, step, "explain", profile.explain_tokens, [], ignore_eos=True,
                                      trimmed_obs=changed)
        if res.ok:
            p.add("thought_action", res.text)
