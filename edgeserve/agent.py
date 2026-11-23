"""The ReAct agent loop (and the one-shot variant) for one HotpotQA task.

The prompt only ever grows: the model's output is appended verbatim, then the
observation, then the next "Thought i:" label. See prompt.py for why.
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
from edgeserve.kvm import KVBudgetExceeded
from edgeserve.llm import LLMResult
from edgeserve.profiles import Profile
from edgeserve.prompt import CHATML, CLOSEDBOOK_RULES, ONESHOT_RULES, Prompt, react_system
from edgeserve.tools import ToolBox

STOP = ["\nObservation", "\nExplanation"] + CHATML["stop"]   # the Explanation is its own request
_ACTION_LINE = re.compile(r"^\s*Action\s*\d*\s*:?\s*(.*)$", re.M)   # the colon is sometimes dropped
_ACTION = re.compile(r"([A-Za-z_]+)\s*\[(.*)\]")
_FINISH_LOOSE = re.compile(r"finish\s*[:\[]?\s*(.+?)\]?\s*$", re.I)


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
    trim: dict[str, Any] | None = None   # selective KV applied during this step's tool call
    trim_deferred: bool = False          # trigger "pressure": no memory pressure, trimming postponed


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

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sf_seen"] = sorted(self.sf_seen)
        return d


class Agent:
    def __init__(self, llm: Any, corpus: Corpus, events: EventLog, seed: int = 0,
                 delay_scale: float = 1.0, kvm: Any = None, selective: Any = None) -> None:
        self.llm, self.corpus, self.events = llm, corpus, events
        self.seed, self.delay_scale = seed, delay_scale
        self.kvm = kvm   # edgeserve.kvm.KVManager, or None for the Phase 1 baseline
        self.selective = selective   # edgeserve.kvtrim.SelectiveKV, or None

    async def _generate(self, trace: TaskTrace, prompt: Prompt, step: int, kind: str,
                        n_predict: int, stop: list[str], ignore_eos: bool = False) -> tuple[LLMResult, StepRecord]:
        self.events.emit("req_start", task_id=trace.task_id, step=step, kind=kind, slot=trace.slot,
                         prompt_chars=len(prompt.text))
        if self.kvm is not None:   # may wait for room in the KV pool, restore offloaded KV, or fork the prefix
            prefix = prompt.segments[0][1] if prompt.segments and prompt.segments[0][0] == "shared_prefix" else None
            try:
                await self.kvm.before_generate(trace.slot, len(prompt.text), n_predict, prefix)
            except KVBudgetExceeded as e:
                # This request can never fit next to the golden slots: fail this task only (as a
                # context overflow) instead of blocking the agents queued behind it.
                res = LLMResult(ok=False, error=f"context exceeds the KV budget: {e}", slot=trace.slot)
                return self._record(trace, step, kind, res)
        res = await self.llm.generate(prompt.text, n_predict, stop, trace.slot, ignore_eos)
        if self.kvm is not None:
            await self.kvm.after_generate(trace.slot, len(prompt.text), res)
        return self._record(trace, step, kind, res)

    def _record(self, trace: TaskTrace, step: int, kind: str, res: LLMResult) -> tuple[LLMResult, StepRecord]:
        info = res.to_dict()
        for k in ("t_start", "t_first_token", "t_end"):
            info[k] = self.events.rel(info[k]) if info[k] else None
        self.events.emit("req_end", task_id=trace.task_id, step=step, kind=kind, **info)
        rec = StepRecord(step=step, kind=kind, llm=info, output=res.text)
        trace.steps.append(rec)
        if not res.ok:
            err = res.error or ""
            trace.status = "ctx_overflow" if "context" in err.lower() else "llm_error"
            trace.error = err
        return res, rec

    async def _trim(self, trace: TaskTrace, p: Prompt, rec: StepRecord) -> None:
        if self.selective.trigger == "pressure" and self.kvm is not None and not self.kvm.under_pressure():
            rec.trim_deferred = True   # memory is not short: leave the observations for a later tool call
            return
        r = await self.selective.trim(trace.slot, p)
        if r is None:
            return
        rec.trim = asdict(r)
        self.events.emit("trim", task_id=trace.task_id, step=rec.step, slot=trace.slot, **rec.trim)
        if self.kvm is not None and r.tokens:
            await self.kvm.on_trim(trace.slot, r.tokens, r.chars)

    async def run(self, task_id: str, item: HotpotItem, profile: Profile, slot: int = -1) -> TaskTrace:
        trace = TaskTrace(task_id=task_id, qid=item.id, profile=profile.name, qtype=item.qtype, slot=slot)
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
        parse_errors = 0
        n = 0
        for n in range(1, profile.max_steps + 1):
            if n > 1:
                p.add("control", f"\nThought {n}:")
            res, rec = await self._generate(trace, p, n, "react", profile.n_predict, STOP)
            if not res.ok:
                return
            p.add("thought_action", res.text)
            parsed = parse_action(res.text)
            if parsed is None:
                parse_errors += 1
                p.add("control", f"\nObservation {n}: ")
                p.add("obs_meta", "Invalid format. Write one Thought line, then one Action line such as "
                                  "Action: search[entity], lookup[keyword] or finish[answer].")
                if parse_errors > profile.max_parse_errors:
                    trace.status = "parse_fail"
                    break
                continue
            rec.action, rec.arg = parsed
            if rec.action == "finish":
                trace.prediction = rec.arg
                await self._explain(trace, p, n + 1, profile)
                return
            rec.tool_delay_s = delays.sample(rec.action, self.seed, trace.task_id, n)
            self.events.emit("tool_start", task_id=trace.task_id, step=n, tool=rec.action, delay_s=rec.tool_delay_s)
            trim_task = None
            if self.selective is not None:
                # Selective KV while the slot is idle. With a KV manager it runs first, so the
                # manager's give-up decision sees the trimmed size; without one it overlaps the tool.
                trim_task = asyncio.create_task(self._trim(trace, p, rec))
                if self.kvm is not None:
                    await trim_task
            if self.kvm is not None:   # the true delay is used only by the clairvoyant reference policy
                await self.kvm.on_tool_start(trace.slot, rec.tool_delay_s)
            # The tool's own compute (BM25, fuzzy title match) runs in a thread, overlapped with
            # the seeded delay: it does not block other agents, and the idle gap is
            # max(delay, compute) instead of delay + compute.
            t_tool = time.monotonic()
            obs, _ = await asyncio.gather(asyncio.to_thread(tools.call, rec.action, rec.arg),
                                          asyncio.sleep(rec.tool_delay_s))
            if trim_task is not None:
                await trim_task
            rec.obs_chars = len(obs.text)
            trace.sf_seen |= obs.sf_hits
            self.events.emit("tool_end", task_id=trace.task_id, step=n, tool=rec.action, obs_chars=rec.obs_chars,
                             idle_s=round(time.monotonic() - t_tool, 4))
            if self.kvm is not None:
                await self.kvm.on_tool_end(trace.slot)
            p.add("control", f"\nObservation {n}: ")
            p.extend(obs.pieces)
        else:
            trace.status = "forced"

        # Out of steps (or too many format errors): ask for the answer directly.
        n += 1
        p.add("control", f"\nThought {n}: I must give my final answer now.\nAction {n}: finish[")
        res, rec = await self._generate(trace, p, n, "forced_finish", profile.forced_finish_tokens,
                                        ["]", "\n"] + CHATML["stop"])
        if res.ok:
            p.add("thought_action", res.text)
            rec.action, rec.arg = "finish", res.text.strip()
            trace.prediction = rec.arg
            p.add("control", "]")
            await self._explain(trace, p, n + 1, profile)

    async def _explain(self, trace: TaskTrace, p: Prompt, step: int, profile: Profile) -> None:
        """Long-output profiles: after the answer, decode a fixed-length Explanation."""
        if profile.explain_tokens <= 0:
            return
        p.add("control", "\nExplanation:")
        res, _ = await self._generate(trace, p, step, "explain", profile.explain_tokens, [], ignore_eos=True)
        if res.ok:
            p.add("thought_action", res.text)
