"""Deterministic agent scenarios on the conftest corpus, shared by the TRIM tests.

`run_scenario(name, **agent_kwargs)` runs one scenario against MockLLM and returns what the
agent sent and recorded with the wall-clock fields removed: prompt hashes, events, and the
trace. tests/golden/agent_phase2.json holds these outputs as produced by the agent before
TRIM existed (git c7e43b0), so test_trim can prove that trim='none' is byte-identical to the
Phase 2 code path. Regenerate it only for an intended behaviour change:

    uv run --no-sync python -m tests.agent_scenarios --write

Scripts answer from the prompt alone (the number of observations in the task's own part),
never from a call counter, so an extra prefill-only request cannot shift the replies.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from edgeserve.agent import Agent
from edgeserve.corpus import Corpus
from edgeserve.data import HotpotItem
from edgeserve.events import EventLog
from edgeserve.llm import MockLLM
from edgeserve.profiles import Profile
from tests.conftest import DOCS

GOLDEN = Path(__file__).parent / "golden" / "agent_phase2.json"
TIMING = {"t", "t_start", "t_first_token", "t_end", "ttft_s", "latency_s", "idle_s"}


def make_item() -> HotpotItem:
    return HotpotItem(
        id="q1", question="Were Scott Derrickson and Ed Wood of the same nationality?", answer="yes",
        qtype="comparison", titles=["Scott Derrickson", "Ed Wood", "Ed Wood (film)"],
        paragraphs=[DOCS["Scott Derrickson"], DOCS["Ed Wood"], DOCS["Ed Wood (film)"]],
        supporting_facts=[("Scott Derrickson", 0), ("Ed Wood", 0)])


def n_obs(prompt: str) -> int:
    """Observations in the task's own part (the few-shot examples have their own)."""
    return prompt.rsplit("Question: ", 1)[-1].count("\nObservation ")


def plan_script(plan: list[str], forced: str = "Ed Wood]", explain: str = " Because.") -> Callable[[str], str]:
    def script(prompt: str) -> str:
        if prompt.endswith("finish["):
            return forced
        if prompt.endswith("Explanation:"):
            return explain
        n = n_obs(prompt)
        return plan[n] if n < len(plan) else " Done.\nAction: finish[yes]"
    return script


MIXED = [
    " I need Scott Derrickson's nationality.\nAction 1: search[Scott Derrickson]\nObservation",
    " He is American, born in 1966. Now Ed Wood.\nAction 2: search[Ed Wood]",
    " Ed Wood was an American filmmaker. Where was he born?\nAction 3: lookup[born]",
    " I am not sure what to do.",
    " Try another search.\nAction 5: search[Zzzz Qqqq]",
    " Check the film too.\nAction 6: search[Ed Wood (film)]",
    " Both are American.\nAction 7: finish[yes]",
]
SEARCHES = [
    " I need Scott Derrickson.\nAction 1: search[Scott Derrickson]",
    " He is American. Now Ed Wood.\nAction 2: search[Ed Wood]",
    " Ed Wood was American. And the film.\nAction 3: search[Doctor Strange]",
    " The film was directed by Scott Derrickson.\nAction 4: search[Ed Wood (film)]",
]
FAST_DELAY = {"type": "lognormal", "median": 0.001, "sigma": 0.5, "max_s": 0.005}

SCENARIOS: dict[str, tuple[Profile, Callable[[str], str]]] = {
    "mixed_explain": (Profile(name="mixed", few_shot=2, max_steps=8, explain_tokens=6, delay=FAST_DELAY),
                      plan_script(MIXED)),
    "forced": (Profile(name="forced", few_shot=1, max_steps=3), plan_script(SEARCHES)),
    "whole_paragraphs_forced_explain": (Profile(name="long", few_shot=1, max_steps=2, search_sentences=0,
                                                explain_tokens=4), plan_script(SEARCHES)),
    "searches_finish": (Profile(name="short", few_shot=1, max_steps=6), plan_script(SEARCHES)),
    "parse_fail": (Profile(name="pf", few_shot=1, max_parse_errors=0), plan_script([" I am confused."] * 4)),
    "oneshot": (Profile(name="o", mode="oneshot"), lambda p: " Both are American.\nAction 1: finish[yes]"),
    "closedbook": (Profile(name="c", mode="closedbook"), lambda p: " I think so.\nAction 1: finish[yes]"),
}


def strip(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in TIMING}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def run_scenario(name: str, **agent_kwargs: Any) -> dict[str, Any]:
    """Prompts (hashed), events and trace of one scenario, without wall-clock fields."""
    profile, script = SCENARIOS[name]
    llm = MockLLM(script)
    events = EventLog()
    agent = Agent(llm, Corpus(dict(DOCS)), events, seed=3, **agent_kwargs)
    trace = (await agent.run("t0", make_item(), profile, slot=0)).to_dict()
    for s in trace["steps"]:
        s["llm"] = strip(s["llm"])
    return {"prompts": [sha(c["prompt"]) for c in llm.calls],
            "prompt_chars": [len(c["prompt"]) for c in llm.calls],
            "events": [strip(e) for e in events.events], "trace": trace,
            "_calls": llm.calls}


async def all_scenarios(**agent_kwargs: Any) -> dict[str, Any]:
    out = {}
    for name in SCENARIOS:
        r = await run_scenario(name, **agent_kwargs)
        r.pop("_calls")
        out[name] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help=f"(re)write {GOLDEN}")
    args = ap.parse_args()
    out = asyncio.run(all_scenarios())
    if args.write:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                          encoding="utf-8", newline="\n")
        print(f"wrote {GOLDEN}")
    for name, r in out.items():
        print(name, r["trace"]["status"], [s["kind"] for s in r["trace"]["steps"]], len(r["events"]))


if __name__ == "__main__":
    main()
