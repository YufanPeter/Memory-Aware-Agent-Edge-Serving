"""HotpotQA multi-hop QA task.

Workload tags understood:
  context:    short (question only; agent must search) | long (all 10 paragraphs pre-loaded)
  max_tokens: per-request generation cap
  max_steps:  per-task step cap override
"""
from __future__ import annotations

from typing import Any

from agentserve.agents.tools.base import Tool
from agentserve.core.registry import register
from agentserve.core.types import AgentContext, TaskInstance, TaskSpec
from agentserve.tasks.base import Task
from agentserve.tasks.hotpotqa.dataset import DEFAULT_PATH, HotpotItem, load_items
from agentserve.tasks.hotpotqa.tools import Corpus, LookupTool, SearchTool

INSTRUCTION = """These questions need two hops: usually you must search one entity, find a second entity in its paragraph, and search that one too.
Only finish when the observations contain the answer to exactly what the question asks.
Example:
Question: Which magazine was started first, Arthur's Magazine or First for Women?
Thought: I need to find when Arthur's Magazine started.
Action: search[Arthur's Magazine]
Observation: [Arthur's Magazine] Arthur's Magazine (1844-1846) was an American literary periodical ...
Thought: Arthur's Magazine started in 1844. Now First for Women.
Action: search[First for Women]
Observation: [First for Women] First for Women is a woman's magazine ... started in 1989.
Thought: 1844 is earlier than 1989.
Action: finish[Arthur's Magazine]"""

INSTRUCTION_LONG = """The Wikipedia paragraphs needed to answer are given in the context above the question.
Read them, reason in the Thought, and usually finish right away. The answer is a short phrase copied from the context, or yes/no.
Example:
Thought: The context says Arthur's Magazine started in 1844 and First for Women in 1989, so Arthur's Magazine is older.
Action: finish[Arthur's Magazine]"""


@register("task", "hotpotqa")
class HotpotQATask(Task):
    default_evaluators = [{"type": "qa_em_f1"}]

    def __init__(self, path: str = DEFAULT_PATH, qtypes: list[str] | None = None,
                 levels: list[str] | None = None, max_tokens: int = 256,
                 n_sentences: int = 5, **params: Any) -> None:
        super().__init__(path=path, qtypes=qtypes, levels=levels, max_tokens=max_tokens,
                         n_sentences=n_sentences, **params)
        self.path, self.qtypes, self.levels = path, qtypes, levels
        self.max_tokens, self.n_sentences = max_tokens, n_sentences

    def load(self) -> list[TaskInstance]:
        items = load_items(self.path)
        if self.qtypes:
            items = [i for i in items if i.qtype in self.qtypes]
        if self.levels:
            items = [i for i in items if i.level in self.levels]
        return [TaskInstance(instance_id=i.id, payload=i) for i in items]

    def context(self, spec: TaskSpec) -> AgentContext:
        item: HotpotItem = spec.instance.payload
        long = spec.tags.get("context", "short") == "long"
        docs = [f"[{t}] " + "".join(s).strip() for t, s in zip(item.titles, item.paragraphs)] if long else []
        return AgentContext(
            task_id=spec.task_id,
            instruction=INSTRUCTION_LONG if long else INSTRUCTION,
            query=item.question,
            documents=docs,
            max_tokens=int(spec.tags.get("max_tokens", self.max_tokens)),
            max_steps=spec.tags.get("max_steps"),
            seed=int(spec.tags.get("seed", 0)),
            tags=spec.tags,
        )

    def tools(self, spec: TaskSpec) -> list[Tool]:
        corpus = Corpus(spec.instance.payload)
        return [SearchTool(corpus, self.n_sentences), LookupTool(corpus)]

    def gold(self, instance: TaskInstance) -> Any:
        return instance.payload.answer

    def describe(self, instance: TaskInstance) -> dict[str, Any]:
        i: HotpotItem = instance.payload
        return {"qtype": i.qtype, "level": i.level}
