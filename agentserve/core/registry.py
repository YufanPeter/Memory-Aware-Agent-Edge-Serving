"""Plugin registry.

Every concrete implementation registers itself under a (kind, name) pair:

    @register("scheduler", "fcfs")
    class FCFSScheduler(Scheduler): ...

and is instantiated from config by `build(kind, {"type": "fcfs", **kwargs})`.
External plugins are loaded by listing their module in an experiment's
`plugins:` list.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable

KINDS = (
    "backend", "scheduler", "retention", "task", "orchestrator", "memory", "delay",
    "arrival", "evaluator", "monitor",
)

_REGISTRY: dict[str, dict[str, type]] = {k: {} for k in KINDS}

# Modules holding the built-in implementations; imported lazily on first build.
BUILTIN_MODULES = [
    "agentserve.backends.llamacpp",
    "agentserve.backends.fake",
    "agentserve.schedulers.fcfs",
    "agentserve.schedulers.policies",
    "agentserve.schedulers.mem_aware",
    "agentserve.schedulers.retention",
    "agentserve.tasks.hotpotqa.task",
    "agentserve.agents.orchestration.react",
    "agentserve.agents.orchestration.single_shot",
    "agentserve.agents.memory.full_history",
    "agentserve.agents.memory.sliding_window",
    "agentserve.agents.tools.delay",
    "agentserve.workloads.arrival",
    "agentserve.evaluators.qa_em_f1",
    "agentserve.monitors.process_mem",
    "agentserve.monitors.backend_probe",
    "agentserve.monitors.scheduler_state",
    "agentserve.monitors.nvidia_smi",
]
_loaded = False


def register(kind: str, name: str) -> Callable[[type], type]:
    if kind not in _REGISTRY:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")

    def deco(cls: type) -> type:
        if name in _REGISTRY[kind] and _REGISTRY[kind][name] is not cls:
            raise ValueError(f"{kind}/{name} registered twice")
        _REGISTRY[kind][name] = cls
        cls.registry_name = name
        return cls

    return deco


def load_plugins(modules: list[str] | None = None) -> None:
    global _loaded
    if not _loaded:
        for m in BUILTIN_MODULES:
            importlib.import_module(m)
        _loaded = True
    for m in modules or []:
        importlib.import_module(m)


def get(kind: str, name: str) -> type:
    load_plugins()
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        avail = ", ".join(sorted(_REGISTRY[kind])) or "<none>"
        raise KeyError(f"no {kind} named {name!r} (available: {avail})") from None


def build(kind: str, cfg: dict[str, Any] | None, **extra: Any) -> Any:
    """Instantiate `cfg["type"]` of `kind` with remaining cfg keys + extra as kwargs."""
    cfg = dict(cfg or {})
    name = cfg.pop("type")
    return get(kind, name)(**cfg, **extra)


def available(kind: str) -> list[str]:
    load_plugins()
    return sorted(_REGISTRY[kind])
