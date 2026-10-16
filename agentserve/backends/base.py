"""L1 Backend interface: an inference runtime serving one fixed model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentserve.core.events import Clock
from agentserve.core.types import GenerationRequest, GenerationResult


class Backend(ABC):
    registry_name: str = "?"

    def __init__(self, model: dict[str, Any], clock: Clock | None = None, **params: Any) -> None:
        # `model` is the L0 model config (pure data); backends interpret what they need.
        self.model = model
        self.params = params
        self.clock = clock or Clock()

    async def start(self) -> None:
        """Bring the runtime up (spawn server, load weights). Must be idempotent."""

    async def stop(self) -> None:
        """Release all resources."""

    @abstractmethod
    async def generate(self, req: GenerationRequest) -> GenerationResult:
        """Run one request to completion. Must never raise for request-level
        failures; report them via GenerationResult.status instead."""

    def capabilities(self) -> dict[str, Any]:
        """What this runtime exposes, so schedulers/monitors can degrade gracefully.
        Keys: max_concurrency (= number of slots), kv_budget_tokens, prefix_cache,
        continuation (honours GenerationRequest.prefix), slot_control (slot pin/erase),
        slot_save (save/restore), kv_metrics, pid."""
        return {}

    # -- optional slot control (capabilities: slot_control / slot_save) --------
    # Each returns a dict with at least {"tokens": int, "ms": float}.

    async def slot_erase(self, slot: int) -> dict[str, Any]:
        raise NotImplementedError(f"{self.registry_name} has no slot control")

    async def slot_save(self, slot: int, name: str) -> dict[str, Any]:
        raise NotImplementedError(f"{self.registry_name} cannot save slots")

    async def slot_restore(self, slot: int, name: str) -> dict[str, Any]:
        raise NotImplementedError(f"{self.registry_name} cannot restore slots")

    async def slot_discard(self, name: str) -> None:
        """Delete a saved slot image."""

    async def probe(self) -> dict[str, Any]:
        """Cheap runtime-side metrics snapshot (queue depth, cache stats...)."""
        return {}

    def pid(self) -> int | None:
        """OS pid of the process holding model/KV memory, if any."""
        return None

    def meta(self) -> dict[str, Any]:
        """Version / launch info recorded with each run for reproducibility."""
        return {"backend": self.registry_name, "params": self.params, "model": self.model}
