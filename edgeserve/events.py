"""Append-only JSONL event log. `t` is seconds since the run started (monotonic).

Listeners see every event right after it is written. The KV manager learns tool_start /
tool_end this way from the events the agent already emits, so agent.py needs no hooks. A
listener runs inside the agent's own call to emit, so its exceptions are caught here and
logged as a `kv_violation listener_error`: a manager bug must never fail an agent's task.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable


class EventLog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.t0 = time.monotonic()
        self.wall0 = time.time()
        self._f = open(path, "w", buffering=1, encoding="utf-8", newline="\n") if path else None
        self.events: list[dict[str, Any]] = []
        self.listeners: list[Callable[[dict[str, Any]], None]] = []

    def rel(self, t_monotonic: float) -> float:
        return round(t_monotonic - self.t0, 6)

    def emit(self, type_: str, **fields: Any) -> dict[str, Any]:
        ev = self._write({"t": self.rel(time.monotonic()), "type": type_, **fields})
        for listener in list(self.listeners):
            try:
                listener(ev)
            except Exception as e:   # written directly, not re-dispatched, so a failing listener cannot recurse
                name = getattr(listener, "__qualname__", repr(listener))
                self._write({"t": self.rel(time.monotonic()), "type": "kv_violation", "kind": "listener_error",
                             "detail": f"{name} on {type_}: {type(e).__name__}: {e}"[:500]})
        return ev

    def _write(self, ev: dict[str, Any]) -> dict[str, Any]:
        self.events.append(ev)
        if self._f:
            self._f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        return ev

    def close(self) -> None:
        if self._f:
            self._f.close()
