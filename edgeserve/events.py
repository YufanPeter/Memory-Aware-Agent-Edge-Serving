"""Append-only JSONL event log. `t` is seconds since the run started (monotonic)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.t0 = time.monotonic()
        self.wall0 = time.time()
        self._f = open(path, "w", buffering=1, encoding="utf-8", newline="\n") if path else None
        self.events: list[dict[str, Any]] = []

    def rel(self, t_monotonic: float) -> float:
        return round(t_monotonic - self.t0, 6)

    def emit(self, type_: str, **fields: Any) -> dict[str, Any]:
        ev = {"t": self.rel(time.monotonic()), "type": type_, **fields}
        self.events.append(ev)
        if self._f:
            self._f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        return ev

    def close(self) -> None:
        if self._f:
            self._f.close()
