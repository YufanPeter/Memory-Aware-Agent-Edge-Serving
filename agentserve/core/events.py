"""Append-only event log. Every layer reports what happened here; metrics are
computed offline from the resulting JSONL, so no layer depends on metrics code."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, IO


class Clock:
    """Monotonic seconds since run start, shared by every component of a run."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.wall0 = time.time()

    def now(self) -> float:
        return time.perf_counter() - self.t0


class EventLog:
    def __init__(self, path: Path | str | None, clock: Clock | None = None) -> None:
        self.clock = clock or Clock()
        self._fh: IO[str] | None = open(path, "w") if path else None
        self.counts: dict[str, int] = {}
        self._subs: list[Callable[[str, dict[str, Any]], None]] = []

    def subscribe(self, fn: Callable[[str, dict[str, Any]], None]) -> None:
        """Observe events in-process (e.g. a scheduler watching tool_start/tool_end)
        without the emitting layer knowing about the observer."""
        self._subs.append(fn)

    def emit(self, kind: str, **fields: Any) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        for fn in self._subs:
            fn(kind, fields)
        if self._fh is None:
            return
        rec = {"t": round(self.clock.now(), 6), "kind": kind, **fields}
        self._fh.write(json.dumps(rec, default=str) + "\n")

    def flush(self) -> None:
        if self._fh:
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


class NullEventLog(EventLog):
    def __init__(self) -> None:
        super().__init__(None)
