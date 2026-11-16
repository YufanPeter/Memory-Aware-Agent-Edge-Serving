"""Count the llama-server log lines that matter for KV accounting (llama.cpp b11960).

  purging slot N with T tokens     try_clear_idle_slots (sc:1924): an idle slot evicted
                                   because a ubatch found no free cells (unified pool only)
  Context size has been exceeded   the overflow collapse (sc:4721): the "off = ..." line is
                                   one event, then one send_error line per failed request
  failed to find free space        one halving of n_batch after a failed purge (sc:4741)
  making room for prompt cache     --cache-ram host cache full, oldest entry dropped
  entry / cache size limit reached (server-task.cpp:1798, ~1918; WRN level, so always logged)
  find_slot: ... n = , used =      LLAMA_KV_CACHE_DEBUG=1 with -lv 4 (kv:895-903): per-ubatch
                                   used_max_p1 and used cells, the attention span gauge, read by
                                   find_slot_samples (analysis.kv_timeline.kvdbg_check)

Shared by scripts/sweep.py (written into summary.json["server_log"]) and analysis.aggregate
(computed from server.log when a summary predates it). Both judge the counts against the run's
gate with judge() (spec §2.4 I3).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_PURGE = re.compile(r"purging slot (\d+) with (\d+) tokens")
_FIND_SLOT = re.compile(r"find_slot: stream\[\d+\], n = *(\d+), used = *(\d+)")
_TS = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) ")   # [M.s.ms.us] prefix of every log line


def line_time_s(line: str) -> float | None:
    """Seconds since server start from the log prefix, or None."""
    m = _TS.match(line)
    if not m:
        return None
    mi, s, ms, us = (int(x) for x in m.groups())
    return mi * 60 + s + ms / 1e3 + us / 1e6


def pad256(n: int) -> int:
    return -(-n // 256) * 256


def count(path: str | Path) -> dict[str, Any]:
    """Line counts from one server.log; {} if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    c = {"purges": 0, "purged_tokens": 0, "overflow_lines": 0, "overflow_events": 0, "overflow_task_errors": 0,
         "batch_halvings": 0, "host_cache_evictions": 0, "host_cache_room_lines": 0, "restore_failures": 0}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "purging slot" in line:
                c["purges"] += 1
                if m := _PURGE.search(line):
                    c["purged_tokens"] += int(m.group(2))
            elif "Context size has been exceeded" in line:
                c["overflow_lines"] += 1
                if "off = " in line:
                    c["overflow_events"] += 1
                elif "send_error" in line:
                    c["overflow_task_errors"] += 1
            elif "failed to find free space in the KV cache" in line:
                c["batch_halvings"] += 1
            elif "making room for prompt cache entry" in line:
                c["host_cache_room_lines"] += 1
                c["host_cache_evictions"] += 1
            elif "cache size limit reached" in line or "cache token limit" in line:
                c["host_cache_evictions"] += 1
            elif "failed to restore state" in line:
                c["restore_failures"] += 1
    return c


def judge(counts: dict[str, Any], gate: str | None) -> dict[str, Any]:
    """Spec §2.4 I3: a purge is a violation only under the full gate, an overflow under full or
    active; with no gate (baselines) neither is. Idempotent."""
    if counts:
        counts["gate"] = gate
        counts["purge_violations"] = counts.get("purges", 0) if gate == "full" else 0
        counts["overflow_violations"] = counts.get("overflow_events", 0) if gate in ("full", "active") else 0
    return counts


def find_slot_samples(path: str | Path) -> list[tuple[float, int, int]]:
    """(server time s, used_max_p1, used) per ubatch from a LLAMA_KV_CACHE_DEBUG log."""
    out: list[tuple[float, int, int]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "find_slot:" in line and (m := _FIND_SLOT.search(line)) and (t := line_time_s(line)) is not None:
                out.append((t, int(m.group(1)), int(m.group(2))))
    return out


def first_launch_s(path: str | Path) -> float | None:
    """Server time of the first task launch, used to align server time with the runner's clock."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "launch_slot_" in line and "processing task" in line:
                return line_time_s(line)
    return None
