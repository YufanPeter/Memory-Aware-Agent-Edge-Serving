"""Memory of the backend process. On macOS also reads `phys_footprint`, which
(unlike RSS) includes Metal/GPU allocations on unified memory."""
from __future__ import annotations

import ctypes
import ctypes.util
import sys
from typing import Any

import psutil

from agentserve.core.registry import register
from agentserve.monitors.base import Monitor, MonitorContext

_libc = ctypes.CDLL(ctypes.util.find_library("c")) if sys.platform == "darwin" else None
_RUSAGE_INFO_V2 = 2


def darwin_phys_footprint(pid: int) -> int | None:
    """proc_pid_rusage(RUSAGE_INFO_V2): 16-byte uuid followed by uint64 fields;
    ri_phys_footprint is the 8th uint64."""
    if _libc is None:
        return None
    buf = (ctypes.c_uint64 * 40)()
    if _libc.proc_pid_rusage(pid, _RUSAGE_INFO_V2, ctypes.byref(buf)) != 0:
        return None
    return int(buf[2 + 7])  # uuid occupies two uint64 slots


@register("monitor", "process_mem")
class ProcessMemMonitor(Monitor):
    def __init__(self, interval: float = 0.2, include_children: bool = True, **params: Any) -> None:
        super().__init__(interval, **params)
        self.include_children = include_children
        self._proc: psutil.Process | None = None

    def available(self, ctx: MonitorContext) -> bool:
        return ctx.backend.pid() is not None

    async def sample(self, ctx: MonitorContext) -> dict[str, Any] | None:
        pid = ctx.backend.pid()
        if pid is None:
            return None
        if self._proc is None or self._proc.pid != pid:
            self._proc = psutil.Process(pid)
        try:
            out = {"rss_bytes": self._proc.memory_info().rss}
            if (fp := darwin_phys_footprint(pid)) is not None:
                out["phys_footprint_bytes"] = fp
            out["sys_used_bytes"] = psutil.virtual_memory().used
            return out
        except psutil.Error:
            return None
