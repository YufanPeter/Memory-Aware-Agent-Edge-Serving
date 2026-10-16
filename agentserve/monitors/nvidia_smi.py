"""GPU memory/utilization via nvidia-smi (auto-skipped when not present)."""
from __future__ import annotations

import asyncio
import shutil
from typing import Any

from agentserve.core.registry import register
from agentserve.monitors.base import Monitor, MonitorContext

_FIELDS = ["memory.used", "memory.total", "utilization.gpu"]


@register("monitor", "nvidia_smi")
class NvidiaSmiMonitor(Monitor):
    def __init__(self, interval: float = 0.5, gpu: int = 0, **params: Any) -> None:
        super().__init__(interval, **params)
        self.gpu = gpu

    def available(self, ctx: MonitorContext) -> bool:
        return shutil.which("nvidia-smi") is not None

    async def sample(self, ctx: MonitorContext) -> dict[str, Any] | None:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", f"--id={self.gpu}", f"--query-gpu={','.join(_FIELDS)}",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        try:
            used, total, util = (float(x) for x in out.decode().strip().split(","))
        except ValueError:
            return None
        return {"gpu_mem_used_bytes": used * 2**20, "gpu_mem_total_bytes": total * 2**20, "gpu_util": util}
