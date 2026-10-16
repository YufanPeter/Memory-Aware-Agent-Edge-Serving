"""llama.cpp `llama-server` backend.

Memory model: llama-server allocates the KV cache once at startup, sized by
`ctx_size` (tokens). With `kv_unified: true` all `parallel` slots share that one
pool, so `ctx_size` is the KV memory budget of the run. Note that `cache_ram`
(host-RAM prompt cache, default 8 GiB in recent builds) adds memory outside the
KV pool; it is set explicitly so its effect is controlled.

Observed behaviour (build 11429): idle slots keep their KV in the shared pool;
when a new request needs room the server purges idle slots on its own; if the
*active* requests do not fit, every in-flight request fails with HTTP 500
"Context size has been exceeded".

Generation uses /completion on a prompt rendered by /apply-template, so a
partially generated assistant turn (`req.prefix`) can be continued exactly —
the chat endpoint cannot do that.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import hashlib
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import httpx

from agentserve.backends.base import Backend
from agentserve.core.registry import register
from agentserve.core.types import GenerationRequest, GenerationResult

_CTX_ERR = re.compile(r"context|exceed|too long|n_ctx", re.I)


@register("backend", "llamacpp")
class LlamaCppBackend(Backend):
    def __init__(
        self,
        model: dict[str, Any],
        binary: str = "llama-server",
        host: str = "127.0.0.1",
        port: int = 8090,
        parallel: int = 4,
        ctx_size: int = 16384,
        kv_unified: bool = True,
        flash_attn: str = "on",
        n_gpu_layers: int = 99,
        cache_ram: int = 0,
        cache_prompt: bool = True,
        batch_size: int | None = None,
        ubatch_size: int | None = None,
        threads: int | None = None,
        extra_args: list[str] | None = None,
        request_timeout: float = 300.0,
        startup_timeout: float = 180.0,
        log_path: str | None = None,
        manage_server: bool = True,
        slot_save_path: str | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(model, **kw)
        self.params = dict(
            binary=binary, host=host, port=port, parallel=parallel, ctx_size=ctx_size,
            kv_unified=kv_unified, flash_attn=flash_attn, n_gpu_layers=n_gpu_layers,
            cache_ram=cache_ram, cache_prompt=cache_prompt, batch_size=batch_size,
            ubatch_size=ubatch_size, threads=threads, extra_args=extra_args or [],
            request_timeout=request_timeout, manage_server=manage_server,
            slot_save_path=slot_save_path,
        )
        self.base_url = f"http://{host}:{port}"
        self.startup_timeout = startup_timeout
        self.log_path = log_path
        self._proc: subprocess.Popen | None = None
        self._log_fh = None
        self._client: httpx.AsyncClient | None = None
        self._version: str | None = None
        self._templates: OrderedDict[str, str] = OrderedDict()  # messages-hash -> rendered prompt
        self._slot_dir: Path | None = None

    # -- lifecycle -----------------------------------------------------------

    def command(self) -> list[str]:
        p = self.params
        model_path = self.model["path"]
        cmd = [
            p["binary"], "-m", str(model_path),
            "--host", p["host"], "--port", str(p["port"]),
            "-np", str(p["parallel"]), "-c", str(p["ctx_size"]),
            "-fa", p["flash_attn"], "-ngl", str(p["n_gpu_layers"]),
            "-cram", str(p["cache_ram"]), "--metrics", "--slots",
            "-kvu" if p["kv_unified"] else "-no-kvu",
        ]
        if self._slot_dir is not None:
            cmd += ["--slot-save-path", str(self._slot_dir)]
        for flag, key in (("-b", "batch_size"), ("-ub", "ubatch_size"), ("-t", "threads")):
            if p[key] is not None:
                cmd += [flag, str(p[key])]
        return cmd + list(p["extra_args"])

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.params["request_timeout"], connect=10.0),
                limits=httpx.Limits(max_connections=256, max_keepalive_connections=256),
            )
        if self._slot_dir is None:
            d = self.params["slot_save_path"] or tempfile.mkdtemp(prefix="llama-slots-")
            self._slot_dir = Path(d).resolve()
            self._slot_dir.mkdir(parents=True, exist_ok=True)
        if self.params["manage_server"] and self._proc is None:
            if not Path(self.model["path"]).exists():
                raise FileNotFoundError(f"model not found: {self.model['path']} (run scripts/download_model.sh)")
            if shutil.which(self.params["binary"]) is None:
                raise FileNotFoundError(f"{self.params['binary']} not on PATH (run scripts/setup_llamacpp.sh)")
            self._log_fh = open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
            self._proc = subprocess.Popen(
                self.command(), stdout=self._log_fh, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        await self._wait_healthy()
        try:
            props = (await self._client.get("/props")).json()
            self._version = props.get("build_info") or props.get("system_fingerprint")
        except Exception:
            pass

    async def _wait_healthy(self) -> None:
        assert self._client is not None
        deadline = asyncio.get_running_loop().time() + self.startup_timeout
        while True:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self._proc.returncode}; see {self.log_path}")
            try:
                r = await self._client.get("/health")
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError("llama-server did not become healthy")
            await asyncio.sleep(0.5)

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._proc is not None:
            self._proc.terminate()
            try:
                await asyncio.to_thread(self._proc.wait, 20)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                await asyncio.to_thread(self._proc.wait)
            self._proc = None
        if self._log_fh not in (None, subprocess.DEVNULL):
            self._log_fh.close()
        self._log_fh = None
        if self._slot_dir is not None:
            for f in self._slot_dir.glob("*.bin"):
                f.unlink(missing_ok=True)

    # -- generation ----------------------------------------------------------

    async def _render(self, messages: list[dict[str, str]]) -> str:
        assert self._client is not None
        key = hashlib.sha1(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        if (hit := self._templates.get(key)) is not None:
            self._templates.move_to_end(key)
            return hit
        r = await self._client.post("/apply-template", json={"messages": messages})
        r.raise_for_status()
        prompt = r.json()["prompt"]
        self._templates[key] = prompt
        if len(self._templates) > 512:
            self._templates.popitem(last=False)
        return prompt

    async def generate(self, req: GenerationRequest) -> GenerationResult:
        assert self._client is not None, "backend not started"
        res = GenerationResult(request_id=req.request_id)
        chunks: list[str] = []
        res.t_send = self.clock.now()
        try:
            prompt = await self._render(req.messages) + req.prefix
            body: dict[str, Any] = {
                "prompt": prompt,
                "n_predict": req.max_tokens,
                "temperature": req.temperature,
                "seed": req.seed,
                "stream": True,
                "cache_prompt": self.params["cache_prompt"],
            }
            if req.stop:
                body["stop"] = req.stop
            if req.slot is not None:
                body["id_slot"] = req.slot
            async with self._client.stream("POST", "/completion", json=body) as r:
                if r.status_code != 200:
                    msg = (await r.aread()).decode(errors="replace")[:500]
                    res.status = "ctx_overflow" if _CTX_ERR.search(msg) else "error"
                    res.error = f"HTTP {r.status_code}: {msg}"
                    return res
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    ev = json.loads(line[6:])
                    if "error" in ev:
                        msg = str(ev["error"])
                        res.status = "ctx_overflow" if _CTX_ERR.search(msg) else "error"
                        res.error = msg[:500]
                        break
                    if piece := ev.get("content"):
                        if res.t_first_token is None:
                            res.t_first_token = self.clock.now()
                        chunks.append(piece)
                    if ev.get("stop"):
                        t = ev.get("timings") or {}
                        res.cached_tokens = int(t.get("cache_n", 0))
                        res.prompt_tokens = res.cached_tokens + int(t.get("prompt_n", 0))
                        res.completion_tokens = int(ev.get("tokens_predicted", t.get("predicted_n", 0)))
                        res.prefill_ms, res.decode_ms = t.get("prompt_ms"), t.get("predicted_ms")
                        res.finish_reason = "length" if ev.get("stop_type") == "limit" else "stop"
                        res.slot = ev.get("id_slot")
                        break
        except httpx.TimeoutException as e:
            res.status, res.error = "timeout", repr(e)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
            res.status, res.error = "error", repr(e)
        finally:
            res.t_done = self.clock.now()
            res.text = "".join(chunks)
        return res

    # -- slot control ----------------------------------------------------------

    async def _slot_action(self, slot: int, action: str, filename: str | None = None) -> dict[str, Any]:
        assert self._client is not None
        t0 = self.clock.now()
        body = {"filename": filename} if filename else {}
        r = await self._client.post(f"/slots/{slot}", params={"action": action}, json=body)
        r.raise_for_status()
        d = r.json()
        tokens = d.get("n_erased", d.get("n_saved", d.get("n_restored", 0)))
        nbytes = d.get("n_written", d.get("n_read"))
        return {"tokens": int(tokens), "bytes": nbytes, "ms": (self.clock.now() - t0) * 1e3}

    async def slot_erase(self, slot: int) -> dict[str, Any]:
        return await self._slot_action(slot, "erase")

    async def slot_save(self, slot: int, name: str) -> dict[str, Any]:
        return await self._slot_action(slot, "save", f"{name}.bin")

    async def slot_restore(self, slot: int, name: str) -> dict[str, Any]:
        return await self._slot_action(slot, "restore", f"{name}.bin")

    async def slot_discard(self, name: str) -> None:
        if self._slot_dir is not None:
            (self._slot_dir / f"{name}.bin").unlink(missing_ok=True)

    # -- introspection -------------------------------------------------------

    def capabilities(self) -> dict[str, Any]:
        p = self.params
        return {
            "max_concurrency": p["parallel"],
            "kv_budget_tokens": p["ctx_size"],
            "kv_unified": p["kv_unified"],
            "kv_bytes_per_token": self.model.get("kv_bytes_per_token"),
            "prefix_cache": p["cache_prompt"],
            "continuation": True,
            "slot_control": True,
            "slot_save": True,
            "kv_metrics": False,  # this llama.cpp build exposes no KV-usage gauge
            "pid": self.pid(),
        }

    async def probe(self) -> dict[str, Any]:
        if self._client is None:
            return {}
        out: dict[str, Any] = {}
        try:
            text = (await self._client.get("/metrics", timeout=2.0)).text
            for line in text.splitlines():
                if line.startswith("llamacpp:"):
                    k, _, v = line.partition(" ")
                    out[k.removeprefix("llamacpp:")] = float(v)
        except Exception:
            pass
        try:
            slots = (await self._client.get("/slots", timeout=2.0)).json()
            out["slots_busy"] = sum(1 for s in slots if s.get("is_processing"))
        except Exception:
            pass
        return out

    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def meta(self) -> dict[str, Any]:
        m = super().meta()
        m.update(command=self.command(), server_version=self._version, env_cuda=os.environ.get("CUDA_VISIBLE_DEVICES"))
        return m
