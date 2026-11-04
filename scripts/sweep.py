"""Run a grid of experiment points unattended: for every point and repeat, start a fresh
llama-server (so no cache state leaks between runs), wait for /health, replay the
workload with the runner, stop the server. Finished runs (summary.json present) are
skipped, so an interrupted sweep resumes where it stopped.

    LLAMA_SERVER=~/llama.cpp/build/bin/llama-server MODEL=~/models/<7b>.gguf \
      uv run python -m scripts.sweep configs/sweeps/e1_concurrency.yaml [--only k8,k16] [--dry-run]

Spec:
    name: e1_concurrency
    defaults: {ctx: 65536, parallel: 64, cache_ram: 0, extra: "", delay_scale: 1.0,
               pin: true, cache: true, reps: 1, task_timeout: 900}
    points:
      - {name: k1, workload: workloads/p1_mixed_50.json, k: 1}
      - {name: k8, workload: workloads/p1_mixed_200.json, k: 8, reps: 3}
`parallel: k` sets -np to the point's K (the llama.cpp-default baseline).
Results: results/<name>/<point>/rep<i>/ (runner outputs + server_cmd.txt + server.log).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

DEFAULTS: dict[str, Any] = {"ctx": 65536, "parallel": 64, "cache_ram": 0, "extra": "", "delay_scale": 1.0,
                            "pin": True, "cache": True, "reps": 1, "task_timeout": 900, "seed": 0, "port": 8080}


def wait_healthy(url: str, proc: subprocess.Popen, timeout_s: float = 300.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if proc.poll() is not None:
            return False
        try:
            if httpx.get(f"{url}/health", timeout=2.0).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    return False


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_point(name: str, p: dict[str, Any], rep: int, out: Path, dry: bool) -> None:
    url = f"http://127.0.0.1:{p['port']}"
    parallel = p["k"] if p["parallel"] == "k" else p["parallel"]
    env = {**os.environ, "CTX": str(p["ctx"]), "PARALLEL": str(parallel), "PORT": str(p["port"]),
           "CACHE_RAM": str(p["cache_ram"]), "EXTRA": p["extra"], "CMD_OUT": str(out / "server_cmd.txt")}
    runner = [sys.executable, "-m", "edgeserve.runner", "--workload", p["workload"], "--out", str(out),
              "--k", str(p["k"]), "--base-url", url, "--delay-scale", str(p["delay_scale"]),
              "--seed", str(p["seed"]), "--task-timeout", str(p["task_timeout"])]
    if not p["pin"]:
        runner.append("--no-pin")
    if not p["cache"]:
        runner.append("--no-cache")
    if dry:
        print(f"[dry] {name} rep{rep}: CTX={env['CTX']} PARALLEL={parallel} CACHE_RAM={env['CACHE_RAM']} "
              f"EXTRA='{p['extra']}' | {' '.join(runner[3:])}")
        return
    out.mkdir(parents=True, exist_ok=True)
    slot_dir = Path("/dev/shm/slots")
    for f in slot_dir.glob("*.bin") if slot_dir.exists() else []:
        f.unlink()
    log = open(out / "server.log", "w", encoding="utf-8")
    server = subprocess.Popen(["bash", "scripts/start_server.sh"], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_healthy(url, server):
            print(f"!! {name} rep{rep}: server did not become healthy (see {out / 'server.log'})")
            return
        t0 = time.monotonic()
        r = subprocess.run(runner + ["--server-pid", str(server.pid)], capture_output=True, text=True)
        (out / "runner.log").write_text(r.stdout + r.stderr, encoding="utf-8", newline="\n")
        if r.returncode != 0:
            print(f"!! {name} rep{rep}: runner failed (see {out / 'runner.log'})")
            return
        s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        print(f"{name} rep{rep} ({time.monotonic() - t0:.0f}s): completed {s['completed']}/{s['n_tasks']}, "
              f"steady {s['steady_throughput_per_min']} tasks/min, p95 {s['latency_p95_s']}s, f1 {s['f1']}",
              flush=True)
    finally:
        stop(server)
        log.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--only", help="comma-separated point names")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    base = {**DEFAULTS, **spec.get("defaults", {})}
    only = set(args.only.split(",")) if args.only else None
    for point in spec["points"]:
        if only and point["name"] not in only:
            continue
        p = {**base, **point}
        for rep in range(p["reps"]):
            out = Path("results") / spec["name"] / point["name"] / f"rep{rep}"
            if (out / "summary.json").exists():
                print(f"skip {point['name']} rep{rep} (done)")
                continue
            run_point(point["name"], p, rep, out, args.dry_run)


if __name__ == "__main__":
    main()
