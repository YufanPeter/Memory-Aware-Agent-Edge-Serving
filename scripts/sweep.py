"""Run a grid of experiment points unattended: for every point and repeat, start a fresh
llama-server (so no cache state leaks between runs), wait for /health, replay the
workload with the runner, stop the server. Finished runs (summary.json present) are
skipped, so an interrupted sweep resumes where it stopped.

    LLAMA_SERVER=~/llama.cpp/build/bin/llama-server MODEL=~/models/<7b>.gguf \
      uv run python -m scripts.sweep configs/sweeps/e1_concurrency.yaml [--only k8,k16] [--dry-run]
    uv run --no-sync python -m scripts.sweep configs/sweeps/p1_dev.yaml --mock --root results/mock_sweeps

Spec:
    name: e1_concurrency
    defaults: {ctx: 65536, parallel: 64, cache_ram: 0, extra: "", delay_scale: 1.0,
               pin: true, cache: true, reps: 1, task_timeout: 900}
    points:
      - {name: k1, workload: workloads/p1_mixed_50.json, k: 1}
      - {name: k8, workload: workloads/p1_mixed_200.json, k: 8, reps: 3}
      - {name: c16k_ours, ctx: 16384, k: 16, policy: ours, kvm: {omega: 1.0}}
      - {name: c16k_kvdbg, ctx: 16384, k: 16, policy: ours, env: {LLAMA_KV_CACHE_DEBUG: "1"}, extra: "-lv 4"}
`parallel: k` sets -np to the point's K (the llama.cpp-default baseline).
Phase 2 keys: `policy` (a preset of configs/kvm.yaml, passed as --policy), `kvm` (overrides,
passed as --kvm JSON; `ctx` is added from the point unless given, so the runner's /props check
compares against the sweep's budget; with --no-kv-unified it is the per-slot n_ctx the server
reports), `kvm_config` (--kvm-config), `env` (merged into the server's environment only).
A point's `kvm` and `env` are merged key by key over the defaults' (other keys replace).
A point without `policy` runs exactly the Phase 1 command line.
Results: results/<name>/<point>/rep<i>/ (runner outputs + server_cmd.txt + server.log). After
each run, the server log's purge / overflow / host-cache lines are counted into
summary.json["server_log"] (analysis/server_log.py).
`--mock` replays every point against the runner's mock LLM, with no server, to check a sweep
file end to end on a laptop.
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

from analysis import server_log

DEFAULTS: dict[str, Any] = {"ctx": 65536, "parallel": 64, "cache_ram": 0, "extra": "", "delay_scale": 1.0,
                            "pin": True, "cache": True, "reps": 1, "task_timeout": 900, "seed": 0, "port": 8080,
                            "policy": None, "kvm": {}, "kvm_config": None, "env": {}}
SLOT_DIR = "/dev/shm/slots"


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


def n_parallel(p: dict[str, Any]) -> int:
    return p["k"] if p["parallel"] == "k" else p["parallel"]


def expected_ctx(p: dict[str, Any]) -> int:
    """The n_ctx /props reports: -c, or -c / -np per slot without the unified pool."""
    return p["ctx"] // n_parallel(p) if "--no-kv-unified" in p["extra"] else p["ctx"]


def kvm_overrides(p: dict[str, Any]) -> dict[str, Any]:
    return {"ctx": expected_ctx(p), **(p["kvm"] or {})} if p["policy"] else {}


def server_env(p: dict[str, Any], out: Path) -> dict[str, str]:
    """The point's `env` first, so the sweep's own variables always win."""
    return {**os.environ, **{k: str(v) for k, v in (p["env"] or {}).items()},
            "CTX": str(p["ctx"]), "PARALLEL": str(n_parallel(p)), "PORT": str(p["port"]),
            "CACHE_RAM": str(p["cache_ram"]), "EXTRA": p["extra"], "CMD_OUT": str(out / "server_cmd.txt")}


def runner_args(p: dict[str, Any], out: Path, mock: bool = False) -> list[str]:
    url = f"http://127.0.0.1:{p['port']}"
    runner = [sys.executable, "-m", "edgeserve.runner", "--workload", p["workload"], "--out", str(out),
              "--k", str(p["k"]), "--base-url", url, "--delay-scale", str(p["delay_scale"]),
              "--seed", str(p["seed"]), "--task-timeout", str(p["task_timeout"])]
    if not p["pin"]:
        runner.append("--no-pin")
    if not p["cache"]:
        runner.append("--no-cache")
    if p["policy"]:
        runner += ["--policy", str(p["policy"])]
        if kvm := kvm_overrides(p):
            runner += ["--kvm", json.dumps(kvm, sort_keys=True)]
        if p["kvm_config"]:
            runner += ["--kvm-config", str(p["kvm_config"])]
    if mock:   # a Phase 2 point gets a MockPoolLLM shaped like its server (-c from --kvm ctx)
        runner.append("--mock")
        if p["policy"]:
            runner += ["--mock-slots", str(n_parallel(p))]
            if p["cache_ram"]:
                runner += ["--mock-cache-ram", str(p["cache_ram"])]
    return runner


def server_log_summary(out: Path) -> dict[str, Any]:
    """Counts from server.log, judged against the run's gate (spec §2.4 I3): with the full gate a
    purge or an overflow is a violation; with the active gate only an overflow is; with no gate
    (baselines) neither is."""
    counts = server_log.count(out / "server.log")
    if not counts:
        return {}
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return server_log.judge(counts, (meta.get("kvm") or {}).get("gate"))


def add_server_log(out: Path) -> dict[str, Any]:
    sp = out / "summary.json"
    s = json.loads(sp.read_text(encoding="utf-8"))
    if counts := server_log_summary(out):
        s["server_log"] = counts
        sp.write_text(json.dumps(s, indent=1), encoding="utf-8", newline="\n")
    return s


def check_ctx(p: dict[str, Any], out: Path) -> str | None:
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    got = (meta.get("kvm") or {}).get("ctx")
    if p["policy"] and got is not None and int(got) != expected_ctx(p):
        return f"kvm.ctx {got} != sweep ctx {expected_ctx(p)}"
    return None


def report(name: str, rep: int, out: Path, s: dict[str, Any], dt: float) -> None:
    line = (f"{name} rep{rep} ({dt:.0f}s): completed {s['completed']}/{s['n_tasks']}, "
            f"steady {s['steady_throughput_per_min']} tasks/min, p95 {s['latency_p95_s']}s, f1 {s['f1']}")
    if sl := s.get("server_log"):
        line += f", purges {sl['purges']}, overflows {sl['overflow_events']}"
        if sl.get("purge_violations") or sl.get("overflow_violations"):
            line += f" (VIOLATION: purge {sl['purge_violations']}, overflow {sl['overflow_violations']})"
    if (kv := s.get("kv")) and isinstance(kv, dict) and kv.get("violations"):
        line += f", kv violations {kv['violations']}"
    print(line, flush=True)


def wall_timeout(p: dict[str, Any]) -> float:
    """Upper bound on one runner's wall time, so a hung run cannot keep the GPU host busy: every
    task at its timeout, K at a time, plus a margin."""
    try:
        n = len(json.loads(Path(p["workload"]).read_text(encoding="utf-8"))["tasks"])
    except (OSError, KeyError, TypeError, ValueError):
        n = 1000
    return float(p["task_timeout"]) * (-(-n // max(1, int(p["k"]))) + 1) + 600.0


def run_runner(cmd: list[str], out: Path, name: str, rep: int, timeout_s: float) -> bool:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout_s)
        log, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        log = "".join(x.decode("utf-8", "replace") if isinstance(x, bytes) else (x or "") for x in (e.stdout, e.stderr))
        log, rc = log + f"\n!! runner killed after {timeout_s:.0f}s\n", -1
    (out / "runner.log").write_text(log, encoding="utf-8", newline="\n")
    if rc != 0:
        print(f"!! {name} rep{rep}: runner failed (see {out / 'runner.log'})")
    return rc == 0


def run_point(name: str, p: dict[str, Any], rep: int, out: Path, dry: bool, mock: bool = False) -> None:
    url = f"http://127.0.0.1:{p['port']}"
    env = server_env(p, out)
    runner = runner_args(p, out, mock)
    if dry:
        extra_env = "".join(f" {k}={v}" for k, v in sorted((p["env"] or {}).items()))
        print(f"[dry] {name} rep{rep}: CTX={env['CTX']} PARALLEL={env['PARALLEL']} CACHE_RAM={env['CACHE_RAM']} "
              f"EXTRA='{p['extra']}'{extra_env} | {' '.join(runner[3:])}")
        return
    out.mkdir(parents=True, exist_ok=True)
    if mock:
        t0 = time.monotonic()
        if run_runner(runner, out, name, rep, wall_timeout(p)):
            report(name, rep, out, json.loads((out / "summary.json").read_text(encoding="utf-8")),
                   time.monotonic() - t0)
        return
    for d in {Path(SLOT_DIR), Path((p["kvm"] or {}).get("slot_dir", SLOT_DIR))}:
        for f in d.glob("*.bin") if d.exists() else []:
            f.unlink()
    log = open(out / "server.log", "w", encoding="utf-8")
    server = subprocess.Popen(["bash", "scripts/start_server.sh"], env=env, stdout=log, stderr=subprocess.STDOUT)
    ok = False
    t0 = time.monotonic()
    try:
        if not wait_healthy(url, server):
            print(f"!! {name} rep{rep}: server did not become healthy (see {out / 'server.log'})")
            return
        t0 = time.monotonic()
        ok = run_runner(runner + ["--server-pid", str(server.pid)], out, name, rep, wall_timeout(p))
    finally:
        stop(server)
        log.close()
    if ok:   # the log is complete only once the server has exited
        s = add_server_log(out)
        report(name, rep, out, s, time.monotonic() - t0)
        if msg := check_ctx(p, out):
            print(f"!! {name} rep{rep}: {msg}")


def points(spec: dict[str, Any], only: set[str] | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Points merged over the defaults; `kvm` and `env` merge key by key."""
    base = {**DEFAULTS, **spec.get("defaults", {})}
    out = []
    for pt in spec["points"]:
        if only and pt["name"] not in only:
            continue
        p = {**base, **pt}
        for key in ("kvm", "env"):
            p[key] = {**(base.get(key) or {}), **(pt.get(key) or {})}
        out.append((pt["name"], p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--only", help="comma-separated point names")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true", help="no server: run every point against the mock LLM")
    ap.add_argument("--root", default="results", help="results root (default results/)")
    args = ap.parse_args()
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    only = set(args.only.split(",")) if args.only else None
    for name, p in points(spec, only):
        for rep in range(p["reps"]):
            out = Path(args.root) / spec["name"] / name / f"rep{rep}"
            if (out / "summary.json").exists():
                # A Phase 2 run interrupted between the runner and the log count gets its count now.
                if not args.dry_run and (out / "meta.json").exists() and (out / "server.log").exists():
                    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
                    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
                    if "kvm" in meta and "server_log" not in s:
                        add_server_log(out)
                print(f"skip {name} rep{rep} (done)")
                continue
            run_point(name, p, rep, out, args.dry_run, args.mock)


if __name__ == "__main__":
    main()
