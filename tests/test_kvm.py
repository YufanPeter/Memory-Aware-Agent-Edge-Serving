"""KV manager against MockPoolLLM (docs/phase2_spec.md §7, tests 3-8 and 10-22).

Synthetic agents mimic agent.py's call pattern exactly (req_start, generate, req_end,
tool_start, sleep, tool_end, append the observation), with prompts built from 4-character
chunks so token counts are exact: a 120-token family prefix, a per-task tag, then the task's
own question, outputs and observations. Times are scaled down ~100x from the GPU (the cost
model's priors with them), so the ratio of op costs to tool delays stays realistic.
"""
import asyncio
import itertools
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from edgeserve.agent import Agent
from edgeserve.events import EventLog
from edgeserve import kvm as kvm_mod
from edgeserve.kvm import (ACTIVE, BYTES_PER_TOKEN, DROPPED, FREE, HOST_CACHED, IDLE, OFFLOADED, READY, RESTORING,
                           STALE, UNKNOWN, CostModel, KVManager, ManagedLLM, _Waiter, default_families)
from edgeserve.llm import SlotOpResult
from edgeserve.mockpool import MockPoolLLM
from edgeserve.policies import Victim, load_policy
from edgeserve.profiles import Profile
from edgeserve.runner import kv_status, mock_script

PREFIX = "".join(f"P{i:03d}" for i in range(120))     # 120 tokens shared by every synthetic agent
FAMILIES = ({"syn": "syn", "solo": None}, {"syn": (PREFIX + "Xaaa", PREFIX + "Ybbb")})
OUT = 20
GPU_FAST = {"t_prefill": 2.2e-6, "t_decode": 1e-5, "op_fixed_s": 2e-5, "kappa_s": 3.5e-7, "kappa_r": 1.5e-7}


def fast_cost(**kw: Any) -> CostModel:
    return CostModel(**({"kappa_s": 3.5e-7, "kappa_r": 1.5e-7, "t_pf": 2.2e-6, "t_pf_span": 9.7e-8,
                         "erase_s": 2e-5, "truncate_s": 2e-4} | kw))


@dataclass
class Spec:
    tid: int
    steps: int = 3
    obs: list[int] = field(default_factory=list)       # tokens appended after each tool call
    delays: list[float] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    question: int = 9
    n_predict: int = 32
    out: int = OUT                                     # generated tokens per request
    timeout: float = 30.0
    profile: str = "syn"

    @property
    def task_id(self) -> str:
        return f"t{self.tid:03d}"

    def first_prompt(self) -> str:
        return (PREFIX if self.profile == "syn" else "S" * 4) + self.task_id + f"q{self.tid % 1000:03d}" * self.question


def make_specs(seed: int, n: int, steps: tuple[int, int] = (2, 5), obs: int = 680,
               delay: tuple[float, float] = (0.004, 0.03)) -> list[Spec]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        st = rng.randint(*steps)
        tools = ["lookup" if rng.random() < 0.2 else "search" for _ in range(st - 1)]
        out.append(Spec(i, st, [obs] * (st - 1),
                        [0.002 if t == "lookup" else rng.uniform(*delay) for t in tools], tools))
    return out


def make_script(specs: list[Spec]):
    outs = {s.task_id: s.out for s in specs}

    def script(prompt: str) -> str:
        tag = prompt[len(PREFIX):len(PREFIX) + 4] if prompt.startswith(PREFIX) else prompt[4:8]
        return "y" * (4 * outs.get(tag, OUT))
    return script


def priors_for(specs: list[Spec]) -> dict[str, Any]:
    """Exact growth quantiles of the synthetic tasks (max over tasks, per step)."""
    G: dict[int, int] = {}
    p0s = []
    for s in specs:
        p = len(s.first_prompt()) // 4
        p0s.append(p)
        sizes = [p]
        for n in range(s.steps):
            sizes.append(p + s.out - 1)
            p += s.out + (s.obs[n] if n < len(s.obs) else 0)
        peak = max(sizes)
        for n, sz in enumerate(sizes[:-1]):
            G[n] = max(G.get(n, 0), peak - sz)
    vals = [G[n] for n in sorted(G)]
    return {"syn": {"P0": max(p0s), "G": {"0.9": vals}, "duty": 0.5},
            "solo": {"P0": max(p0s), "G": {"0.9": vals}, "duty": 0.5}}


async def synth_run(llm: Any, events: EventLog, spec: Spec, slot: int, results: list) -> str:
    p = spec.first_prompt()
    for n in range(1, spec.steps + 1):
        events.emit("req_start", task_id=spec.task_id, step=n, kind="react", slot=slot, prompt_chars=len(p))
        res = await llm.generate(p, spec.n_predict, [], slot, False)
        events.emit("req_end", task_id=spec.task_id, step=n, kind="react", ok=res.ok, prompt_n=res.prompt_n,
                    cache_n=res.cache_n, predicted_n=res.predicted_n, slot=res.slot, error=res.error)
        results.append(res)
        if not res.ok:
            return "kv_infeasible" if (res.error or "").startswith("kv_infeasible") else "error"
        p += res.text
        if n == spec.steps:
            return "ok"
        d, tool = spec.delays[n - 1], spec.tools[n - 1]
        events.emit("tool_start", task_id=spec.task_id, step=n, tool=tool, delay_s=d)
        t = time.monotonic()
        await asyncio.sleep(d)
        o = spec.obs[n - 1]
        events.emit("tool_end", task_id=spec.task_id, step=n, tool=tool, obs_chars=4 * o,
                    idle_s=round(time.monotonic() - t, 4))
        p += f"o{spec.tid % 1000:03d}" * o
    return "ok"


@dataclass
class Run:
    pool: MockPoolLLM
    kvm: KVManager
    events: EventLog
    status: dict[str, str]
    results: dict[str, list]
    errors: list[str]

    def ev(self, typ: str) -> list[dict[str, Any]]:
        return [e for e in self.events.events if e["type"] == typ]


def ledger_check(kvm: KVManager, errors: list[str]):
    """I1 and I2 at every cell mutation of the mock: charge >= cells per slot, U <= c - m."""
    def check(p: MockPoolLLM) -> None:
        for k, sk in kvm.slots.items():
            if p.cells(k) > sk.charge:
                errors.append(f"slot {k} ({sk.state}): cells {p.cells(k)} > charge {sk.charge}")
        if kvm.cfg.gate == "full" and kvm.U > kvm.cap:
            errors.append(f"U {kvm.U} > cap {kvm.cap}")
        if kvm.cfg.gate == "active" and kvm.active_charge() > kvm.cap:
            errors.append(f"active {kvm.active_charge()} > cap {kvm.cap}")
        if kvm.host_bytes > kvm.host_cap:
            errors.append(f"host {kvm.host_bytes} > {kvm.host_cap}")
    return check


async def run_tasks(policy: str, specs: list[Spec], *, capacity: int, k: int, tmp: Path,
                    overrides: dict[str, Any] | None = None, mock: dict[str, Any] | None = None,
                    cost: CostModel | None = None, priors: dict[str, Any] | None = None, check: bool = True,
                    listeners: tuple = (), during: Any = None, setup: Any = None) -> Run:
    tmp.mkdir(parents=True, exist_ok=True)
    pool = MockPoolLLM(make_script(specs), **({"capacity": capacity, "n_slots": 16, "slot_dir": tmp} | GPU_FAST
                                              | (mock or {})))
    events = EventLog()
    cfg = load_policy(policy, {"ctx": capacity, "k": k, "slot_dir": str(tmp), "tick_s": 0.005, "omega": 4.0,
                               "reconcile_s": 0.02, "op_timeout_s": 2.0} | (overrides or {}))
    kvm = KVManager(pool, events, cfg, pool, profiles={}, priors=priors or priors_for(specs), families=FAMILIES,
                    cost=cost or fast_cost())
    llm = ManagedLLM(pool, kvm)
    events.listeners.append(kvm.on_event)
    events.listeners.extend(listeners)
    errors: list[str] = []
    if check:
        pool.check = ledger_check(kvm, errors)
    run = Run(pool, kvm, events, {}, {s.task_id: [] for s in specs}, errors)
    if setup is not None:
        setup(run)
    await kvm.start()
    q: asyncio.Queue = asyncio.Queue()
    for s in specs:
        q.put_nowait(s)
    for _ in range(k):
        q.put_nowait(None)
    tickets = itertools.count()

    async def worker(w: int) -> None:
        while (spec := await q.get()) is not None:
            slot = await kvm.admit(spec.task_id, spec.profile, next(tickets), worker=w)
            status = "error"
            try:
                status = await asyncio.wait_for(synth_run(llm, events, spec, slot, run.results[spec.task_id]),
                                                spec.timeout)
            except asyncio.TimeoutError:
                status = "timeout"
            finally:
                await asyncio.shield(kvm.release(spec.task_id, status))
            run.status[spec.task_id] = status

    jobs = [worker(w) for w in range(k)] + ([during(run)] if during else [])
    await asyncio.wait_for(asyncio.gather(*jobs), 60)
    await kvm.close()
    return run


def assert_clean(r: Run, tmp: Path, purges_ok: bool = False) -> None:
    assert set(r.status.values()) == {"ok"}, r.status
    assert r.errors == [] and r.pool.check_errors == [], (r.errors[:5], r.pool.check_errors[:5])
    assert r.pool.overflows == 0 and r.pool.http500 == 0
    if not purges_ok:
        assert r.pool.purges == 0
    assert not r.kvm.violations, (r.kvm.violations, r.ev("kv_violation")[:3])
    assert list(tmp.glob("*.bin")) == []          # no orphan slot file
    assert r.kvm._waiters == [] and r.kvm.n_not_started == 0 and r.kvm.agents == {}


# --- 3. the gate never overflows ------------------------------------------------------------

@pytest.mark.parametrize("policy,k", [("ours", 8), ("always_off", 8), ("managed_k", 6), ("adm_retain", 8)])
async def test_gate_never_overflows(policy, k, tmp_path):
    gave_up = 0
    for seed in range(20):
        specs = make_specs(seed, 12)
        r = await run_tasks(policy, specs, capacity=4096, k=k, tmp=tmp_path / f"s{seed}")
        assert_clean(r, tmp_path / f"s{seed}", purges_ok=r.kvm.cfg.gate == "active")
        assert r.kvm.max_U <= r.kvm.cap or r.kvm.cfg.gate == "active"
        gave_up += r.kvm.give_ups + r.pool.purges
    assert gave_up > 0                            # the pool really was under pressure


# --- 4. the ledger is exact -----------------------------------------------------------------

async def test_ledger_exact_and_reconcile_clean(tmp_path):
    mism: list[str] = []
    holder: dict[str, Any] = {}

    def exact(ev):   # after every response and op, an idle slot's s equals the mock's cells
        if ev["type"] == "kv_state" and ev["to"] in (IDLE, READY, STALE, DROPPED, OFFLOADED, FREE):
            cells = holder["pool"].cells(ev["slot"])
            if cells != ev["resident"]:
                mism.append(f"{ev}: cells {cells}")
    specs = make_specs(7, 12)
    r = await run_tasks("ours", specs, capacity=4096, k=8, tmp=tmp_path, listeners=(exact,),
                        overrides={"reconcile_s": 0.003}, setup=lambda run: holder.update(pool=run.pool))
    assert_clean(r, tmp_path)
    assert mism == []
    assert r.kvm.give_ups > 0 and r.ev("kv_recon") == []
    assert not any(e["kind"] == "tok_mismatch" for e in r.ev("kv_violation"))


# --- 5/6. no deadlock with READY agents; synchronized returns -------------------------------

async def test_no_deadlock_when_ready_agents_hold_the_memory(tmp_path):
    # 10 agents with large first contexts return together; each next request is ~half the pool.
    specs = [Spec(i, 2, [500], [0.05], ["search"], question=400) for i in range(10)]
    biggest = max(len(s.first_prompt()) // 4 + OUT + 500 + 32 for s in specs)
    r = await asyncio.wait_for(run_tasks("managed_k", specs, capacity=2 * biggest, k=10, tmp=tmp_path), 30)
    assert_clean(r, tmp_path)
    gates = r.ev("kv_gate")
    assert any(v for g in gates if g["prio"] == 0 for v in g["victims"])
    # 6. FIFO dispatch within each priority class, and bounded waits
    for prio in (0, 1):
        seqs = [g["seq"] for g in gates if g["prio"] == prio]
        assert seqs == sorted(seqs)
    assert max(g["wait_s"] for g in gates) < 5.0


async def test_synchronized_returns_dispatch_fifo(tmp_path):
    specs = [Spec(i, 3, [600, 600], [0.03, 0.03], ["search", "search"]) for i in range(8)]
    r = await run_tasks("ours", specs, capacity=4096, k=8, tmp=tmp_path)
    assert_clean(r, tmp_path)
    p0 = [g for g in r.ev("kv_gate") if g["prio"] == 0]
    assert [g["seq"] for g in p0] == sorted(g["seq"] for g in p0)
    assert r.kvm.summary()["gate_wait"]["p0"]["max_s"] < 2.0


# --- 7. priority-1 heads and the TTL --------------------------------------------------------

async def test_p1_head_waits_for_ttl_expiry(tmp_path):
    # A holds 1,640 cells in a 1 s tool; B's first request needs most of the pool. B is blocked
    # (pressure), so A's rent accrues and A expires at T* = c_drop(S) (no active tokens).
    a = Spec(0, 2, [10], [1.0], ["search"], question=1519)
    b = Spec(1, 1, [], [], [], question=1510)
    cost = fast_cost(t_pf=2e-4, t_pf_span=0.0, alpha=0.0)
    r = await run_tasks("ours", [a, b], capacity=3200, k=2, tmp=tmp_path, cost=cost,
                        overrides={"host_frac": 0, "tick_s": 0.02, "omega": 10.0})
    assert_clean(r, tmp_path)
    exp = [e for e in r.ev("kv_expire") if e["task_id"] == a.task_id]
    dec = next(e for e in r.ev("kv_decision") if e["task_id"] == a.task_id)
    assert len(exp) == 1
    assert dec["t_star"] == pytest.approx(2e-4 * (dec["S"] - 120), rel=1e-6)
    assert abs(exp[0]["pressure_s"] - dec["t_star"]) <= 2 * 0.02
    t_exp = exp[0]["t"]
    ops_a = [o for o in r.ev("kv_op") if o["owner"] == a.task_id and o["reason"] == "deficit"]
    assert ops_a and all(o["t"] >= t_exp for o in ops_a) and ops_a[0]["for_task"] == b.task_id
    assert ops_a[0]["prio"] == 1


async def test_nosteal_p1_head_never_evicts_an_idle_agent(tmp_path):
    a = Spec(0, 2, [10], [0.3], ["search"], question=1519)
    b = Spec(1, 1, [], [], [], question=1510)
    r = await run_tasks("nosteal", [a, b], capacity=3200, k=2, tmp=tmp_path, overrides={"omega": 10.0})
    assert_clean(r, tmp_path)
    assert not [o for o in r.ev("kv_op") if o["owner"] == a.task_id]
    gb = next(g for g in r.ev("kv_gate") if g["task_id"] == b.task_id)
    assert gb["wait_s"] >= 0.25                   # B waited for A to finish (STALE) instead


# --- 8. give-ups ----------------------------------------------------------------------------

def _evictions(r: Run) -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for o in r.ev("kv_op"):
        if o["owner"] and o["reason"] not in ("restore", "cancel", "resolve") and o["ok"]:
            out.setdefault((o["owner"], o["step"]), []).append(o)
    return out


async def test_at_most_one_give_up_per_step_and_erase_only_for_p0(tmp_path):
    n_drop_erase = 0
    for seed in range(6):
        r = await run_tasks("ours", make_specs(seed, 12), capacity=4096, k=8, tmp=tmp_path / str(seed),
                            overrides={"host_frac": 0})
        assert_clean(r, tmp_path / str(seed))
        ev = _evictions(r)
        for key, ops in ev.items():
            kinds = [o["op"] for o in ops]
            assert kinds in (["truncate"], ["erase"], ["truncate", "erase"], ["save", "erase"]), (key, kinds)
            if kinds == ["truncate", "erase"]:
                assert ops[1]["prio"] == 0
                n_drop_erase += 1
        for tid in {k[0] for k in ev}:
            assert r.kvm.task_stats(tid)["give_ups"] == len([k for k in ev if k[0] == tid])
        assert r.kvm.give_ups == len(ev)


async def test_wasted_save_does_not_use_up_an_eviction(tmp_path):
    # X's tool ends during its save (wasted: no give-up). The EWMA then makes H drop X (READY,
    # class 4), and X's prefix must still be erasable by H, a priority-0 head (spec §2.4, I5).
    cap = 4096 - 128
    x = Spec(1, 2, [10], [0.05], ["search"], question=1500, timeout=10.0)
    h = Spec(0, 2, [cap - 60 - 241], [0.005], ["search"], question=100, timeout=3.0)   # tok(P_H2) = cap - 60
    r = await run_tasks("managed_k", [h, x], capacity=4096, k=2, tmp=tmp_path, mock={"kappa_s": 2e-4}, check=False)
    assert r.status == {"t000": "ok", "t001": "ok"}
    ops = [(o["op"], bool(o.get("wasted"))) for o in r.ev("kv_op") if o["owner"] == x.task_id]
    assert ops == [("save", True), ("truncate", False), ("erase", False)]
    assert r.kvm.task_stats(x.task_id)["give_ups"] == 1 and r.kvm.wasted_saves == 1


async def test_stale_prefix_is_erased_before_a_live_tail(tmp_path):
    # Spec §4.2: STALE (class 1) goes before any live agent, also when it is already at its prefix.
    pool, kvm, _ = await _idle_manager(tmp_path)
    for prio in (0, 1):
        w = _Waiter(None, 0, prio, 0, 100, 10, 0.0)   # type: ignore[arg-type]
        stale = Victim(5, 1, 1400, 1400, owned=False)
        live = Victim(6, 2, 3000, 1400, residual=1.0, owner="A", erase_ok=prio == 0)
        v, action = kvm._next_victim(w, [live, stale])
        assert (v.slot, action) == (5, "erase")
        v, action = kvm._next_victim(w, [live, Victim(5, 1, 2000, 1400, owned=False)])
        assert (v.slot, action) == (5, "drop")         # a STALE tail is dropped first, as before
        dropped = Victim(6, 2, 1400, 1400, residual=1.0, owner="A", erase_ok=prio == 0)
        assert kvm._next_victim(w, [dropped, stale])[0].slot == 5   # a dead prefix before a live one
    await kvm.close()


async def test_no_give_up_with_ample_memory_and_same_cache_hits_as_observe(tmp_path):
    specs = make_specs(3, 12, delay=(0.01, 0.03))
    ours = await run_tasks("ours", specs, capacity=65536, k=4, tmp=tmp_path / "a")
    obs = await run_tasks("observe", specs, capacity=65536, k=4, tmp=tmp_path / "b")
    assert_clean(ours, tmp_path / "a")
    assert_clean(obs, tmp_path / "b")
    assert ours.kvm.give_ups == 0 and not ours.ev("kv_op")
    for tid in ours.results:
        assert [x.cache_n for x in ours.results[tid]] == [x.cache_n for x in obs.results[tid]]


# --- 10-12. offload, restore, wasted save, failed restore -----------------------------------

def _offload_scene(tmp_path: Path, **kw: Any):
    a = Spec(0, 2, [10], [0.3], ["search"], question=1519)     # offloaded while in its tool
    b = Spec(1, 1, [], [], [], question=1510)
    return [a, b], dict(capacity=3200, k=2, tmp=tmp_path, cost=fast_cost(alpha=0.0),
                        overrides={"omega": 10.0} | kw.pop("overrides", {}), **kw)


async def test_offload_then_restore_reuses_the_whole_context(tmp_path):
    specs, kw = _offload_scene(tmp_path)
    r = await run_tasks("lazy", specs, **kw)
    assert_clean(r, tmp_path)
    saves = [o for o in r.ev("kv_op") if o["op"] == "save" and o["owner"] == "t000"]
    assert len(saves) == 1 and saves[0]["ok"]
    restore = [o for o in r.ev("kv_op") if o["op"] == "restore"]
    assert len(restore) == 1 and restore[0]["ok"] and restore[0]["tokens"] == saves[0]["tokens"]
    first, second = r.results["t000"]
    assert second.cache_n == first.kv_tokens == saves[0]["tokens"]   # the whole pre-offload context
    assert r.kvm.restores == 1 and r.kvm.host_bytes == 0 and r.kvm.host_peak > 0


async def test_full_host_tier_forces_drop(tmp_path):
    specs, kw = _offload_scene(tmp_path, overrides={"host_frac": 0.3})   # < one 1.6k image at c = 3.2k
    r = await run_tasks("lazy", specs, **kw)
    assert_clean(r, tmp_path)
    ops = [o["op"] for o in r.ev("kv_op") if o["owner"] == "t000"]
    assert ops == ["truncate"] and r.kvm.host_peak == 0
    assert r.results["t000"][1].cache_n == 120                          # the prefix was kept


async def test_tool_ending_during_save_skips_the_erase(tmp_path):
    a = Spec(0, 2, [10], [0.03], ["search"], question=1519)
    r = await run_tasks("always_off", [a], capacity=8192, k=1, tmp=tmp_path, mock={"kappa_s": 2e-4})  # 0.3 s save
    assert_clean(r, tmp_path)
    ops = [o for o in r.ev("kv_op") if o["owner"] == a.task_id]
    assert [o["op"] for o in ops] == ["save"] and ops[0]["wasted"]
    first, second = r.results[a.task_id]
    assert second.cache_n == first.kv_tokens                           # the slot stayed intact
    assert r.kvm.wasted_saves == 1 and r.kvm.host_bytes == 0


async def test_failed_restore_becomes_a_reprefill(tmp_path):
    holder: dict[str, Any] = {}

    def lose_file(ev):   # the image disappears while its owner is still in the tool
        if ev["type"] == "kv_op" and ev["op"] == "erase" and ev["reason"] == "deficit" and ev["ok"]:
            (tmp_path / f"slot_{ev['slot']}.bin").unlink()
    specs, kw = _offload_scene(tmp_path, listeners=(lose_file,))
    r = await run_tasks("lazy", specs, **kw)
    assert_clean(r, tmp_path)
    restore = [o for o in r.ev("kv_op") if o["op"] == "restore"]
    assert len(restore) == 1 and not restore[0]["ok"]
    assert r.results["t000"][1].cache_n == 0 and r.status["t000"] == "ok"
    assert r.kvm.restore_failed == 1 and r.kvm.host_bytes == 0


# --- 13. cancellation -----------------------------------------------------------------------

async def test_cancel_mid_gate_leaves_no_queue_entry(tmp_path):
    a = Spec(0, 2, [10], [0.4], ["search"], question=1519)
    b = Spec(1, 1, [], [], [], question=1510, timeout=0.15)          # blocked at the gate, then cancelled
    c = Spec(2, 1, [], [], [], question=50)
    r = await run_tasks("nosteal", [a, b, c], capacity=3200, k=2, tmp=tmp_path, overrides={"omega": 10.0})
    assert r.status == {"t000": "ok", "t001": "timeout", "t002": "ok"}
    assert r.errors == [] and r.pool.overflows == 0 and r.pool.purges == 0 and not r.kvm.violations
    assert r.kvm._waiters == [] and r.kvm.n_not_started == 0 and list(tmp_path.glob("*.bin")) == []
    assert not [g for g in r.ev("kv_gate") if g["task_id"] == b.task_id]


async def test_cancel_mid_request_erases_the_slot(tmp_path):
    a = Spec(0, 1, [], [], [], question=100, n_predict=400, out=300, timeout=0.1)   # 3 s decode
    r = await run_tasks("ours", [a], capacity=4096, k=1, tmp=tmp_path, mock={"t_decode": 0.01})
    assert r.status == {"t000": "timeout"} and r.errors == []
    erase = [o for o in r.ev("kv_op") if o["reason"] == "cancel"]
    assert len(erase) == 1 and erase[0]["ok"] and erase[0]["tokens"] > 100
    # the cancelled request had written cells the ledger's s never saw; they are within its
    # upper-bound charge, so no undercount (it was one before the UNKNOWN case of _check_op_tokens)
    assert not r.kvm.violations
    sk = r.kvm.slots[0]
    assert sk.state == FREE and sk.charge == 0 == r.pool.cells(0)


async def test_cancel_mid_op_lets_the_victims_op_finish(tmp_path):
    # V (in a tool) is offloaded for H's request; H is cancelled during V's 0.4 s save.
    v = Spec(0, 2, [10], [0.8], ["search"], question=1519)
    h = Spec(1, 2, [10], [0.01], ["search"], question=1510, timeout=0.25)
    seen: dict[str, Any] = {}

    async def during(run: Run) -> None:
        while not any(o["op"] == "erase" and o["owner"] == v.task_id for o in run.ev("kv_op")):
            await asyncio.sleep(0.01)
        sk = run.kvm.slots[0]
        seen.update(state=sk.state, charge=sk.charge, cells=run.pool.cells(0))
    r = await run_tasks("lazy", [v, h], capacity=3200, k=2, tmp=tmp_path, cost=fast_cost(alpha=0.0),
                        mock={"kappa_s": 2.5e-4}, overrides={"omega": 10.0}, during=during)
    assert r.status == {"t000": "ok", "t001": "timeout"}
    assert r.errors == [] and r.pool.overflows == 0 and r.pool.purges == 0 and not r.kvm.violations
    assert seen == {"state": OFFLOADED, "charge": 0, "cells": 0}      # the op completed after H left
    assert [o["op"] for o in r.ev("kv_op") if o["owner"] == v.task_id][:2] == ["save", "erase"]
    assert r.kvm.restores == 1 and r.results["t000"][1].cache_n == r.results["t000"][0].kv_tokens
    assert r.kvm._waiters == [] and list(tmp_path.glob("*.bin")) == []


# --- 14. kv_infeasible and kv_truncated -----------------------------------------------------

async def _run_agents(policy: str, tasks: list[tuple[str, Profile]], corpus, item, *, capacity: int,
                      k: int = 1, tmp: Path, overrides: dict[str, Any] | None = None, mock: dict | None = None):
    profiles = {p.name: p for _, p in tasks}
    pool = MockPoolLLM(mock_script, **({"capacity": capacity, "n_slots": 8, "slot_dir": tmp, "t_prefill": 0.0,
                                        "t_decode": 0.0, "op_fixed_s": 0.0} | (mock or {})))
    events = EventLog()
    cfg = load_policy(policy, {"ctx": capacity, "k": k, "slot_dir": str(tmp)} | (overrides or {}))
    kvm = KVManager(pool, events, cfg, pool, profiles=profiles, priors={})
    events.listeners.append(kvm.on_event)
    agent = Agent(ManagedLLM(pool, kvm), corpus, events)
    await kvm.start()
    traces = {}
    q: asyncio.Queue = asyncio.Queue()
    for t in tasks:
        q.put_nowait(t)
    for _ in range(k):
        q.put_nowait(None)
    tickets = itertools.count()

    async def worker(w: int) -> None:
        while (t := await q.get()) is not None:
            tid, prof = t
            slot = await kvm.admit(tid, prof.name, next(tickets), worker=w)
            trace = None
            try:
                trace = await agent.run(tid, item, prof, slot)
            finally:
                await asyncio.shield(kvm.release(tid, trace.status if trace else "error"))
            trace.status = kv_status(trace.status, trace.error)
            traces[tid] = trace
    await asyncio.gather(*(worker(w) for w in range(k)))
    await kvm.close()
    return pool, kvm, events, traces


async def test_kv_infeasible_fails_the_task_without_blocking_the_queue(corpus, item, tmp_path):
    big, small = Profile(name="big", few_shot=6), Profile(name="small", few_shot=0)
    pool, kvm, events, traces = await _run_agents(
        "ours", [("a", big), ("b", small), ("c", big), ("d", small)], corpus, item, capacity=1200, k=2, tmp=tmp_path)
    assert {t: tr.status for t, tr in traces.items()} == {"a": "kv_infeasible", "b": "ok", "c": "kv_infeasible",
                                                          "d": "ok"}
    assert traces["a"].error.startswith("kv_infeasible") and sorted(kvm.infeasible) == ["a", "c"]
    assert not any("Here are some examples" in c["prompt"] for c in pool.calls)   # never sent
    assert pool.overflows == 0 and pool.purges == 0 and not kvm.violations
    assert kvm.summary()["kv_exceeded"] == ["a", "c"]               # the same set every configuration lists
    # observe (B1): nothing refused and the same requests sent, but the same tasks listed
    tasks = [("a", big), ("b", small), ("c", big), ("d", small)]
    p2, k2, ev2, _ = await _run_agents("observe", tasks, corpus, item, capacity=1200, k=2, tmp=tmp_path / "o")
    p3, _, _, _ = await _run_agents("observe", tasks, corpus, item, capacity=1200, k=2, tmp=tmp_path / "n",
                                    overrides={"lifecycle": "observe"})
    assert k2.summary()["kv_exceeded"] == ["a", "c"] and k2.infeasible == []
    assert sorted(e["task_id"] for e in ev2.events if e["type"] == "kv_exceeded") == ["a", "c"]
    assert k2.task_stats("a")["exceeded"] and not k2.task_stats("b")["exceeded"]
    assert [c["prompt"] for c in p2.calls] == [c["prompt"] for c in p3.calls]


async def test_short_tail_room_sends_n_predict_eff(corpus, item, tmp_path):
    prof = Profile(name="small", few_shot=0, n_predict=128)
    probe = MockPoolLLM(mock_script)
    first = await Agent(probe, corpus, EventLog()).run("x", item, prof, 0)
    tok0 = first.steps[0].llm["prompt_n"] + first.steps[0].llm["cache_n"]
    pool, kvm, events, traces = await _run_agents("ours", [("a", prof)], corpus, item, capacity=tok0 + 128 + 50,
                                                  tmp=tmp_path)
    tr = [e for e in events.events if e["type"] == "kv_truncated"]
    assert tr and tr[0]["n_predict"] == 128 and tr[0]["n_predict_eff"] == 50 and tr[0]["tok_P"] == tok0
    assert pool.calls[0]["n_predict"] == 50 and kvm.truncated >= 1
    assert traces["a"].status in ("ok", "kv_infeasible") and pool.overflows == 0


# --- 15. observe reproduces Phase 1 ---------------------------------------------------------

async def test_observe_reproduces_the_phase1_request_sequence(corpus, item, tmp_path):
    prof = Profile(name="react_short", few_shot=2)
    tasks = [(f"t{i}", prof) for i in range(5)]

    async def plain() -> tuple[MockPoolLLM, dict]:
        pool = MockPoolLLM(mock_script, n_slots=8, t_prefill=1e-5, t_decode=1e-4)
        agent = Agent(pool, corpus, EventLog())
        q: asyncio.Queue = asyncio.Queue()
        for t in tasks:
            q.put_nowait(t)
        for _ in range(2):
            q.put_nowait(None)
        traces = {}

        async def worker(w: int) -> None:
            while (t := await q.get()) is not None:
                traces[t[0]] = await agent.run(t[0], item, t[1], w)
        await asyncio.gather(worker(0), worker(1))
        return pool, traces
    p1, t1 = await plain()
    p2, kvm, events, t2 = await _run_agents("observe", tasks, corpus, item, capacity=None, k=2, tmp=tmp_path,
                                            overrides={"ctx": 65536}, mock={"t_prefill": 1e-5, "t_decode": 1e-4})
    key = lambda c: (c["slot"], c["prompt"], c["n_predict"], c["ids"])  # noqa: E731
    for slot in (0, 1):
        assert [key(c) for c in p1.calls if c["slot"] == slot] == [key(c) for c in p2.calls if c["slot"] == slot]
    assert sorted(map(key, p1.calls)) == sorted(map(key, p2.calls))
    def trace_key(tr):   # everything but wall-clock timings
        return (tr.status, tr.slot, tr.prediction, tr.segments, sorted(tr.sf_seen),
                [(s.kind, s.output, s.action, s.arg, s.llm["prompt_n"], s.llm["cache_n"], s.llm["predicted_n"],
                  s.llm["slot"]) for s in tr.steps])
    assert t1.keys() == t2.keys()
    for tid in t1:
        assert trace_key(t1[tid]) == trace_key(t2[tid])
    assert not kvm.violations and kvm.give_ups == 0 and not [e for e in events.events if e["type"] == "kv_op"]


# --- 16. active gate: purges are the lifecycle ----------------------------------------------

async def test_active_gate_purges_are_states_not_violations(tmp_path):
    seen = 0
    for seed in range(4):
        r = await run_tasks("gated_k", make_specs(seed, 12, delay=(0.02, 0.05)), capacity=4096, k=6,
                            tmp=tmp_path / str(seed), overrides={"reconcile_s": 0.004})
        assert_clean(r, tmp_path / str(seed), purges_ok=True)
        assert r.pool.purges > 0 and not r.ev("kv_violation")
        seen += len([e for e in r.ev("kv_state") if e["reason"] == "server_purge"])
    assert seen > 0


async def test_active_gate_counts_a_reused_slots_old_tail(tmp_path):
    # A new task admitted into a STALE slot that holds a long foreign context: until the server
    # batches the new prompt the old tail stays, so both the dispatch check and the in-flight
    # charge must use max(s, f), not the request's f alone.
    pool = MockPoolLLM(lambda p: "", capacity=4096, n_slots=8)
    kvm = KVManager(pool, EventLog(), load_policy("adm_retain", {"ctx": 4096}), pool, profiles={}, priors={},
                    families=FAMILIES)
    await kvm.start()
    from edgeserve.kvm import AgentKV
    other, stale = kvm.slots[0], kvm.slots[1]
    other.state, other.s, other.f, other.charge = ACTIVE, 1000, 1000, 1000
    stale.state, stale.s, stale.charge = STALE, 3000, 3000
    w = _Waiter(AgentKV("t1", "syn", 1, 1), 1, 1, 0, 500, 1, 0.0)
    assert kvm._deficit(w) == 1000 + 3000 - kvm.cap > 0      # the old formula gave 1000 + 500 - cap
    stale.state, stale.f, stale.charge = ACTIVE, 500, 3000   # dispatched anyway: counted at 3000
    assert kvm.active_charge(exclude=other) == 3000
    await kvm.close()


# --- 18. wake-ups ---------------------------------------------------------------------------

async def test_kick_wakes_only_the_head(tmp_path):
    pool = MockPoolLLM(lambda p: "", capacity=4096, n_slots=8)
    kvm = KVManager(pool, EventLog(), load_policy("ours", {"ctx": 4096}), pool, profiles={}, priors={},
                    families=FAMILIES)
    await kvm.start()
    loop = asyncio.get_running_loop()
    from edgeserve.kvm import AgentKV
    ws = []
    for i, prio in enumerate((1, 0, 0, 1)):
        w = _Waiter(AgentKV(f"t{i}", "syn", i, i), i, prio, i, 10, 1, 0.0, fut=loop.create_future())
        kvm._waiters.append(w)
        ws.append(w)
    for _ in range(5):
        kvm._kick()   # sync: no RuntimeError outside the lock
    assert [w.fut.done() for w in ws] == [False, True, False, False]  # head = priority 0, earliest
    kvm._committed = ws[0]
    ws[0].fut = loop.create_future()
    kvm._kick()
    assert ws[0].fut.done() and [w.wakeups for w in ws] == [1, 1, 0, 0]
    await kvm.close()


async def test_sync_listener_wakes_a_blocked_head_and_listener_errors_are_contained(tmp_path):
    def boom(ev):
        if ev["type"] == "tool_start":
            raise RuntimeError("listener bug")
    specs, kw = _offload_scene(tmp_path, listeners=(boom,))
    r = await run_tasks("lazy", specs, **kw)
    assert_clean(r, tmp_path)
    errs = r.ev("kv_violation")                  # written by EventLog; summary() counts them
    assert r.status == {"t000": "ok", "t001": "ok"}
    assert errs and all(e["kind"] == "listener_error" and "listener bug" in e["detail"] for e in errs)
    gb = next(g for g in r.ev("kv_gate") if g["task_id"] == "t001")
    assert gb["victims"]                          # woken by A's tool_start (lazy: eligible at once)


async def test_manager_bug_in_listener_does_not_fail_the_agent(tmp_path, monkeypatch):
    specs = make_specs(1, 3)

    def broken(self, a):
        raise ValueError("decide bug")
    monkeypatch.setattr(KVManager, "_decide", broken)
    r = await run_tasks("ours", specs, capacity=65536, k=3, tmp=tmp_path)
    assert set(r.status.values()) == {"ok"}
    errs = [e for e in r.events.events if e.get("kind") == "listener_error"]
    assert errs and r.kvm.summary()["violations"].get("listener_error") == len(errs)   # summary.json sees them


async def test_background_loop_error_is_a_violation_and_the_loop_goes_on(tmp_path, monkeypatch):
    # The TTL scene of test 7 with one _tick raising: expiry must still happen, and the run is a
    # violation run instead of silently behaving like nosteal.
    calls = {"n": 0}
    orig = KVManager._tick

    def flaky(self, now, dt):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyError("tick bug")
        return orig(self, now, dt)
    monkeypatch.setattr(KVManager, "_tick", flaky)
    a = Spec(0, 2, [10], [1.0], ["search"], question=1519)
    b = Spec(1, 1, [], [], [], question=1510)
    r = await run_tasks("ours", [a, b], capacity=3200, k=2, tmp=tmp_path, cost=fast_cost(t_pf=2e-4, t_pf_span=0.0,
                        alpha=0.0), overrides={"host_frac": 0, "tick_s": 0.02, "omega": 10.0})
    assert set(r.status.values()) == {"ok"}
    assert r.kvm.violations == {"loop_error": 1} and r.kvm.summary()["loop_errors"] == {"tick:KeyError": 1}
    assert [e for e in r.ev("kv_expire") if e["task_id"] == a.task_id]


async def test_failing_side_task_is_recorded_and_not_respawned(tmp_path, monkeypatch):
    async def broken(self, a, reason):
        a.proactive = True
        try:
            raise RuntimeError("proactive bug")
        except Exception:
            a.side_err_step = a.step
            raise
        finally:
            a.proactive = False
    monkeypatch.setattr(KVManager, "_proactive", broken)
    specs = make_specs(1, 3, delay=(0.05, 0.1))
    r = await run_tasks("ttl_off", specs, capacity=65536, k=3, tmp=tmp_path, overrides={"ttl_x": 0.01})
    assert set(r.status.values()) == {"ok"}
    n = r.kvm.loop_errors["side:RuntimeError"]
    assert r.kvm.violations["loop_error"] == 1 and 0 < n <= sum(s.steps - 1 for s in specs)


# --- 19. one victim at a time ---------------------------------------------------------------

async def test_one_victim_at_a_time(tmp_path):
    # H returns needing D = 830 cells. V1 (650) is the top victim; R's request in flight frees
    # its unused n_predict reservation (381) while V1's save hangs, so V2 is never touched.
    # (20 cells of slack let H's first request in while V2's is still in flight.)
    v1 = Spec(0, 2, [10], [1.5], ["search"], question=529, out=1)
    v2 = Spec(1, 2, [10], [1.5], ["lookup"], question=479, out=1)    # ranked after any search
    rr = Spec(2, 1, [], [], [], question=179, n_predict=400, out=20)
    h = Spec(3, 2, [867], [0.05], ["search"], question=29, out=1)
    r = await run_tasks("ours", [v1, v2, rr, h], capacity=2170 + 128, k=4, tmp=tmp_path, cost=fast_cost(alpha=0.0),
                        mock={"t_decode": 0.02, "op_hang": {"save": 0.6}}, overrides={"omega": 10.0})
    assert_clean(r, tmp_path)
    gh = [g for g in r.ev("kv_gate") if g["task_id"] == h.task_id]
    assert gh[1]["deficit"] == 830 and [v[0] for v in gh[1]["victims"]] == [0]
    assert not [o for o in r.ev("kv_op") if o["owner"] == v2.task_id and o["reason"] == "deficit"]


# --- 20. reconcile race ---------------------------------------------------------------------

async def _idle_manager(tmp_path: Path, **mock: Any) -> tuple[MockPoolLLM, KVManager, EventLog]:
    pool = MockPoolLLM(lambda p: "y" * 40, **({"capacity": 4096, "n_slots": 4, "slot_dir": tmp_path,
                                             "t_prefill": 0.0, "t_decode": 0.05} | mock))
    events = EventLog()
    kvm = KVManager(pool, events, load_policy("ours", {"ctx": 4096, "reconcile_s": 3600, "slot_dir": str(tmp_path)}),
                    pool, profiles={}, priors={}, families=FAMILIES, cost=fast_cost())
    await kvm.start()
    return pool, kvm, events


async def test_reconcile_skips_slots_changed_since_the_poll(tmp_path):
    pool, kvm, events = await _idle_manager(tmp_path, slots_stale=True)
    llm = ManagedLLM(pool, kvm)
    s1 = await kvm.admit("x", "syn", 0)
    await llm.generate(PREFIX + "x000" * 100, 1, [], s1)
    await kvm.release("x", "ok")                                   # slot s1: STALE with 220 cells
    s0 = await kvm.admit("y", "solo", 1)                             # no family: a FREE slot
    long = asyncio.create_task(llm.generate("Z000" * 50, 6, [], s0))   # decodes hold the main loop
    await asyncio.sleep(0.03)
    sk = kvm.slots[s1]
    sk.busy, sk.op = True, "erase"
    erase = kvm._run_op(sk, kvm._op_erase(sk, None, "stale", None))
    await asyncio.sleep(0.002)                                     # the erase waits for the loop
    await kvm._reconcile_once()                                    # answered from the pre-erase state
    await erase
    await long
    await kvm._reconcile_once()
    bad = [e for e in events.events if e["type"] in ("kv_violation", "kv_recon")
           or (e["type"] == "kv_state" and e["to"] == DROPPED)]
    assert bad == [] and sk.state == FREE and pool.cells(s1) == 0
    await kvm.close()


async def test_repeated_clean_mismatch_raises_undercount(tmp_path):
    pool, kvm, events = await _idle_manager(tmp_path)
    llm = ManagedLLM(pool, kvm)
    s = await kvm.admit("x", "syn", 0)
    await llm.generate(PREFIX + "x000" * 100, 1, [], s)
    kvm.slots[s].s -= 10                                           # simulate an undercounting ledger
    await kvm._reconcile_once()
    rec = [e for e in events.events if e["type"] == "kv_recon"]
    assert [e["action"] for e in rec] == ["raise"] and not kvm.violations
    assert kvm.slots[s].charge == pool.cells(s)                    # raised to the server's count
    await kvm._reconcile_once()
    assert kvm.violations == {"undercount": 1}
    await kvm.release("x", "ok")
    await kvm.close()


# --- 21. op timeout -------------------------------------------------------------------------

async def test_op_timeout_leaves_unknown_until_resolved(tmp_path):
    specs, kw = _offload_scene(tmp_path, mock={"op_timeout_s": 0.1, "op_hang": {"save": 0.3}},
                               overrides={"op_timeout_s": 0.1})
    unknown: list[tuple[int, int]] = []
    holder: dict[str, Any] = {}

    def watch(ev):
        if ev["type"] == "kv_state" and ev["to"] == UNKNOWN:
            unknown.append((ev["charge"], holder["pool"].cells(ev["slot"])))
    r = await run_tasks("lazy", specs, listeners=(watch,), setup=lambda run: holder.update(pool=run.pool), **kw)
    assert set(r.status.values()) == {"ok"} and r.errors == [] and r.pool.overflows == 0 and r.pool.purges == 0
    assert unknown and all(charge >= cells > 0 for charge, cells in unknown)
    assert r.kvm.violations.get("op_timeout", 0) >= 1 and set(r.kvm.violations) == {"op_timeout"}
    assert r.kvm.resolved >= 1 and r.kvm.restores == 1               # the late save became a valid image
    assert list(tmp_path.glob("*.bin")) == [] and r.kvm.host_bytes == 0


async def test_unreachable_server_resolve_gives_up_and_release_returns(tmp_path, monkeypatch):
    # The server is gone: every erase is a transport error (status 0) and /slots raises. The
    # resolver must stop after RESOLVE_TRIES, leave the slot UNKNOWN at its bound, and let
    # release() finish, so the run ends like a Phase 1 run with llm_error tasks.
    monkeypatch.setattr(kvm_mod, "RESOLVE_BACKOFF_S", 0.001)
    pool, kvm, events = await _idle_manager(tmp_path)
    llm = ManagedLLM(pool, kvm)
    s = await kvm.admit("x", "syn", 0)
    await llm.generate(PREFIX + "x000" * 100, 1, [], s)
    sk = kvm.slots[s]
    sk.req_out = True
    kvm.on_cancel("x")                                             # a request the client gave up on
    assert sk.state == UNKNOWN and not sk.busy
    tries = {"n": 0}

    async def dead_erase(slot):
        tries["n"] += 1
        return SlotOpResult(ok=False, op="erase", slot=slot, error="ConnectError: refused", status=0)

    async def dead_slots():
        raise ConnectionError("refused")
    monkeypatch.setattr(pool, "slot_erase", dead_erase)
    monkeypatch.setattr(pool, "slots", dead_slots)
    await asyncio.wait_for(kvm.release("x", "llm_error"), 5)
    assert tries["n"] == 1 + kvm_mod.RESOLVE_TRIES                 # release's erase, then the resolver's
    assert kvm.violations["resolve_failed"] == 1
    assert sk.state == UNKNOWN and not sk.busy and sk.op_task is None and sk.charge >= pool.cells(s)
    assert len([e for e in events.events if e["type"] == "kv_op" and e["reason"] == "resolve"]) == 1
    await kvm.close()


# --- 22. HTTP 500 in observe mode -----------------------------------------------------------

async def test_http500_in_observe_zeroes_only_the_failed_slots(tmp_path):
    a = Spec(0, 1, [], [], [], question=100, n_predict=60, out=50)
    b = Spec(1, 1, [], [], [], question=100, n_predict=60, out=50)
    c = Spec(2, 2, [10], [1.0], ["search"], question=300, out=1)
    seen: dict[str, Any] = {}

    async def during(run: Run) -> None:
        await asyncio.sleep(0.15)
        seen["failed"] = sorted(run.pool.inject_500())
        await asyncio.sleep(0)
        seen["s"] = {k: run.kvm.slots[k].s for k in range(3)}
        seen["cells"] = {k: run.pool.cells(k) for k in range(3)}
        seen["c_state"] = run.kvm.slots[0].state
    r = await run_tasks("observe", [c, a, b], capacity=8192, k=3, tmp=tmp_path, mock={"t_decode": 0.01},
                        during=during)
    assert seen["failed"] == [1, 2]
    assert seen["s"][1] == seen["s"][2] == 0 and seen["s"][0] == seen["cells"][0] > 0
    assert seen["c_state"] == IDLE and r.status["t002"] == "ok"
    assert r.status["t000"] == r.status["t001"] == "error" and r.kvm.overflows == 2 and not r.kvm.violations


# --- every preset, under pressure -----------------------------------------------------------

PRESETS = [p for p in __import__("yaml").safe_load(open("configs/kvm.yaml", encoding="utf-8"))["presets"]
           if p != "observe"]


@pytest.mark.parametrize("policy", PRESETS)
async def test_every_preset_runs_clean_under_pressure(policy, tmp_path):
    fixed = load_policy(policy).admission == "fixed"
    for seed in range(3):
        specs = make_specs(seed, 10)
        r = await run_tasks(policy, specs, capacity=4096, k=5 if fixed else 8, tmp=tmp_path / str(seed),
                            overrides={"prior_delay_scale": 0.02, "theta": 2.5})   # theta: adm_duty only
        assert_clean(r, tmp_path / str(seed), purges_ok=r.kvm.cfg.gate == "active")
        if r.kvm.cfg.gate == "full":
            assert r.kvm.give_ups + r.kvm.evictions.get("drop", 0) > 0


async def test_prefetch_never_jumps_the_gate_queue(tmp_path, monkeypatch):
    # clairvoyant_pf restores early only while no priority-0 waiter or committed head is queued.
    bad: list[str] = []
    orig = KVManager._set

    def watch(self, sk, state=None, **kw):
        if state == RESTORING and kw.get("reason") == "prefetch":
            c = self._committed
            if (c is not None and c in self._waiters) or any(w.prio == 0 for w in self._waiters):
                bad.append(f"slot {sk.slot}")
        return orig(self, sk, state, **kw)
    monkeypatch.setattr(KVManager, "_set", watch)
    for seed in range(4):
        r = await run_tasks("clairvoyant_pf", make_specs(seed, 12, delay=(0.05, 0.3)), capacity=4096, k=8,
                            tmp=tmp_path / str(seed))
        assert_clean(r, tmp_path / str(seed))
    assert bad == []


async def test_observe_unpinned_tracks_host_cached_slots(tmp_path):
    # B2: the server picks slots; with --cache-ram every launch saves and clears the idle slots.
    specs = make_specs(2, 8)
    limit = 256 * 2**20        # a few synthetic contexts: the FIFO eviction is exercised
    seen: list[tuple[int, int, int]] = []

    def host_check(run: Run):
        def check(p: MockPoolLLM) -> None:   # the ledger's model of llama.cpp's host prompt cache
            shown = sum(sk.host_tok for sk in run.kvm.slots.values() if sk.state == HOST_CACHED)
            seen.append((run.kvm.host_bytes, shown * BYTES_PER_TOKEN, p.host_tokens * BYTES_PER_TOKEN))
        run.pool.check = check
    r = await run_tasks("observe", specs, capacity=8192, k=4, tmp=tmp_path,
                        overrides={"pinned": False, "cache_ram_mib": 256.0},
                        mock={"cache_ram_mib": 256.0, "return_progress": True}, check=False, setup=host_check)
    assert set(r.status.values()) == {"ok"} and not r.kvm.violations
    cached = [e for e in r.ev("kv_state") if e["to"] == "HOST_CACHED"]
    assert cached and r.pool.idle_clears > 0 and r.kvm.host_peak > 0
    assert r.pool.host_evictions > 0 and r.kvm.host_evictions > 0
    assert seen and all(led <= limit and shown <= led for led, shown, _ in seen)   # never above --cache-ram
    assert r.kvm.host_peak <= limit
    diffs = [abs(led - real) for led, _, real in seen]                  # tracks the mock's real cache
    assert sum(diffs) / len(diffs) < 0.05 * limit and diffs[-1] < 0.1 * limit
    assert r.kvm._unassigned == {} and all(sk.owner is None for sk in r.kvm.slots.values())
    plain = await run_tasks("observe", specs, capacity=65536, k=4, tmp=tmp_path / "p", overrides={"pinned": False},
                            check=False)
    assert set(plain.status.values()) == {"ok"} and not [e for e in plain.ev("kv_state") if e["to"] == "HOST_CACHED"]
