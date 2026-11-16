"""Admission and the slot pool (docs/phase2_spec.md §2.3, §5, test 9)."""
import asyncio
from dataclasses import dataclass

import pytest

from edgeserve.admission import DutyAdmission, FixedK, KVAdmission, SlotPool
from edgeserve.events import EventLog
from edgeserve.kvm import KVManager
from edgeserve.mockpool import MockPoolLLM
from edgeserve.policies import load_policy
from tests.test_kvm import Spec, assert_clean, make_specs, run_tasks


@dataclass
class A:
    R_hat: int
    profile: str = "p"


def test_fixed_k_counts_agents_and_the_gate_queue():
    adm = FixedK(k=2, cap=1000)
    assert adm.can_admit([], 10, waiters=0)
    assert not adm.can_admit([A(1), A(1)], 10, waiters=0) and adm.blocked == "kmax"
    assert not adm.can_admit([A(1)], 10, waiters=1) and adm.blocked == "gate"
    assert not adm.can_admit([A(1)], 10, waiters=0, not_started=1) and adm.blocked == "gate"
    assert FixedK(k=2, cap=1000, gated=False).can_admit([A(1)], 10, waiters=3, not_started=1)   # observe


def test_kv_admission_projects_peaks_with_overcommit():
    adm = KVAdmission(k=8, cap=1000, omega=1.0)
    assert adm.limit() == 2000
    assert adm.can_admit([A(900), A(900)], 200, waiters=0)
    assert not adm.can_admit([A(900), A(900)], 201, waiters=0) and adm.blocked == "kv"
    assert adm.can_admit([], 5000, waiters=0)          # nothing admitted: (c) is waived, the gate keeps it safe


def test_duty_admission_weights_peaks_by_duty_cycle():
    duty = {"short": 0.25, "long": 1.0}
    adm = DutyAdmission(k=8, cap=1000, theta=0.8, duty=duty.__getitem__)
    assert adm.limit() == 800
    assert adm.can_admit([A(1000, "short"), A(1200, "short")], 250, waiters=0)   # 250 + 300 + 250
    assert not adm.can_admit([A(400, "long")], 401, waiters=0)


def test_slot_pool_prefers_same_family_stale_then_free_then_any():
    pool = SlotPool(range(4))
    sizes = {0: 0, 1: 0, 2: 0, 3: 0}
    assert [pool.take("f") for _ in range(4)] == [0, 1, 2, 3]
    with pytest.raises(RuntimeError):
        pool.take("f")
    pool.give(2, stale=True, family="g")
    pool.give(3, stale=True, family="f")
    pool.give(1, stale=True, family="f")
    pool.give(0, stale=False)
    sizes.update({1: 900, 3: 400})
    assert pool.take("f", size=sizes.__getitem__) == 3     # same family, least to free
    assert pool.take("f", size=sizes.__getitem__) == 1
    assert pool.take("f") == 0                             # then a FREE slot
    assert pool.take("f") == 2                             # then any STALE one
    pool.give(2, stale=True, family="g")
    pool.give(1, stale=True, family="g")
    assert pool.take("g", busy=lambda k: k == 1) == 2      # avoid a slot with an op in flight
    pool.erased(1)
    assert 1 in pool.free and 1 not in pool.stale


async def test_admission_is_fifo_and_waits_for_first_acquire(tmp_path):
    # 9. A 50 ms /tokenize must not let a burst through: each admission waits until the previous
    # task's first request has entered (and left) the gate queue.
    specs = make_specs(4, 10, delay=(0.01, 0.03))
    r = await run_tasks("ours", specs, capacity=4096, k=6, tmp=tmp_path, mock={"tokenize_delay_s": 0.05})
    assert_clean(r, tmp_path)
    adm = r.ev("kv_admit")
    assert [e["ticket"] for e in adm] == list(range(10))
    assert all(e["gate_q"] == 0 and e["not_started"] == 0 for e in adm)
    assert max(e["n_agents"] for e in adm) <= 6 and r.kvm.max_agents <= 6
    gaps = [b["t"] - a["t"] for a, b in zip(adm, adm[1:])]
    assert min(gaps) >= 0.045


async def test_head_of_line_blocking_has_no_bypass(tmp_path):
    pool = MockPoolLLM(lambda p: "", capacity=4096, n_slots=8, slot_dir=tmp_path)
    priors = {"big": {"P0": 100, "G": {"0.9": [3900]}}, "small": {"P0": 100, "G": {"0.9": [100]}}}
    kvm = KVManager(pool, EventLog(), load_policy("ours", {"ctx": 4096, "omega": 0.0, "slot_dir": str(tmp_path)}),
                    pool, profiles={}, priors=priors, families=({"big": None, "small": None}, {}))
    await kvm.start()
    await kvm.admit("a", "big", 0)
    kvm.agents["a"].started = True
    kvm.n_not_started = 0
    big = asyncio.create_task(kvm.admit("b", "big", 1))     # (c) fails: 4000 + 4000 > 3968
    small = asyncio.create_task(kvm.admit("c", "small", 2))  # would fit, but must not pass ticket 1
    await asyncio.sleep(0.05)
    assert not big.done() and not small.done() and kvm.adm.blocked == "kv"
    await kvm.release("a", "ok")
    await asyncio.wait_for(big, 1)
    assert not small.done()                                   # b is admitted but has not started yet
    kvm.agents["b"].started = True
    kvm.n_not_started = 0
    kvm._bump()
    await asyncio.sleep(0.01)
    assert not small.done()                                   # 4000 + 200 > 3968: still blocked by (c)
    await kvm.release("b", "ok")
    await asyncio.wait_for(small, 1)
    await kvm.close()


async def test_k_max_caps_admission_under_ample_memory(tmp_path):
    specs = [Spec(i, 2, [10], [0.05], ["search"]) for i in range(12)]
    r = await run_tasks("ours", specs, capacity=65536, k=12, tmp=tmp_path, overrides={"k": 3})
    assert_clean(r, tmp_path)
    assert r.kvm.max_agents == 3 and max(e["n_agents"] for e in r.ev("kv_admit")) == 3
