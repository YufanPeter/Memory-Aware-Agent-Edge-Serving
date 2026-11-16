"""Presets and the pure decision functions (docs/phase2_spec.md §4, test 17)."""
import math
import random

import pytest
import yaml

from edgeserve.kvm import CostModel, KVConfig
from edgeserve.policies import (KVM_CONFIG_PATH, PRIOR_UNTIL, DelayModel, Victim, break_even, choose_action,
                                lognormal_residual, load_policy, rank_victims, residual)


def test_every_preset_loads_with_a_consistent_gate():
    presets = yaml.safe_load(KVM_CONFIG_PATH.read_text(encoding="utf-8"))["presets"]
    gates = {"observe": "none", "server": "active", "managed": "full"}
    for name in presets:
        cfg = load_policy(name, {"ctx": 16384})
        assert isinstance(cfg, KVConfig) and cfg.ctx == 16384
        assert cfg.gate == gates[cfg.lifecycle], name
    assert load_policy("observe").admission == "fixed"
    assert (load_policy("gated_k").lifecycle, load_policy("adm_retain").admission) == ("server", "kv")
    assert load_policy("ours_h0").host_frac == 0
    assert (load_policy("ttl_off").timing, load_policy("ttl_off").actions) == ("fixed_ttl", "offload")
    assert load_policy("ours", {"omega": 0.5, "host_frac": 2.0}).omega == 0.5
    assert load_policy("adm_duty").theta == 0.8


def test_load_policy_rejects_bad_input():
    with pytest.raises(ValueError, match="unknown policy"):
        load_policy("nope")
    with pytest.raises(ValueError, match="unknown keys"):
        load_policy("ours", {"omgea": 1.0})
    with pytest.raises(ValueError, match="needs gate"):
        load_policy("gated_k", {"gate": "full"})          # retain + full gate deadlocks
    with pytest.raises(ValueError, match="timing"):
        load_policy("ours", {"timing": "sometimes"})


def test_lognormal_residual_matches_monte_carlo():
    rng = random.Random(0)
    xs = [min(rng.lognormvariate(math.log(0.8), 0.6), 5.0) for _ in range(200_000)]
    for e in (0.0, 0.5, 1.0, 2.0):
        tail = [x - e for x in xs if x > e]
        assert lognormal_residual(e, 0.8, 0.6, 5.0) == pytest.approx(sum(tail) / len(tail), rel=0.02)
    assert lognormal_residual(5.0, 0.8, 0.6, 5.0) == 0.0


def test_oracle_prior_scales_with_the_delay_scale():
    assert residual("lookup", 0.05, 1.0) == pytest.approx(0.05)
    assert residual("lookup", 0.1, 4.0) == pytest.approx(0.3) and residual("lookup", 0.5, 4.0) == 0.0
    assert residual("search", 0.0, 1.0) == pytest.approx(0.8 * math.exp(0.18), rel=0.01)   # ~ the mean
    assert residual("search", 1.2, 4.0) == pytest.approx(4 * residual("search", 0.3, 1.0))
    assert 0 < residual("search", 4.0, 1.0) < 1.0 and residual("search", 5.0, 1.0) == 0.0   # the 5 s cap


def test_online_residual_is_empirical_after_enough_calls():
    m = DelayModel()
    assert m.residual("search", 0.0) == pytest.approx(lognormal_residual(0.0, 1.0, 1.0))   # prior only
    for d in (1.0, 2.0, 3.0, 4.0):
        m.observe("search", d)
    prior = lognormal_residual(1.5, 1.0, 1.0)
    got = m.residual("search", 1.5)
    assert min(1.5, prior) < got < max(1.5, prior)                                    # blended
    for i in range(PRIOR_UNTIL):
        m.observe("search", 0.1 * (i + 1))
    xs = m.obs["search"]
    tail = [x - 1.5 for x in xs if x > 1.5]
    assert m.residual("search", 1.5) == pytest.approx(sum(tail) / len(tail))           # empirical only
    assert m.residual("search", 99.0) == 0.0 and m.median("search") == xs[len(xs) // 2]
    assert m.count("lookup") == 0


def test_rank_victims_orders_classes_residuals_and_ready_agents():
    vs = [Victim(5, 4, 900, ready_seq=3), Victim(6, 4, 900, ready_seq=8), Victim(1, 3, 3000, residual=0.0),
          Victim(2, 2, 1000, residual=0.4), Victim(3, 2, 1200, residual=2.5), Victim(4, 2, 2000, residual=2.5),
          Victim(7, 1, 500), Victim(8, 0, 4000)]
    assert [v.slot for v in rank_victims(vs)] == [8, 7, 4, 3, 2, 1, 6, 5]
    assert [v.slot for v in rank_victims(vs, "lowest_id")] == [8, 1, 2, 3, 4, 5, 6, 7]
    assert [v.slot for v in rank_victims(vs, "largest")] == [8, 1, 4, 3, 2, 5, 6, 7]


def test_choose_action_offloads_long_tails_and_drops_short_ones():
    cm = CostModel()                                  # the E0b priors
    room = True

    def act(s, keep=1400, room=room, actions="cost", owned=True):
        return choose_action(s, keep, room, cm.c_off(s), cm.c_drop(s, keep, 0), actions, owned)
    assert cm.c_off(3500) == pytest.approx(0.175) and cm.c_drop(3500, 1400, 0) == pytest.approx(0.462)
    assert act(3500) == "offload"                     # react_long tail: 175 vs 460 ms
    assert cm.c_off(1800) == pytest.approx(0.09) and cm.c_drop(1800, 1400, 0) == pytest.approx(0.088)
    assert act(1800) == "drop"                        # react_short: 90 vs 88 ms, the prefix stays
    assert act(3500, room=False) == "drop"            # full host tier
    assert act(3500, keep=0) == "erase"               # no family (oneshot)
    assert act(1400) == "erase"                       # already at its prefix
    assert act(3500, owned=False) == "drop"           # stale KV is never offloaded
    assert act(1500, actions="offload") == "offload" and act(3500, actions="drop") == "drop"
    assert act(3500, actions="erase") == "erase"
    assert cm.c_drop(3500, 1400, 16000) > cm.c_drop(3500, 1400, 0)     # the span term


def test_break_even_and_t_star_by_hand():
    B, t_star = break_even(9000, 3000, 0.15, 0.4, True)          # spec §4.3: A 9k, S 3k, c_off 0.15
    assert B == pytest.approx(12000 * 0.15) and t_star == pytest.approx(0.6)
    B, t_star = break_even(9000, 3000, 0.15, 0.4, False)         # no host room: drop's cost
    assert B == pytest.approx(12000 * 0.4) and t_star == pytest.approx(1.6)
    assert break_even(0, 0, 0.1, 0.1, True) == (0.0, 0.0)


def test_cost_model_refits_from_server_timings():
    cm = CostModel(alpha=0.5)
    cm.observe("save", 3000, 120.0)                    # 40 µs/token
    assert cm.kappa_s == pytest.approx(0.5 * 35e-6 + 0.5 * 40e-6)
    cm.observe("save", 100, 50.0)                      # too small: the fixed cost dominates
    assert cm.kappa_s == pytest.approx(37.5e-6)
    cm.observe("restore", 3000, 45.0)
    assert cm.kappa_r == pytest.approx(0.5 * 15e-6 + 0.5 * 15e-6)
    cm.observe("prefill", 2000, 1014.0, span=24000)    # E0b idlecost: 2k tokens at 24k idle cells
    assert cm.t_pf == pytest.approx(0.5 * 220e-6 + 0.5 * (1014e-3 / 2000 - 9.7e-6 * 24))
    cm.observe("erase", 3000, 1.4)
    cm.observe("erase", 3000, None)                    # no server timing: ignored (callers pass wall time)
    assert cm.erase_s == pytest.approx(0.5 * 0.002 + 0.5 * 0.0014) and cm.n_obs["erase"] == 1
