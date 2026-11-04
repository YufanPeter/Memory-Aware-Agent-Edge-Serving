import random
from collections import Counter

import pytest

from edgeserve import workload
from edgeserve.workload import arrivals, make_workload


@pytest.fixture
def fake_split(monkeypatch, item):
    from dataclasses import replace
    items = [replace(item, id=f"q{i}") for i in range(50)]
    monkeypatch.setattr(workload, "load_split", lambda name: items)


SPEC = {"name": "w", "split": "dev", "n_tasks": 10, "seed": 3,
        "mix": {"a": 0.5, "b": 0.3, "c": 0.2}, "arrival": {"type": "poisson", "rate": 1.0}}


def test_workload_reproducible_and_mix_exact(fake_split):
    w1, w2 = make_workload(SPEC), make_workload(SPEC)
    assert w1 == w2
    assert Counter(t["profile"] for t in w1["tasks"]) == {"a": 5, "b": 3, "c": 2}
    assert len({t["qid"] for t in w1["tasks"]}) == 10
    times = [t["arrival_s"] for t in w1["tasks"]]
    assert times == sorted(times) and times[0] == 0.0


def test_arrival_types():
    rng = random.Random(0)
    assert arrivals({"type": "batch"}, 3, rng) == [0.0, 0.0, 0.0]
    assert arrivals({"type": "uniform", "interval": 2}, 3, rng) == [0, 2, 4]
    b = arrivals({"type": "bursty", "rate_on": 5, "rate_off": 0.1, "mean_on_s": 5, "mean_off_s": 20}, 50, rng)
    assert len(b) == 50 and b == sorted(b) and b[0] == 0.0
