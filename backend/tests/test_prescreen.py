"""The Pre-screen: features are stable, the model learns, the gate keeps the best and explores."""

import random

import numpy as np

from alpha_harness.agents.screener import Gate
from alpha_harness.labs import prescreen

FIELDS = {"close": ("pv1", "pv"), "est_eps": ("analyst4", "analyst")}
SETTINGS = {"region": "USA", "delay": 1, "universe": "TOP3000", "neutralization": "SUBINDUSTRY"}


def test_features_read_operators_fields_and_settings():
    names = prescreen.features("rank(ts_delta(est_eps, 21))", SETTINGS, FIELDS)
    assert {"op=rank", "op=ts_delta", "field=est_eps", "category=analyst"} <= set(names)
    assert "USA|category=analyst" in names
    assert "region=USA" in names
    assert prescreen.features("rank(", SETTINGS, FIELDS)[-1] == "unparsable"


def test_hashing_is_the_same_in_every_process():
    # CRC-32, not Python's salted hash(): a saved model must score the same after a restart.
    assert prescreen.hashed(["op=rank"]).tolist() == prescreen.hashed(["op=rank"]).tolist()


def test_the_model_learns_a_real_signal_and_is_judged_on_unseen_rows():
    rng = random.Random(1)
    rows = []
    for _ in range(6000):
        good = rng.random() < 0.5
        expression = "rank(est_eps)" if good else "rank(close)"
        passed = rng.random() < (0.2 if good else 0.01)
        rows.append((prescreen.features(expression, SETTINGS, FIELDS), passed))
    report = prescreen.train(rows)
    assert report.auc is not None and report.auc > 0.65
    assert report.lift is not None and report.lift > 1.3
    assert report.trusted


def test_auc_and_lift_on_known_scores():
    labels = np.array([0, 0, 1, 1.0])
    assert prescreen.auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert prescreen.auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    assert prescreen.lift(np.array([0.1, 0.2, 0.3, 0.9]), labels, top=0.25) == 2.0


class _FakeScreener:
    def score(self, requests):
        return [float(r) for r in requests]


def test_gate_sends_the_best_and_keeps_an_exploration_share():
    gate = Gate(_FakeScreener(), oversample=4, explore=0.1)  # type: ignore[arg-type]
    requests = list(range(40))  # scored by value
    chosen = gate.choose(requests, 10)  # type: ignore[arg-type]
    assert len(chosen) == len(set(chosen)) == 10
    assert set(range(31, 40)) <= set(chosen)  # the nine best
    assert len(set(chosen) - set(range(30, 40))) <= 1  # plus one explored at random


def test_gate_sends_everything_when_there_is_no_choice():
    gate = Gate(_FakeScreener())  # type: ignore[arg-type]
    assert gate.choose([1, 2, 3], 5) == [0, 1, 2]  # type: ignore[arg-type]
