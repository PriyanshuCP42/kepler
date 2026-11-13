"""PnL twins: an Alpha moving with one BRAIN already checked gets that one's verdict."""

import json

import numpy as np

from alpha_harness.agents import gate, twins


def _grid(days: int = 800, seed: int = 4) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=days)
    other = rng.normal(size=days)
    columns = {
        "new": base + 0.1 * rng.normal(size=days),  # twin of "crowded"
        "crowded": base,
        "fresh": other,
        "short": np.where(np.arange(days) < days - 100, np.nan, base),  # 100 days only
    }
    return list(columns), np.column_stack(list(columns.values()))


def test_a_candidate_takes_its_twins_production_correlation():
    ids, grid = _grid()
    twin = twins.nearest(ids, grid, "new", {"crowded": 0.83, "fresh": 0.41})
    assert twin is not None and twin.alpha_id == "crowded" and twin.prod_correlation == 0.83
    assert twin.correlation >= twins.TWIN_CORRELATION


def test_no_twin_when_nothing_checked_moves_with_it():
    ids, grid = _grid()
    assert twins.nearest(ids, grid, "new", {"fresh": 0.41}) is None
    # Too few shared days is no evidence, however close they move.
    assert twins.nearest(ids, grid, "new", {"short": 0.9}) is None
    # Never its own twin, and an Alpha without a PnL has none.
    assert twins.nearest(ids, grid, "crowded", {"crowded": 0.83}) is None
    assert twins.nearest(ids, grid, "missing", {"crowded": 0.83}) is None


def test_the_measured_value_is_read_from_checks_first():
    checks = [{"name": "PROD_CORRELATION", "result": "FAIL", "value": 0.78}]
    assert twins.prod_value({"checks": json.dumps(checks), "prod_correlation": None}) == 0.78
    assert twins.prod_value({"checks": checks}) == 0.78
    pending = [{"name": "PROD_CORRELATION", "result": "PENDING"}]
    assert twins.prod_value({"checks": json.dumps(pending), "prod_correlation": 0.6}) == 0.6
    assert twins.prod_value({"checks": json.dumps(pending)}) is None


def _facts(**over):
    base = {
        "alpha_id": "x",
        "sharpe": 2.0,
        "fitness": 1.2,
        "sharpe_limit": 1.58,
        "fitness_limit": 1.0,
        "margin": 0.0008,
        "returns": 0.08,
        "drawdown": 0.03,
        "whole_delta": 0.01,
        "power_delta": 0.01,
    }
    return gate.Facts(**(base | over))


def test_the_gate_spends_no_check_on_an_alpha_whose_twin_failed():
    failed = gate.judge(_facts(twin_of="t", twin_prod=0.81))
    assert failed.verdict is gate.Verdict.REJECT and "PROD_TWIN" in failed.reasons
    # A passing twin is not a pass: BRAIN still has to check it.
    passed = gate.judge(_facts(twin_of="t", twin_prod=0.52))
    assert passed.verdict is gate.Verdict.CHECK and "BRAIN_CHECK_NOT_RUN" in passed.reasons
    # Once BRAIN has measured the Alpha itself, its own figure decides.
    own = gate.judge(_facts(twin_of="t", twin_prod=0.81, prod_correlation=0.6, checked=True))
    assert own.verdict is gate.Verdict.READY
