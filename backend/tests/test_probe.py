"""Probe first: a large template task that finds nothing near BRAIN's bar stops early."""

import json

from alpha_harness.labs import probe


def _row(sharpe: float | None, limit: float = 1.58) -> dict:
    return {
        "sharpe": sharpe,
        "checks": json.dumps([{"name": "LOW_SHARPE", "result": "FAIL", "limit": limit}]),
    }


def _rows(best: float, limit: float = 1.58) -> list[dict]:
    return [_row(0.3, limit) for _ in range(probe.PROBE_TRIALS - 1)] + [_row(best, limit)]


def test_nothing_near_the_bar_stops():
    found = probe.judge(_rows(0.79))
    assert found is not None and not found.passed
    assert abs(found.bar - probe.PROBE_SHARE * 1.58) < 1e-9
    assert "0.79" in found.message
    assert "run the task again" in found.message


def test_one_close_result_carries_on():
    found = probe.judge(_rows(1.3))
    assert found is not None and found.passed


def test_the_bar_is_the_markets_own():
    assert not probe.judge(_rows(2.0, limit=2.69)).passed
    assert probe.judge(_rows(2.2, limit=2.69)).passed


def test_it_waits_for_enough_stored_results_with_a_sharpe():
    short = _rows(0.1)[1:]
    assert probe.judge(short) is None
    assert probe.judge([*short, _row(None)]) is None


def test_without_a_reported_limit_delay_one_is_assumed():
    assert probe.sharpe_limit([{"sharpe": 1.0, "checks": None}]) == probe.FALLBACK_LIMIT


def test_due_once_a_fifth_is_back_on_a_large_task_never_twice():
    assert probe.due({}, 150, 30)
    assert not probe.due({}, 300, 59)
    assert probe.due({}, 300, 60)
    assert not probe.due({}, 50, 40)
    assert not probe.due({probe.KEY: "stopped"}, 300, 60)


def test_a_larger_task_waits_longer_but_never_past_the_cap():
    assert probe.trials_for(100) == probe.PROBE_TRIALS
    assert probe.trials_for(300) == 60
    assert probe.trials_for(5000) == probe.PROBE_MAX
    assert probe.judge(_rows(0.79), trials=60) is None
