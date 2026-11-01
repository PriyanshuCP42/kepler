"""Probe first: a large template task that finds nothing near BRAIN's bar stops early."""

import json

from alpha_harness.labs import probe


def _row(sharpe: float | None, limit: float = 1.58) -> dict:
    return {
        "sharpe": sharpe,
        "checks": json.dumps([{"name": "LOW_SHARPE", "result": "FAIL", "limit": limit}]),
    }


def test_nothing_near_the_bar_stops():
    found = probe.judge([_row(0.79), _row(0.72), _row(None)])
    assert not found.passed
    assert abs(found.bar - probe.PROBE_SHARE * 1.58) < 1e-9
    assert "0.79" in found.message
    assert "run the task again" in found.message


def test_one_close_result_carries_on():
    assert probe.judge([_row(0.5), _row(1.3)]).passed


def test_the_bar_is_the_markets_own():
    assert not probe.judge([_row(2.0, limit=2.69)]).passed
    assert probe.judge([_row(2.2, limit=2.69)]).passed


def test_without_a_reported_limit_delay_one_is_assumed():
    assert probe.sharpe_limit([{"sharpe": 1.0, "checks": None}]) == probe.FALLBACK_LIMIT


def test_due_once_enough_is_back_on_a_large_task_never_twice():
    assert probe.due({}, 300, 30)
    assert not probe.due({}, 300, 29)
    assert not probe.due({}, 50, 40)
    assert not probe.due({probe.KEY: "stopped"}, 300, 60)
