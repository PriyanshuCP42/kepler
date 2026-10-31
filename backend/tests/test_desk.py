"""The Submission Desk: a candidate's effect on your pool, and its verdict."""

from datetime import date, timedelta

import numpy as np
import pytest

from alpha_harness.api.desk import WHOLE, Effect, Figures, judge, sharpe_change
from alpha_harness.tools import portfolio


def _series(seed: int, days: int = 300, drift: float = 1.0):
    rng = np.random.default_rng(seed)
    start = date(2022, 1, 3)
    return {start + timedelta(days=i): (float(drift + rng.normal(0, 10)), 0.1) for i in range(days)}


def _figures(sharpe: float | None) -> Figures:
    return Figures(
        sharpe=sharpe, fitness=1.0, returns=0.1, drawdown=0.05, turnover=0.2, margin=1e-3
    )


class TestWhatIf:
    def test_after_is_the_pool_computed_with_the_candidate(self):
        series = {"a": _series(1), "b": _series(2), "c": _series(3, drift=3.0)}
        found = portfolio.what_if(series, {}, {WHOLE: ["a", "b"]}, {"c": [WHOLE]})
        effect = found["c"][WHOLE]
        assert effect["members"] == 2
        alone = portfolio.compute(series, {}, ["a", "b"], 0.0)["stats"]["in_sample"]
        joined = portfolio.compute(series, {}, ["a", "b", "c"], 0.0)["stats"]["in_sample"]
        assert effect["before"]["sharpe"] == pytest.approx(alone["sharpe"])
        assert effect["after"]["sharpe"] == pytest.approx(joined["sharpe"])

    def test_a_new_partition_starts_with_the_candidate_alone(self):
        series = {"a": _series(1), "c": _series(3)}
        found = portfolio.what_if(series, {}, {WHOLE: ["a"], "new": []}, {"c": [WHOLE, "new"]})
        assert found["c"]["new"]["before"] is None
        assert found["c"]["new"]["after"] is not None

    def test_a_candidate_without_pnl_is_left_out(self):
        found = portfolio.what_if({"a": _series(1)}, {}, {WHOLE: ["a"]}, {"x": [WHOLE]})
        assert "x" not in found


class TestVerdict:
    def _lifts(self, before: float = 1.0, after: float = 1.2) -> list[Effect]:
        return [Effect(scope=WHOLE, members=5, before=_figures(before), after=_figures(after))]

    def test_high_production_correlation_blocks(self):
        assert judge(0.75, 0.2, self._lifts())[0] == "blocked"

    def test_high_pool_correlation_blocks(self):
        assert judge(None, 0.71, self._lifts())[0] == "blocked"

    def test_lowering_the_pool_hurts(self):
        assert judge(0.3, 0.2, self._lifts(1.2, 1.0))[0] == "hurts"

    def test_unchecked_production_correlation_is_checked_first(self):
        assert judge(None, 0.2, self._lifts())[0] == "check"

    def test_checked_and_lifting_is_ready(self):
        readiness, reasons = judge(0.4, 0.2, self._lifts())
        assert readiness == "ready"
        assert any("1.00 → 1.20" in r for r in reasons)

    def test_sharpe_change(self):
        assert sharpe_change(self._lifts(1.0, 1.25), WHOLE) == pytest.approx(0.25)
        assert sharpe_change(self._lifts(), "USA · D1 · Regular") is None


def test_what_if_reports_the_recent_years_too():
    series = {"a": _series(1, days=900), "c": _series(3, days=900, drift=3.0)}
    found = portfolio.what_if(series, {}, {WHOLE: ["a"]}, {"c": [WHOLE]}, date(2023, 1, 1))
    effect = found["c"][WHOLE]
    assert effect["recent_before"] is not None
    assert effect["recent_after"]["days"] < effect["after"]["days"]


def test_combined_matches_the_what_if():
    series = {"a": _series(1, days=900), "b": _series(2, days=900), "c": _series(3, days=900)}
    since = date(2023, 1, 1)
    joined = portfolio.combined(series, {}, ["a", "b", "c"], since)
    found = portfolio.what_if(series, {}, {WHOLE: ["a", "b"]}, {"c": [WHOLE]}, since)
    assert joined["all"]["sharpe"] == pytest.approx(found["c"][WHOLE]["after"]["sharpe"])
    assert joined["recent"]["sharpe"] == pytest.approx(found["c"][WHOLE]["recent_after"]["sharpe"])


def test_combined_of_nothing_is_empty():
    assert portfolio.combined({}, {}, ["x"]) == {"all": None, "recent": None}


def test_the_ready_basket_moves_both_bars():
    from alpha_harness.api.desk import POWER, _basket

    series = {k: _series(i, days=900) for i, k in enumerate(["a", "b", "c", "d"], start=1)}
    scopes = {WHOLE: ["a", "b"], POWER: ["b"]}
    found = _basket(series, {}, scopes, ["c", "d"], {"c": True, "d": False})
    whole, power = found
    assert (whole.scope, whole.members, power.scope, power.members) == (WHOLE, 2, POWER, 1)
    assert whole.after != whole.before
    assert power.after != power.before


def test_an_empty_basket_changes_nothing():
    from alpha_harness.api.desk import POWER, _basket

    series = {"a": _series(1, days=900)}
    found = _basket(series, {}, {WHOLE: ["a"], POWER: []}, [], {})
    assert found[0].after == found[0].before
