"""The Submission Desk: a candidate's effect on your pool, and its verdict."""

from datetime import date, timedelta

import numpy as np
import pytest

from alpha_harness.api.desk import (
    WHOLE,
    BrainComparison,
    Effect,
    Figures,
    comparison_of,
    judge,
    sharpe_change,
)
from alpha_harness.tools import portfolio


def _series(seed: int, days: int = 300, drift: float = 1.0):
    rng = np.random.default_rng(seed)
    start = date(2022, 1, 3)
    return {start + timedelta(days=i): (float(drift + rng.normal(0, 10)), 0.1) for i in range(days)}


def _brain(before: float, after: float) -> BrainComparison:
    return BrainComparison(
        "EQUITY:ASI:1",
        {"sharpe": before, "fitness": 2.0, "drawdown": 0.02, "margin": 5e-4},
        {"sharpe": after, "fitness": 2.1, "drawdown": 0.018, "margin": 5.2e-4},
    )


LIFTS = _brain(4.45, 4.82)


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

    def test_over_the_production_target_is_blocked(self):
        assert judge(0.65, 0.2, self._lifts())[0] == "blocked"

    def test_lowering_the_pool_hurts(self):
        assert judge(0.3, 0.2, self._lifts(1.2, 1.0))[0] == "hurts"

    def test_a_small_partition_cannot_hold_back_a_pool_lift(self):
        partition = Effect(
            scope="EUR · D1 · Power Pool", members=1, before=_figures(1.2), after=_figures(1.0)
        )
        assert judge(0.3, 0.2, [*self._lifts(), partition], brain=LIFTS)[0] == "ready"

    def test_unchecked_production_correlation_is_checked_first(self):
        assert judge(None, 0.2, self._lifts())[0] == "check"

    def test_checked_and_lifting_is_ready(self):
        readiness, reasons = judge(0.4, 0.2, self._lifts(), brain=LIFTS)
        assert readiness == "ready"
        assert reasons[0].startswith("Production Correlation")
        assert any("1.00 → 1.20" in r for r in reasons)
        assert any("lifts your ASI D1 alphas (Sharpe 4.45 → 4.82" in r for r in reasons)

    def test_nothing_is_ready_before_brain_compares_it(self):
        readiness, reasons = judge(0.4, 0.2, self._lifts())
        assert readiness == "check"
        assert "before-and-after comparison not run yet" in reasons[0]

    def test_brain_saying_it_lowers_the_group_overrules_the_estimate(self):
        readiness, reasons = judge(0.4, 0.2, self._lifts(), brain=_brain(4.45, 3.80))
        assert readiness == "hurts"
        assert reasons[0].startswith("BRAIN's own comparison: it lowers your ASI D1 alphas")

    def test_a_rounding_sized_fall_is_not_harm(self):
        assert judge(0.4, 0.2, self._lifts(), brain=_brain(3.80, 3.795))[0] == "ready"

    def test_comparison_reads_brains_body_and_names_the_group(self):
        body = {
            "partitionName": "EQUITY:ASI:1:POWER_POOL",
            "stats": {"before": {"sharpe": 4.45}, "after": {"sharpe": 4.82}},
        }
        found = comparison_of(body)
        assert found is not None and found.group == "ASI D1 Power Pool"
        assert comparison_of({"stats": {}}) is None

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


def _kept(*checks: dict) -> dict:
    return {"is": {"checks": list(checks)}}


LOW_SHARPE_FAIL = {"name": "LOW_SHARPE", "result": "FAIL", "value": 1.17, "limit": 1.58}
LOW_SHARPE_WARN = {"name": "LOW_SHARPE", "result": "WARNING", "value": 1.17, "limit": 1.58}
NO_DESCRIPTION = {
    "name": "POWER_POOL_DESCRIPTION_LENGTH",
    "result": "WARNING",
    "value": 0,
    "limit": 100,
}
FITNESS_PASS = {"name": "LOW_FITNESS", "result": "PASS", "value": 1.39, "limit": 1.0}


class TestSubmissionCheck:
    def test_not_run_carries_what_the_simulation_warned(self):
        from alpha_harness.api.desk import submission_of

        found = submission_of(None, None, [LOW_SHARPE_WARN, FITNESS_PASS])
        assert found.status == "not_run"
        assert [c.name for c in found.warned] == ["LOW_SHARPE"]
        assert found.failing == []

    def test_a_refusal_fails_and_asks_for_the_description(self):
        from datetime import UTC, datetime

        from alpha_harness.api.desk import submission_of

        found = submission_of(
            _kept(LOW_SHARPE_FAIL, FITNESS_PASS, NO_DESCRIPTION),
            datetime(2026, 10, 31, tzinfo=UTC),
            [],
        )
        assert found.status == "fails"
        assert [c.name for c in found.failing] == ["LOW_SHARPE"]
        assert found.needs_description

    def test_a_clean_check_passes(self):
        from datetime import UTC, datetime

        from alpha_harness.api.desk import submission_of

        assert (
            submission_of(_kept(FITNESS_PASS), datetime(2026, 10, 31, tzinfo=UTC), []).status
            == "passes"
        )

    def test_a_refused_alpha_is_never_ready(self):
        from datetime import UTC, datetime

        from alpha_harness.api.desk import submission_of

        refused = submission_of(
            _kept(LOW_SHARPE_FAIL, NO_DESCRIPTION), datetime(2026, 10, 31, tzinfo=UTC), []
        )
        readiness, reasons = judge(0.3, 0.2, [], None, 1.5, refused)
        assert readiness == "fails_check"
        assert "Power Pool description" in reasons[0]

    def test_unchecked_can_be_ready_with_a_caution(self):
        from alpha_harness.api.desk import submission_of

        unchecked = submission_of(None, None, [LOW_SHARPE_WARN])
        readiness, reasons = judge(0.3, 0.2, [], None, 1.5, unchecked, brain=LIFTS)
        assert readiness == "ready"
        assert any("run Check Submission" in r for r in reasons)

    def test_unchecked_without_warnings_has_no_caution(self):
        from alpha_harness.api.desk import submission_of

        unchecked = submission_of(None, None, [FITNESS_PASS])
        readiness, reasons = judge(0.3, 0.2, [], None, 1.5, unchecked, brain=LIFTS)
        assert readiness == "ready"
        assert not any("Check Submission" in r for r in reasons)


class TestFixRule:
    def _refused(self, *lines: dict, describe: bool = False):
        from datetime import UTC, datetime

        from alpha_harness.api.desk import submission_of

        checks = list(lines) + ([NO_DESCRIPTION] if describe else [])
        return submission_of(_kept(*checks), datetime(2026, 10, 31, tzinfo=UTC), [])

    def test_mechanical_refusals_are_repaired(self):
        from alpha_harness.api.desk import fix_for

        concentrated = {
            "name": "CONCENTRATED_WEIGHT",
            "result": "FAIL",
            "value": 0.17,
            "limit": 0.1,
        }
        found = fix_for(self._refused(concentrated), power_pool=False, repaired=False)
        assert found is not None and found.action == "repair"

    def test_a_near_strength_miss_is_repaired(self):
        from alpha_harness.api.desk import fix_for

        near = {"name": "LOW_2Y_SHARPE", "result": "FAIL", "value": 2.65, "limit": 2.69}
        found = fix_for(self._refused(near), power_pool=False, repaired=False)
        assert found is not None and found.action == "repair"

    def test_a_far_strength_miss_is_dropped(self):
        from alpha_harness.api.desk import fix_for

        found = fix_for(self._refused(LOW_SHARPE_FAIL), power_pool=False, repaired=False)
        assert found is not None and found.action == "drop"
        assert "overfitting" in found.reason

    def test_a_power_pool_alpha_without_description_is_told_to_add_one(self):
        from alpha_harness.api.desk import fix_for

        refused = self._refused(LOW_SHARPE_FAIL, describe=True)
        found = fix_for(refused, power_pool=True, repaired=False)
        assert found is not None and found.action == "describe"

    def test_one_repair_round_only(self):
        from alpha_harness.api.desk import fix_for

        concentrated = {
            "name": "CONCENTRATED_WEIGHT",
            "result": "FAIL",
            "value": 0.17,
            "limit": 0.1,
        }
        found = fix_for(self._refused(concentrated), power_pool=False, repaired=True)
        assert found is not None and found.action == "repaired"

    def test_nothing_refused_needs_no_fix(self):
        from alpha_harness.api.desk import fix_for, submission_of

        assert fix_for(submission_of(None, None, []), power_pool=True, repaired=False) is None


class TestBrainComparisonHarm:
    def test_fitness_and_margin_falls_block_even_when_sharpe_rises(self):
        # alpha-b's own comparison: Sharpe up, but Fitness down and margin to 4.1 bps.
        found = BrainComparison(
            "EQUITY:ASI:1:POWER_POOL",
            {"sharpe": 4.45, "fitness": 2.38, "drawdown": 0.0187, "margin": 5.7e-4},
            {"sharpe": 4.70, "fitness": 2.12, "drawdown": 0.0182, "margin": 4.1e-4},
        )
        assert found.harms == ["Fitness falls", "margin falls under 5 bps"]
        readiness, reasons = judge(0.4, 0.2, [], brain=found)
        assert readiness == "hurts"
        assert "Fitness falls and margin falls under 5 bps" in reasons[0]

    def test_a_margin_already_thin_but_rising_is_not_held_against_it(self):
        found = BrainComparison(
            "EQUITY:EUR:1",
            {"sharpe": 3.0, "fitness": 2.0, "margin": 3.0e-4},
            {"sharpe": 3.1, "fitness": 2.1, "margin": 3.5e-4},
        )
        assert found.harms == []


class TestBindingCorrelation:
    def _days(self, seed: int, base=None):
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 1, 900) if base is None else base + rng.normal(0, 0.1, 900)
        start = date(2020, 1, 1)
        return {start + timedelta(days=i): float(noise[i]) for i in range(900)}

    def test_a_twin_of_a_submitted_alpha_is_held_to_its_correlation(self):
        from alpha_harness.agents import families
        from alpha_harness.api.desk import binding_correlation

        member = self._days(1)
        twin = self._days(2, base=np.array(list(member.values())))
        pool = families.pool_series({"alpha-a": member, "other": self._days(3)})
        rho = binding_correlation(twin, 4.74, pool, {"alpha-a": 4.45, "other": 2.0})
        assert rho is not None and rho > 0.9

    def test_ten_percent_more_sharpe_is_brains_exception(self):
        from alpha_harness.agents import families
        from alpha_harness.api.desk import binding_correlation

        member = self._days(1)
        twin = self._days(2, base=np.array(list(member.values())))
        pool = families.pool_series({"alpha-a": member, "other": self._days(3)})
        rho = binding_correlation(twin, 5.0, pool, {"alpha-a": 4.45, "other": 2.0})
        assert rho is not None and rho < 0.7
