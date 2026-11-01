"""The Overfitting Guard: neighbours, the luck bar, consistency, and the verdict."""

from datetime import date, timedelta

import pytest

from alpha_harness.agents import guard
from alpha_harness.agents.guard import NeighbourResult as N
from alpha_harness.api.desk import WHOLE, BrainComparison, Effect, Figures, GuardView, judge

LIFTS = BrainComparison("EQUITY:USA:1", {"sharpe": 2.0}, {"sharpe": 2.2})


class TestNeighbours:
    def test_scales_windows_only(self):
        expression = "ts_mean(close, 20) / ts_std_dev(adv20, 10) + winsorize(x, std=4)"
        assert guard.scale_windows(expression, 0.7) == (
            "ts_mean(close, 14) / ts_std_dev(adv20, 7) + winsorize(x, std=4)"
        )

    def test_leaves_small_numbers_and_says_when_nothing_changed(self):
        assert guard.scale_windows("signed_power(x, 2)", 1.4) is None
        assert guard.scale_windows("rank(close)", 1.4) is None

    def test_every_finalist_gets_settings_neighbours(self):
        row = {
            "expression": "rank(close)",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "neutralization": "SUBINDUSTRY",
            "decay": 4,
            "truncation": 0.08,
        }
        found = dict(guard.neighbours_of(row, {}))
        assert found["decay_up"].settings.decay == 7
        assert found["tighter_truncation"].settings.truncation == pytest.approx(0.06)
        assert "windows_down" not in found

    def test_rank_and_sign_take_the_whole_expression(self):
        row = {
            "expression": "a = ts_rank(close, 20);\na - ts_mean(a, 5)",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "neutralization": "SUBINDUSTRY",
            "decay": 4,
            "truncation": 0.08,
        }
        found = dict(guard.neighbours_of(row, {}))
        assert found["rank"].regular.rstrip().endswith("rank(alpha)")
        assert found["sign"].regular.rstrip().endswith("sign(alpha)")

    def test_the_same_idea_is_tried_in_another_market(self):
        markets = {"DEU": [{"value": "TOP500"}], "GBR": [{"value": "TOP300"}, {"value": "TOP700"}]}
        schema = {
            "universe": {"choices": {"region": markets}},
            "neutralization": {
                "choices": {
                    "region": {
                        "DEU": [{"value": "SUBINDUSTRY"}, {"value": "INDUSTRY"}],
                        "GBR": [{"value": "INDUSTRY"}],
                    }
                }
            },
        }
        row = {
            "expression": "rank(close)",
            "region": "DEU",
            "universe": "TOP500",
            "delay": 1,
            "neutralization": "SUBINDUSTRY",
            "decay": 4,
            "truncation": 0.08,
        }
        moved = dict(guard.neighbours_of(row, schema))["other_market"].settings
        assert (moved.region, moved.universe, moved.neutralization) == ("GBR", "TOP700", "INDUSTRY")
        assert guard.other_market({"region": "BRA"}, schema) is None

    def test_sign_and_another_market_are_held_to_less(self):
        assert guard.holds(2.0, 1.1, 0.5, guard.KEEPS["sign"]) is True
        assert guard.holds(2.0, 1.1, 0.5, guard.KEEPS["windows_up"]) is False
        assert guard.holds(2.0, 0.85, 0.2, guard.KEEPS["other_market"]) is True


class TestLuckBar:
    def test_one_try_has_no_bar(self):
        assert guard.luck_bar(1) == 0.0

    def test_grows_with_tries(self):
        assert 0 < guard.luck_bar(10) < guard.luck_bar(1000)
        # Half of sqrt(2 ln 1000) standard errors, over two test years.
        assert guard.luck_bar(1000, 2) == pytest.approx(0.5 * 3.717 / 2**0.5, abs=0.01)

    def test_skeleton_blanks_numbers_and_spaces(self):
        assert guard.skeleton("ts_mean(x, 20)") == guard.skeleton("ts_mean(x,5)")
        assert guard.skeleton("ts_mean(x, 20)") != guard.skeleton("ts_rank(x, 20)")


def _days(start: date, values: list[float]) -> dict[date, float]:
    return {start + timedelta(days=i): v for i, v in enumerate(values)}


class TestConsistency:
    def test_counts_only_years_it_traded_in(self):
        idle = _days(date(2018, 1, 1), [0.0] * 365)
        traded = _days(date(2021, 1, 1), [1.0] * 365)
        found = guard.consistency(idle | traded, None)
        assert (found.years, found.positive_years) == (1, 1)
        assert found.steady

    def test_a_handful_of_test_days_is_not_active(self):
        days = _days(date(2022, 1, 1), [0.0] * 500)
        days[date(2022, 3, 1)] = 5.0
        found = guard.consistency(days, date(2022, 1, 1))
        assert found.test_days == 1
        assert not found.active

    def test_a_test_carried_by_its_best_days_is_spiked(self):
        values = [-1.0] * 200 + [100.0] * 10
        found = guard.consistency(_days(date(2022, 1, 1), values), date(2022, 1, 1))
        assert found.test_without_spike == pytest.approx(-200.0)
        assert not found.unspiked


class TestVerdict:
    def test_nothing_run_is_untested(self):
        assert guard.verdict([]).status == "untested"

    def test_most_holding_is_robust(self):
        results = [N("a", "a", "done", holds=True)] * 2 + [N("b", "b", "done", holds=False)]
        assert guard.verdict(results).status == "robust"

    def test_decided_fragile_before_the_rest_come_back(self):
        results = [N("a", "a", "done", holds=False)] * 2 + [N("b", "b", "queued")]
        assert guard.verdict(results).status == "fragile"

    def test_waits_while_it_could_still_go_either_way(self):
        results = [N("a", "a", "done", holds=True), N("b", "b", "queued")]
        assert guard.verdict(results).status == "running"

    def test_too_few_back_is_inconclusive(self):
        results = [N("a", "a", "done", holds=True), N("b", "b", "failed")]
        assert guard.verdict(results).status == "inconclusive"

    def test_holds_needs_most_of_the_sharpe_and_a_positive_test(self):
        assert guard.holds(2.0, 1.5, 0.5)
        assert not guard.holds(2.0, 1.2, 0.5)
        assert not guard.holds(2.0, 1.8, -0.1)
        assert guard.holds(None, 1.0, 1.0) is None


def _view(**over) -> GuardView:
    base = {
        "status": "robust",
        "held": 3,
        "measured": 4,
        "neighbours": [],
        "siblings": 1,
        "luck_bar": 0.0,
        "years": 5,
        "positive_years": 5,
        "test_days": 480,
        "test_without_spike": 10.0,
    }
    return GuardView.model_validate(base | over)


def _lifts() -> list[Effect]:
    fig = {"fitness": 1.0, "returns": 0.1, "drawdown": 0.05, "turnover": 0.2, "margin": 1e-3}
    return [
        Effect(
            scope=WHOLE,
            members=5,
            before=Figures(sharpe=1.0, **fig),
            after=Figures(sharpe=1.2, **fig),
        )
    ]


class TestDeskVerdict:
    def test_robust_and_checked_is_ready(self):
        assert judge(0.3, 0.2, _lifts(), _view(), 1.5, brain=LIFTS)[0] == "ready"

    def test_untested_waits_for_the_check(self):
        readiness, reasons = judge(0.3, 0.2, _lifts(), _view(status="untested"), 1.5)
        assert readiness == "check"
        assert "Neighbour check not run yet." in reasons

    def test_fragile_is_overfit(self):
        assert judge(0.3, 0.2, _lifts(), _view(status="fragile"), 1.5)[0] == "overfit"

    def test_under_the_luck_bar_is_overfit(self):
        assert judge(0.3, 0.2, _lifts(), _view(luck_bar=1.2, siblings=500), 1.0)[0] == "overfit"

    def test_few_test_days_is_overfit(self):
        assert judge(0.3, 0.2, _lifts(), _view(test_days=3), 3.3)[0] == "overfit"

    def test_blocked_beats_everything(self):
        assert judge(0.9, 0.2, _lifts(), _view(status="fragile"), 1.5)[0] == "blocked"


def _year_series(sharpe_by_year: dict[int, float]) -> dict[date, tuple[float, float]]:
    """Daily PnL whose yearly Sharpe follows ``sharpe_by_year``, roughly: mean over noise."""
    import numpy as np

    rng = np.random.default_rng(0)
    out: dict[date, tuple[float, float]] = {}
    for year, target in sharpe_by_year.items():
        day = date(year, 1, 1)
        while day.year == year:
            if day.weekday() < 5:
                out[day] = (float(target / 16 + rng.normal(0, 1)), 0.1)
            day += timedelta(days=1)
    return out


class TestRecentForm:
    def test_steady_alpha_is_neither_weak_nor_fading(self):
        form = guard.recent_form(_year_series(dict.fromkeys(range(2016, 2024), 3.0)))
        assert set(form.years) == {2021, 2022, 2023}
        assert form.weak_years == []
        assert not form.fading

    def test_dying_alpha_fades(self):
        early = dict.fromkeys(range(2016, 2021), 6.0)
        late = {2021: 1.0, 2022: 0.8, 2023: 0.6}
        form = guard.recent_form(_year_series(early | late))
        assert form.fading
        assert form.recent is not None
        assert form.full is not None
        assert form.recent < form.full


class TestFadingVerdict:
    def test_a_weak_recent_year_is_fading(self):
        years = [
            {"year": 2021, "sharpe": 2.0},
            {"year": 2022, "sharpe": 0.2},
            {"year": 2023, "sharpe": 2.0},
        ]
        view = _view(recent_years=years, recent_sharpe=1.5, full_sharpe=1.6)
        readiness, reasons = judge(0.3, 0.2, _lifts(), view, 1.5)
        assert readiness == "fading"
        assert "2022" in reasons[0]

    def test_fading_into_the_recent_years(self):
        view = _view(recent_sharpe=1.0, full_sharpe=2.0)
        assert judge(0.3, 0.2, _lifts(), view, 1.5)[0] == "fading"

    def test_holding_up_is_ready(self):
        view = _view(recent_sharpe=1.9, full_sharpe=2.0)
        assert judge(0.3, 0.2, _lifts(), view, 1.5, brain=LIFTS)[0] == "ready"

    def test_lowering_the_pool_in_the_recent_years_hurts(self):
        fig = {"fitness": 1.0, "returns": 0.1, "drawdown": 0.05, "turnover": 0.2, "margin": 1e-3}
        effect = Effect(
            scope=WHOLE,
            members=5,
            before=Figures(sharpe=1.0, **fig),
            after=Figures(sharpe=1.2, **fig),
            recent_before=Figures(sharpe=1.0, **fig),
            recent_after=Figures(sharpe=0.9, **fig),
        )
        readiness, reasons = judge(0.3, 0.2, [effect], _view(), 1.5)
        assert readiness == "hurts"
        assert "2021-2023" in reasons[0]
