"""The Genius tracker: levels, the estimate by date, and whether it tracks BRAIN."""

from datetime import date, timedelta

import numpy as np

from alpha_harness.agents import genius
from alpha_harness.agents.genius import Member, Step, agreement, build, estimate, level_of
from alpha_harness.db.models import GeniusReading


def _series(seed: int, drift: float = 1.0) -> dict[date, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    start = date(2020, 1, 1)
    return {start + timedelta(days=i): (float(drift + rng.normal(0, 10)), 0.1) for i in range(1400)}


def _step(estimate: float | None, brain: float | None) -> Step:
    return Step(date(2026, 1, 1), date(2026, 1, 8), [], brain, None, estimate, None)


def test_levels():
    assert level_of("combined", -0.8) is None
    assert level_of("combined", 0.5) == "Expert"
    assert level_of("signals", 150) == "Master"
    assert level_of("pyramids", None) is None


def test_estimate_counts_only_alphas_submitted_by_then():
    members = [
        Member("a", date(2026, 8, 1), False),
        Member("b", date(2026, 9, 1), True),
    ]
    series = {"a": _series(1), "b": _series(2)}
    early = estimate(series, {}, members, date(2026, 8, 15))
    late = estimate(series, {}, members, date(2026, 9, 15))
    assert (early.alphas, late.alphas) == (1, 2)
    assert early.power_pool is None
    assert late.power_pool is not None


def test_too_few_steps_wait():
    found = agreement([_step(0.5, -0.2), _step(0.4, 0.1)])
    assert found.pairs == 2
    assert found.correlation is None
    assert "2 of 3" in found.verdict


def test_an_estimate_that_moves_with_brain_is_trusted():
    found = agreement([_step(0.5, 0.3), _step(-0.2, -0.4), _step(0.8, 0.6), _step(0.1, 0.05)])
    assert found.same_direction == 4
    assert found.correlation is not None and found.correlation > 0.9
    assert "trust" in found.verdict


def test_an_estimate_that_moves_against_brain_says_so():
    found = agreement([_step(0.5, -0.3), _step(0.9, -0.7), _step(0.2, -0.1)])
    assert found.same_direction == 0
    assert "against" in found.verdict


def test_build_lines_up_readings_and_steps():
    members = [Member("a", date(2026, 8, 1), False), Member("b", date(2026, 9, 1), False)]
    series = {"a": _series(1), "b": _series(2)}
    readings = [
        GeniusReading(id=2, read_on=date(2026, 9, 15), combined=-0.4),
        GeniusReading(id=1, read_on=date(2026, 8, 15), combined=0.3),
    ]
    report = build(readings, members, series, {})
    assert [r["readOn"] for r in report.readings] == ["2026-08-15", "2026-09-15"]
    assert len(report.steps) == 1
    step = report.steps[0]
    assert step.submitted == ["b"]
    assert step.brain_combined is not None and abs(step.brain_combined + 0.7) < 1e-9
    assert report.today is not None and report.today.alphas == 2


def test_bars_cover_every_genius_property():
    assert set(genius.BARS) == {"signals", "pyramids", "combined", "power_pool"}


def test_steps_where_nothing_moved_are_not_agreement():
    found = agreement([_step(0.0, 0.0), _step(0.0, 0.0), _step(0.5, -0.3)])
    assert found.pairs == 1
    assert found.same_direction == 0


def test_two_readings_of_one_day_make_no_step():
    members = [Member("a", date(2026, 8, 1), False)]
    readings = [
        GeniusReading(id=1, read_on=date(2026, 9, 1), combined=0.1),
        GeniusReading(id=2, read_on=date(2026, 9, 1), combined=0.2),
    ]
    report = build(readings, members, {"a": _series(1)}, {})
    assert report.steps == []


def test_the_quality_score_rewards_less_drawdown_and_turnover():
    from alpha_harness.tools.portfolio import quality_change

    base = {"sharpe": 2.0, "fitness": 1.0, "margin": 5e-4, "turnover": 0.3, "drawdown": 0.05}
    calmer = base | {"turnover": 0.2, "drawdown": 0.03}
    found = quality_change(base, calmer)
    assert found is not None and found["total"] > 0
    assert found["parts"]["turnover"] > 0 and found["parts"]["drawdown"] > 0
    worse = quality_change(base, base | {"drawdown": 0.10})
    assert worse is not None and worse["total"] < 0
    assert worse["parts"]["drawdown"] == -0.5  # capped
    assert quality_change(None, base) is None
