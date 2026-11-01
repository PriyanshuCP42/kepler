"""Path to Master: the bars, the peers already there, and the pace to the quarter's end."""

from datetime import date

from alpha_harness.agents import genius, master

ME = {
    "alphaCount": 20,
    "pyramidCount": 5,
    "combinedAlphaPerformance": -0.8,
    "combinedPowerPoolAlphaPerformance": -0.4,
    "combinedSelectedAlphaPerformance": None,
    "combinedOsmosisPerformance": None,
    "operatorAvg": 4.8,
    "fieldAvg": 1.8,
    "operatorCount": 24,
    "fieldCount": 21,
    "maxSimulationStreak": 15,
}


def _peer(level: str, alphas: int, perf: float, osmosis: float = 0.0) -> dict:
    return {
        "geniusLevel": level,
        "alphaCount": alphas,
        "pyramidCount": 20,
        "combinedAlphaPerformance": perf,
        "combinedPowerPoolAlphaPerformance": perf - 0.5,
        "combinedOsmosisPerformance": osmosis,
        "operatorAvg": 3.5,
        "fieldAvg": 1.5,
    }


BOARD = [_peer("MASTER", 100, 1.4), _peer("MASTER", 110, 0.2, osmosis=1.6), _peer("GOLD", 300, -1)]


def test_the_quarter_and_the_month_that_counts_for_performance():
    plan = master.plan(ME, BOARD, date(2026, 11, 2))
    assert plan.quarter_end == date(2026, 12, 31)
    assert plan.performance_cutoff == date(2026, 11, 30)
    assert plan.days_left == 60
    assert master.quarter_end(date(2026, 3, 31)) == date(2026, 3, 31)


def test_the_best_of_four_performances_counts():
    plan = master.plan(ME | {"combinedOsmosisPerformance": 1.2}, BOARD, date(2026, 11, 2))
    performance = next(r for r in plan.bars if r.key == "performance")
    assert performance.you == 1.2
    assert performance.met is (genius.BARS["combined"][1] <= 1.2)


def test_peers_are_only_those_at_the_level_and_their_best_performance():
    medians, counted = master.peer_medians(BOARD, "Master")
    assert counted == 2
    assert medians["alphaCount"] == 105
    assert medians["best"] == 1.5  # median of 1.4 and 1.6 (the Osmosis one)


def test_tie_breaker_averages_are_better_lower():
    plan = master.plan(ME, BOARD, date(2026, 11, 2))
    ops = next(r for r in plan.tie_breakers if r.key == "operatorAvg")
    assert ops.lower_better and ops.met is False
    assert master.Row("x", "x", 3.0, None, 3.5, lower_better=True).met is True


def test_pace_names_each_gap():
    pace = master.plan(ME, BOARD, date(2026, 11, 2)).pace
    assert any("100 more signals in 60 days: 1.7 a day" in p for p in pace)
    assert any("25 more pyramids" in p for p in pace)
    assert any("after 30 Nov may not count" in p for p in pace)
