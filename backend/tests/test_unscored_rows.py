"""A task's results say why a quality trial has no score, and fill figures BRAIN sent late."""

from types import SimpleNamespace

from alpha_harness.agents import themes
from alpha_harness.api import lab_tasks
from alpha_harness.db.models import TrialState
from alpha_harness.labs import quality
from alpha_harness.labs.study import ranked


def _trial(number, result, values=(0.0, 0.0, 0.4)):
    return SimpleNamespace(
        id=number,
        number=number,
        study_id=1,
        state=TrialState.COMPLETE,
        values=list(values),
        alpha_id=f"a{number}",
        expression="rank(close)",
        settings={},
        params={},
        result=result,
        generation=None,
    )


MISSED_MARGIN = {
    "stats": {"sharpe": 1.3, "turnover": 0.3},
    "quality": {"kRatio": None, "calmar": None},
    # sharpe, checks, margin, pool, prod: only the margin is unmet.
    "constraints": [0.0, 0.0, 0.0002, 0.0, 0.0],
}
SCORED = {
    "stats": {"sharpe": 1.8},
    "quality": {"kRatio": 4.2, "calmar": 1.1},
    "constraints": [0.0, 0.0, 0.0, 0.0, 0.0],
}


def test_an_unscored_quality_trial_says_why_instead_of_scoring_zero():
    assert quality.unscored(MISSED_MARGIN) == ["margin under 5 bps"]
    assert quality.unscored(SCORED) is None
    assert quality.unscored({"stats": {}}) is None  # not a quality trial at all
    rows = {r["alphaId"]: r for r in ranked([_trial(1, MISSED_MARGIN), _trial(2, SCORED, (4.2,))])}
    assert rows["a1"]["value"] is None and rows["a1"]["unscored"] == ["margin under 5 bps"]
    assert rows["a2"]["value"] == 4.2 and rows["a2"]["unscored"] == []


def test_a_figure_brain_sent_late_is_filled_from_the_vault_never_overwritten():
    row = {"alphaId": "a1", "sharpe": None, "fitness": None, "turnover": 0.3}
    vault = {"a1": {"sharpe": 0.41, "fitness": 0.08, "turnover": 0.9, "test_fitness": 0.2}}
    merged = lab_tasks._merged(row, lab_tasks._from_vault("a1", vault, {}), vault["a1"])
    assert merged["sharpe"] == 0.41 and merged["fitness"] == 0.08
    assert merged["turnover"] == 0.3  # the trial's own figure stands
    assert merged["testFitness"] == 0.2


def test_an_alpha_that_traded_in_only_one_period_is_held_back():
    def periods(train, test):
        return {
            "test_start": "2021-12-31",
            "train_sharpe": train,
            "train_fitness": train,
            "test_sharpe": test,
            "test_fitness": test,
        }

    train_only, test_only, both = periods(9.78, 0.0), periods(0.0, 3.46), periods(1.66, 1.24)
    assert lab_tasks.short_history(train_only) and lab_tasks.short_history(test_only)
    assert not lab_tasks.short_history(both)
    assert not lab_tasks.short_history({"train_sharpe": 0.0, "test_sharpe": 0.0})  # no test period
    row = {"alphaId": "a1", "submittable": True, "pending": True}
    row |= {"failedChecks": [], "refusedBy": []}
    held = lab_tasks._merged(row, {}, test_only)
    assert held["submittable"] is False and held["pending"] is False
    assert held["refusedBy"] == [lab_tasks.SHORT_HISTORY]


def test_a_measured_production_correlation_at_the_ceiling_holds_an_alpha_back():
    row = {"alphaId": "a1", "submittable": True, "pending": True}
    row |= {"failedChecks": [], "refusedBy": []}
    crowded = {"prodCorrelation": 0.81}
    held = lab_tasks._merged(row, crowded, {})
    assert held["submittable"] is False and held["refusedBy"] == ["PROD_CORRELATION"]
    # BRAIN's own check passed it (its Sharpe beat each production Alpha it repeats by 10%).
    assert lab_tasks._merged(row, crowded, {}, prod_passed=True)["submittable"] is True
    assert lab_tasks._merged(row, {"prodCorrelation": 0.62}, {})["submittable"] is True


def test_a_pure_power_pool_alpha_no_theme_covers_is_held_back():
    nov = themes.parse({"id": "t1", "name": "GLB/D1 Power Pool Nov`26", "multiplier": 1.0})
    assert nov is not None

    def checks(sharpe, amer=1.2):
        return [
            {"name": "LOW_SHARPE", "result": "WARNING", "value": sharpe, "limit": 1.58},
            {
                "name": "LOW_GLB_AMER_SHARPE",
                "result": "PASS" if amer > 1.0 else "WARNING",
                "value": amer,
                "limit": 1.0,
            },
            {"name": "POWER_POOL_CORRELATION", "result": "PASS", "value": 0.2},
        ]

    row = {"alphaId": "a1", "submittable": True, "pending": True}
    row |= {"failedChecks": [], "refusedBy": []}
    gbr = {"region": "GBR", "delay": 0, "universe": "TOP800"}
    glb = {"region": "GLB", "delay": 1, "universe": "TOP3000"}
    held = lab_tasks._merged(row, {}, gbr, themes=[nov], checks=checks(1.25))
    assert held["submittable"] is False and held["refusedBy"] == [lab_tasks.OFF_THEME]
    # Clear of the regular limits: a regular alpha, which needs no theme.
    assert lab_tasks._merged(row, {}, gbr, themes=[nov], checks=checks(1.9))["submittable"]
    # In the theme's market and clear of what it asks beyond it.
    assert lab_tasks._merged(row, {}, glb, themes=[nov], checks=checks(1.25))["submittable"]
    # In the market but under the Americas Sharpe the November theme asks for.
    missed = lab_tasks._merged(row, {}, glb, themes=[nov], checks=checks(1.25, amer=0.8))
    assert missed["refusedBy"] == [lab_tasks.OFF_THEME]
    # Themes unknown: nothing is held back for want of them.
    assert lab_tasks._merged(row, {}, gbr, themes=[], checks=checks(1.25))["submittable"]


def test_an_alpha_brain_warns_has_a_reversion_component_is_held_back():
    row = {"alphaId": "a1", "submittable": True, "pending": True}
    row |= {"failedChecks": [], "refusedBy": []}
    warned = [{"name": "REVERSION_COMPONENT", "result": "WARNING"}]
    held = lab_tasks._merged(row, {}, {}, checks=warned)
    assert held["submittable"] is False
    assert held["refusedBy"] == [lab_tasks.REVERSION_COMPONENT]
    # Read from the stored checks when BRAIN's latest check is not to hand.
    stored = {"checks": warned}
    assert lab_tasks._merged(row, {}, stored)["refusedBy"] == [lab_tasks.REVERSION_COMPONENT]
    clean = [{"name": "REVERSION_COMPONENT", "result": "PASS"}]
    assert lab_tasks._merged(row, {}, {}, checks=clean)["submittable"] is True
