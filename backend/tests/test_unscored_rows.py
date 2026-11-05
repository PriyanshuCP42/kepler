"""A task's results say why a quality trial has no score, and fill figures BRAIN sent late."""

from types import SimpleNamespace

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
    merged = lab_tasks._merged(row, lab_tasks._from_vault("a1", vault, {}))
    assert merged["sharpe"] == 0.41 and merged["fitness"] == 0.08
    assert merged["turnover"] == 0.3  # the trial's own figure stands
    assert merged["testFitness"] == 0.2
