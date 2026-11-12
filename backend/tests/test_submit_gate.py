"""The Submit Gate reaches the verdicts this week's picks were given by hand."""

from alpha_harness.agents import gate
from alpha_harness.agents.gate import Facts, Verdict


def _facts(**over):
    """A clean, checked regular Alpha that lifts both pools: READY unless overridden."""
    base = {
        "alpha_id": "a",
        "sharpe": 2.0,
        "fitness": 1.01,
        "sharpe_limit": 1.58,
        "fitness_limit": 1.0,
        "margin": 0.00051,
        "returns": 0.0323,
        "drawdown": 0.0169,
        "train_sharpe": 1.88,
        "test_sharpe": 2.4,
        "prod_correlation": 0.64,
        "self_correlation": 0.19,
        "power_pool_correlation": 0.18,
        "checked": True,
        "whole_delta": 0.009,
        "power_delta": 0.020,
        "operators": 7,
        "fields": 2,
    }
    return Facts(**(base | over))


def test_a_clean_checked_alpha_that_lifts_both_pools_is_ready():
    j = gate.judge(_facts())
    assert j.verdict is Verdict.READY and j.reasons == []
    assert j.regular and j.power_pool
    assert round(j.return_to_drawdown, 2) == 1.91


def test_one_that_loses_money_after_costs_by_a_little_goes_to_repair():
    # 4.5 bps: a near miss on margin, losing money once costs are paid.
    j = gate.judge(_facts(margin=0.00045))
    assert j.verdict is Verdict.REPAIR and j.reasons == ["LOW_MARGIN"]


def test_a_crowded_reversion_or_pool_lowering_alpha_is_rejected():
    assert gate.judge(_facts(prod_correlation=0.84)).reasons == ["PROD_CORRELATION"]
    assert gate.judge(_facts(reversion=True)).verdict is Verdict.REJECT
    lowers = gate.judge(_facts(whole_delta=-0.011, power_delta=-0.007))
    assert lowers.reasons == ["LOWERS_COMBINED", "LOWERS_POWER_POOL"]
    assert gate.judge(_facts(lookalike_of="b")).reasons == ["LOOKALIKE"]


def test_a_weak_drawdown_is_rejected_and_a_slightly_thin_one_repaired():
    # 3.58% a year against a 3.31% drawdown: a year given back on one bad stretch.
    far = gate.judge(_facts(returns=0.0358, drawdown=0.0331, fitness=0.95))
    assert far.verdict is Verdict.REJECT and "RETURN_TO_DRAWDOWN" in far.reasons
    assert "LOW_FITNESS" in far.reasons  # near on its own, listed after the hard reason
    near = gate.judge(_facts(returns=0.04, drawdown=0.029))  # 1.38
    assert near.verdict is Verdict.REPAIR and near.reasons == ["RETURN_TO_DRAWDOWN"]


def test_one_not_yet_checked_or_measured_needs_a_check_not_a_verdict():
    j = gate.judge(_facts(checked=False, prod_correlation=None, whole_delta=None))
    assert j.verdict is Verdict.CHECK
    assert j.reasons == ["BRAIN_CHECK_NOT_RUN", "POOL_EFFECT_NOT_MEASURED"]


def test_brains_own_refusals_reject_and_its_test_years_must_hold():
    refused = gate.judge(_facts(brain_fails=["LOW_INVESTABILITY_CONSTRAINED_SHARPE"]))
    assert refused.reasons == ["BRAIN_LOW_INVESTABILITY_CONSTRAINED_SHARPE"]
    # Test years keeping 40% of the train Sharpe: fitted, not found.
    assert "TEST_DECAY" in gate.judge(_facts(test_sharpe=0.75)).reasons


def test_a_pure_power_pool_alpha_needs_its_theme():
    weak = {"sharpe": 1.2, "fitness": 0.6}
    themed = gate.judge(_facts(**weak, theme_match=True))
    assert themed.verdict is Verdict.READY and not themed.regular and themed.power_pool
    # Under Power Pool's own Sharpe floor of 1.0, even with the theme.
    assert "LOW_SHARPE" in gate.judge(_facts(sharpe=0.7, fitness=0.4, theme_match=True)).reasons
    off = gate.judge(_facts(**weak, theme_match=False))
    assert off.verdict is Verdict.REJECT and "NOT_SUBMITTABLE" in off.reasons
