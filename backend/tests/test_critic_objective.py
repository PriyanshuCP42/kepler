"""The Critic Score the labs maximise: train years only, structural checks, pool overlap."""

import json

import pytest

from alpha_harness.agents.critic import overlap_factor, search_score
from alpha_harness.agents.overlap import Overlap, fields_of, jaccard
from alpha_harness.labs.launch import lab_of
from alpha_harness.labs.objectives import CRITIC, OBJECTIVES, resolve
from alpha_harness.labs.study import row_values


def _row(**over):
    row = {
        "sharpe": 1.6,
        "fitness": 1.1,
        "margin": 0.0006,
        "train_sharpe": 1.7,
        "train_fitness": 1.2,
        "test_sharpe": 1.3,
        "operator_count": 4,
        "checks": json.dumps([{"name": "LOW_SHARPE", "result": "PASS"}]),
        "expression": "rank(close / open)",
        "region": "USA",
    }
    return {**row, **over}


class TestSearchScore:
    def test_never_reads_the_test_years(self):
        assert search_score(_row(test_sharpe=2.5)) == search_score(_row(test_sharpe=-1.0))

    def test_rises_with_train_sharpe(self):
        assert search_score(_row(train_sharpe=2.0)) > search_score(_row(train_sharpe=1.0))

    def test_scores_the_whole_period_without_a_split(self):
        bare = _row(train_sharpe=None, train_fitness=None)
        assert search_score(bare) == search_score(_row(train_sharpe=1.6, train_fitness=1.1))

    def test_no_sharpe_is_no_score(self):
        assert search_score(_row(train_sharpe=None, sharpe=None)) is None

    def test_thin_margin_costs_sharpe(self):
        assert search_score(_row(margin=0.0002)) < search_score(_row(margin=0.001))

    def test_a_structural_check_costs_a_return_check_does_not(self):
        turnover = json.dumps([{"name": "HIGH_TURNOVER", "result": "FAIL"}])
        sharpe = json.dumps([{"name": "LOW_SHARPE", "result": "FAIL"}])
        assert search_score(_row(checks=turnover)) == pytest.approx(
            0.7 * search_score(_row()), abs=0.01
        )
        assert search_score(_row(checks=sharpe)) == search_score(_row())

    def test_reads_checks_already_decoded(self):
        decoded = [{"name": "CONCENTRATED_WEIGHT", "result": "FAIL"}]
        assert search_score(_row(checks=decoded)) < search_score(_row())

    def test_oversized_expressions_lose_half(self):
        big, small = search_score(_row(operator_count=12)), search_score(_row(operator_count=8))
        assert big < 0.6 * small

    def test_overlap_with_the_pool_costs_up_to_half(self):
        assert overlap_factor(0.0) == 1.0
        assert overlap_factor(1.0) == 0.5
        assert search_score(_row(), 1.0) == pytest.approx(0.5 * search_score(_row()), abs=0.01)


class TestOverlap:
    def test_fields_and_jaccard(self):
        a = fields_of("rank(close / open) + ts_mean(volume, 5)")
        assert a == {"close", "open", "volume"}
        assert jaccard(a, fields_of("rank(close - open)")) == pytest.approx(2 / 3)
        assert fields_of("rank(") == frozenset()
        assert jaccard(frozenset(), a) == 0.0

    def test_worst_is_within_the_region(self):
        overlap = Overlap(None, None)  # type: ignore[arg-type]
        overlap.pool = {"USA": [frozenset({"close", "open"})], "EUR": [frozenset({"volume"})]}
        assert overlap.worst("rank(close / open)", "USA") == 1.0
        assert overlap.worst("rank(volume)", "USA") == 0.0
        assert overlap.worst("rank(volume)", "EUR") == 1.0


class TestObjective:
    def test_is_a_known_objective(self):
        assert resolve([CRITIC]) == [OBJECTIVES[CRITIC]]

    def test_row_values_scores_a_row_and_waits_for_a_missing_one(self):
        objectives = resolve([CRITIC])
        assert row_values(_row(), objectives, None) == [search_score(_row())]
        assert row_values(_row(train_sharpe=None, sharpe=None), objectives, None) is None

    def test_row_values_charges_the_pool_overlap(self):
        overlap = Overlap(None, None)  # type: ignore[arg-type]
        overlap.pool = {"USA": [frozenset({"close", "open"})]}
        charged = row_values(_row(), resolve([CRITIC]), overlap)
        assert charged == [search_score(_row(), 1.0)]

    def test_plain_statistics_still_read_straight_off_the_row(self):
        assert row_values(_row(), resolve(["train_sharpe"]), None) == [1.7]


def test_lab_of_names_the_lab_from_the_task():
    assert lab_of("template-261018204305843437") == "Template Lab"
    assert lab_of("settings-sampler-261018204305843437") is not None
    assert lab_of("manual") is None


class TestCost:
    """Costs at the 5 bps the After-Cost Sharpe columns charge, not a lighter 2."""

    def test_charged_like_the_tables(self):
        from alpha_harness.agents import critic
        from alpha_harness.vault import metrics

        assert critic.COST_BPS == metrics.DEFAULT_COST_BPS

    def test_a_margin_that_loses_money_after_costs_earns_nothing_for_its_sharpe(self):
        # 4 bps of margin kept a third of the Sharpe at 2 bps; at 5 bps it is a loss, scored
        # like the 5 bps margin that leaves nothing.
        assert search_score(_row(margin=0.0004)) == search_score(_row(margin=0.0005))
        assert search_score(_row(margin=0.0004)) < search_score(_row(margin=0.0009))

    def test_the_critic_gates_a_margin_under_its_cost(self):
        from alpha_harness.agents.critic import judge

        thin = judge(_row(margin=0.0006))
        assert thin.gates["cost"]["ok"] is False
        assert judge(_row(margin=0.0009)).gates["cost"]["ok"] is True
