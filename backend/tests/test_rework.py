"""Rework: which of BRAIN's refusals the Autopilot hands to the Repair Engineer."""

import json

from alpha_harness.agents import rework


def _row(sharpe=2.1, fitness=1.1, **checks):
    listed = [
        {"name": "LOW_SHARPE", "result": "PASS" if sharpe >= 1.58 else "FAIL", "limit": 1.58},
        {"name": "LOW_FITNESS", "result": "PASS" if fitness >= 1 else "FAIL", "limit": 1.0},
    ]
    listed += [{"name": n, **c} for n, c in checks.items()]
    return {"sharpe": sharpe, "fitness": fitness, "checks": json.dumps(listed)}


def test_a_strong_alpha_refused_by_a_little_is_reworked():
    crowded = _row(PROD_CORRELATION={"result": "FAIL", "value": 0.76})
    assert rework.worth_reworking(crowded) == ("PROD_CORRELATION",)
    small_caps = _row(LOW_SUB_UNIVERSE_SHARPE={"result": "FAIL", "value": 0.84})
    assert rework.worth_reworking(small_caps) == ("LOW_SUB_UNIVERSE_SHARPE",)


def test_what_no_move_fixes_is_left_alone():
    # Too crowded to bring back, under the Sharpe bar, or a reversion bet.
    assert rework.worth_reworking(_row(PROD_CORRELATION={"result": "FAIL", "value": 0.88})) is None
    weak = _row(sharpe=1.3, LOW_SUB_UNIVERSE_SHARPE={"result": "FAIL"})
    assert rework.worth_reworking(weak) is None
    reverting = _row(
        LOW_SUB_UNIVERSE_SHARPE={"result": "FAIL"}, REVERSION_COMPONENT={"result": "WARNING"}
    )
    assert rework.worth_reworking(reverting) is None
    # Nothing failed: BRAIN passes it already. Three failures: it needs a new idea.
    assert rework.worth_reworking(_row()) is None
    many = _row(
        LOW_SUB_UNIVERSE_SHARPE={"result": "FAIL"},
        HIGH_TURNOVER={"result": "FAIL"},
        CONCENTRATED_WEIGHT={"result": "FAIL"},
    )
    assert rework.worth_reworking(many) is None
    # A check with a move is tried; one no move answers is never fixable.
    assert rework.worth_reworking(_row(LOW_2Y_SHARPE={"result": "FAIL"})) == ("LOW_2Y_SHARPE",)
    assert "MATCHES_PYRAMID" not in rework.FIXABLE
