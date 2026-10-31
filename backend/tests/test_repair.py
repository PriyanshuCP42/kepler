"""The Repair Engineer: what failed, which fix, what each fix writes, how a repair is settled."""

import json
import random
from types import SimpleNamespace

from alpha_harness.agents import repair
from alpha_harness.db.models import TrialState

OPTIONS = repair.Options(
    universes=("TOP200", "TOP1000", "TOP3000", "MINVOL1M"),
    neutralizations=("MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY", "STATISTICAL"),
)
BASE = {
    "region": "USA",
    "universe": "TOP1000",
    "delay": 1,
    "neutralization": "INDUSTRY",
    "decay": 3,
    "truncation": 0.08,
}


def test_failures_read_brain_checks_and_critic_gates():
    row = {
        "checks": json.dumps(
            [
                {"name": "HIGH_TURNOVER", "result": "FAIL"},
                {"name": "LOW_SHARPE", "result": "PASS"},
                {"name": "PROD_CORRELATION", "result": "FAIL"},  # a label, not a gate
            ]
        )
    }
    gates = {"decay": {"ok": False}, "cost": {"ok": True}, "pool": {"ok": None}}
    assert repair.failures(row, gates) == ["HIGH_TURNOVER", "DECAY"]


def test_every_failure_has_a_fix():
    known = {
        "HIGH_TURNOVER",
        "LOW_TURNOVER",
        "CONCENTRATED_WEIGHT",
        "LOW_SUB_UNIVERSE_SHARPE",
        "LOW_SHARPE",
        "LOW_FITNESS",
        "LOW_2Y_SHARPE",
        "SELF_CORRELATION",
        repair.DECAY,
        repair.COST,
        repair.POOL,
    }
    covered = set().union(*(m.fixes for m in repair.MOVES))
    assert known <= covered


def test_choose_prefers_fixes_that_have_worked():
    stats = {("HIGH_TURNOVER", "smooth_mean"): (20, 15), ("HIGH_TURNOVER", "slower_decay"): (20, 1)}
    rng = random.Random(0)
    firsts = [repair.choose(["HIGH_TURNOVER"], stats, 1, rng)[0][0].id for _ in range(200)]
    assert firsts.count("smooth_mean") > firsts.count("slower_decay") * 5


def test_choose_only_offers_fixes_for_what_failed():
    picked = repair.choose(["LOW_TURNOVER"], {}, 5, random.Random(1))
    assert [m.id for m, _ in picked] == ["faster_decay"]


def test_expression_fixes_wrap_the_original():
    expression, settings = repair.BY_ID["smooth_mean"].apply("rank(close)", BASE, OPTIONS)
    assert expression.endswith("ts_mean(alpha, 5)") and "alpha = rank(close)" in expression
    assert settings == BASE


def test_setting_fixes_step_through_what_the_market_allows():
    _, wider = repair.BY_ID["wider_universe"].apply("x", BASE, OPTIONS)
    assert wider["universe"] == "TOP3000"
    _, narrower = repair.BY_ID["narrower_universe"].apply("x", BASE, OPTIONS)
    assert narrower["universe"] == "TOP200"
    _, finer = repair.BY_ID["finer_neutralization"].apply("x", BASE, OPTIONS)
    assert finer["neutralization"] == "SUBINDUSTRY"
    _, slower = repair.BY_ID["slower_decay"].apply("x", BASE, OPTIONS)
    assert slower["decay"] == 6
    top = BASE | {"universe": "TOP3000"}
    assert repair.BY_ID["wider_universe"].apply("x", top, OPTIONS) is None


def _trial(state, alpha_id=None, finished=None):
    return SimpleNamespace(state=state, alpha_id=alpha_id, finished_at=finished)


def test_a_repair_is_settled_only_once_the_critic_has_judged_it():
    assert repair._outcome(_trial(TrialState.RUNNING), {}) is None
    assert repair._outcome(_trial(TrialState.PRUNED), {}) is None  # not sent yet
    assert repair._outcome(_trial(TrialState.COMPLETE, "a1"), {}) is None  # not judged yet
    # A pass waits for the Family Clerk's pool check before it counts as fixed.
    assert repair._outcome(_trial(TrialState.COMPLETE, "a1"), {"a1": "pass"}) is None
    assert repair._outcome(_trial(TrialState.COMPLETE, "a1"), {"a1": "pass"}, {"a1"}) == "fixed"
    assert repair._outcome(_trial(TrialState.COMPLETE, "a1"), {"a1": "fail"}) == "not_fixed"
    assert repair._outcome(_trial(TrialState.FAIL, finished=1), {}) == "error"
    assert repair._outcome(None, {}) == "cancelled"


def test_near_misses_are_taken_in_turn_from_each_market():
    rows = [
        ("e1", "EUR"),
        ("e2", "EUR"),
        ("e3", "EUR"),
        ("u1", "USA"),
        ("g1", "GLB"),
        ("u2", "USA"),
    ]
    assert repair.interleave(rows) == ["e1", "u1", "g1", "e2", "u2", "e3"]
