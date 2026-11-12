"""Today's Picks measures candidates for the Submit Gate: checks, pool effect, lookalikes."""

import asyncio
from datetime import timedelta
from types import SimpleNamespace

from alpha_harness.agents import picks
from alpha_harness.agents.gate import Verdict
from alpha_harness.db.models import utcnow

CLEAN_CHECKS = [
    {"name": "PROD_CORRELATION", "result": "PASS", "value": 0.62},
    {"name": "SELF_CORRELATION", "result": "PASS", "value": 0.2},
    {"name": "POWER_POOL_CORRELATION", "result": "PASS", "value": 0.2},
]


def _row(alpha_id, *, prod=0.62, pending=False):
    return SimpleNamespace(
        alpha_id=alpha_id,
        settings={"region": "USA", "delay": 1, "universe": "TOP3000"},
        sharpe=2.0,
        fitness=1.05,
        margin=0.0006,
        returns=0.04,
        drawdown=0.02,
        turnover=0.12,
        test_sharpe=2.1,
        prod_correlation=prod,
        limits={"LOW_SHARPE": 1.58, "LOW_FITNESS": 1.0},
        pending=pending,
        submitted=False,
    )


class _State:
    def __init__(self, stored, members):
        self._stored = stored
        self._members = members
        self.alphas = self
        self.queries = self
        self.themes = None

    async def by_ids(self, ids):
        return {i: self._stored[i] for i in ids if i in self._stored}

    async def submitted_members(self):
        return self._members

    async def series(self, ids):
        return {}

    async def sole_datasets(self, names):
        owners = {"news_score": "news_ds", "call_tone": "calls_ds", "close": "pv1"}
        return {n: owners[n] for n in names if n in owners}


def test_candidates_are_measured_and_judged(monkeypatch):
    async def sizes(state, texts, region, delay):
        return [{"operators": [4], "fields": [1]} for _ in texts]

    def what_if(series, meta, scopes, joins, since):
        # "good" lifts both pools, "drag" lowers combined.
        return {
            "good": {
                "all": {"before": {"sharpe": 7.0}, "after": {"sharpe": 7.02}},
                "Power Pool": {"before": {"sharpe": 7.0}, "after": {"sharpe": 7.03}},
            },
            "drag": {
                "all": {"before": {"sharpe": 7.0}, "after": {"sharpe": 6.98}},
                "Power Pool": {"before": {"sharpe": 7.0}, "after": {"sharpe": 7.01}},
            },
            "repeat": {
                "all": {"before": {"sharpe": 7.0}, "after": {"sharpe": 7.01}},
                "Power Pool": {"before": {"sharpe": 7.0}, "after": {"sharpe": 7.01}},
            },
        }

    monkeypatch.setattr("alpha_harness.api.template_lab.sizes", sizes)
    monkeypatch.setattr("alpha_harness.tools.portfolio.what_if", what_if)

    stored = {
        "good": {
            "expression": "rank(call_tone)",
            "checks": CLEAN_CHECKS,
            "region": "USA",
            "delay": 1,
        },
        "drag": {
            "expression": "rank(call_tone)",
            "checks": CLEAN_CHECKS,
            "region": "USA",
            "delay": 1,
        },
        "repeat": {
            "expression": "rank(news_score) * rank(close)",
            "checks": CLEAN_CHECKS,
            "region": "USA",
            "delay": 1,
        },
        "fresh": {"expression": "rank(call_tone)", "checks": [], "region": "USA", "delay": 1},
    }
    recent = (utcnow() - timedelta(days=3)).date().isoformat()
    members = [
        {
            "alpha_id": "sub1",
            "region": "USA",
            "delay": 1,
            "date_submitted": recent,
            "expression": "ts_rank(news_score, 20)",
            "classifications": ["Regular Alpha"],
        },
    ]
    rows = [_row("good"), _row("drag"), _row("repeat"), _row("fresh", prod=None, pending=True)]
    measured = {
        m.facts.alpha_id: m for m in asyncio.run(picks.measure(_State(stored, members), rows))
    }

    assert measured["good"].judgement.verdict is Verdict.READY
    assert round(measured["good"].whole_delta, 3) == 0.02
    assert "LOWERS_COMBINED" in measured["drag"].judgement.reasons
    # Same market and dataset (news_ds) as a submission three days old; pv1 alone never counts.
    assert measured["repeat"].lookalike_of == "sub1"
    assert measured["repeat"].datasets == frozenset({"news_ds"})
    # No BRAIN check yet and no effect measured: to check, not judged.
    fresh = measured["fresh"].judgement
    assert fresh.verdict is Verdict.CHECK and "BRAIN_CHECK_NOT_RUN" in fresh.reasons


def test_the_shortlist_keeps_each_unsubmitted_alpha_once_measured_ones_first():
    a, b, c = _row("a", prod=None), _row("b"), _row("b")
    done = _row("d")
    done.submitted = True
    assert [r.alpha_id for r in picks.shortlist([a, b, c, done])] == ["b", "a"]
