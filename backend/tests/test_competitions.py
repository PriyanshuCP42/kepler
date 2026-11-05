"""Competitions: each live one with your tagged Alphas, and each Power Pool theme with yours."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from alpha_harness.agents.themes import Theme
from alpha_harness.api import competitions

NOW = datetime.now(UTC)


def _member(alpha_id, region, tags, classes=("Power Pool Alpha",), sharpe=1.5):
    return {
        "alpha_id": alpha_id,
        "region": region,
        "delay": 1,
        "universe": "TOP3000",
        "sharpe": sharpe,
        "fitness": 0.8,
        "date_submitted": "2026-11-04",
        "tags": json.dumps(list(tags)),
        "classifications": json.dumps(list(classes)),
    }


AMER_FAIL = [{"name": "LOW_GLB_AMER_SHARPE", "result": "FAIL", "value": 0.9, "limit": 1}]
AMER_PASS = [{"name": "LOW_GLB_AMER_SHARPE", "result": "PASS", "value": 1.2, "limit": 1}]


def _state():
    members = [
        _member("glbPass", "GLB", ["PowerPoolSelected"], sharpe=1.4),
        _member("glbFail", "GLB", ["PowerPoolSelected"], sharpe=2.0),
        _member("glbUntagged", "GLB", []),
        _member("usaAqp", "USA", ["AQP202611"], classes=("Regular Alpha",)),
    ]
    checks = {"glbPass": AMER_PASS, "glbFail": AMER_FAIL}

    async def listed():
        return [
            {
                "id": "AQP202611",
                "name": "Alpha Quality Project November 2026",
                "status": "ACCEPTED",
                "signUpDate": "2026-11-05",
                "startDate": (NOW - timedelta(days=1)).isoformat(),
                "endDate": (NOW + timedelta(days=20)).isoformat(),
            }
        ]

    async def detail(cid):
        return {"id": cid, "leaderboard": {"rank": 7, "alphas": 3}}

    async def members_():
        return members

    async def by_ids(ids):
        return {a: {"checks": json.dumps(checks.get(a, []))} for a in ids}

    theme = Theme(
        id="t1",
        name="GLB/D1 Power Pool Nov'26",
        multiplier=1.0,
        region="GLB",
        delay=1,
        qualifier=None,
        requires=("LOW_GLB_AMER_SHARPE",),
    )

    async def current():
        return [theme]

    return SimpleNamespace(
        endpoints=SimpleNamespace(competitions=listed, competition=detail),
        alphas=SimpleNamespace(submitted_members=members_, by_ids=by_ids),
        themes=SimpleNamespace(current=current),
    )


def test_a_live_competition_lists_the_alphas_tagged_with_its_id():
    view = asyncio.run(competitions.competitions(_state()))
    aqp = view.competitions[0]
    assert [e.alpha_id for e in aqp.entries] == ["usaAqp"]
    assert aqp.standing is not None and aqp.standing.alphas == 3


def test_a_theme_lists_tagged_power_pool_alphas_in_its_market_and_which_match():
    view = asyncio.run(competitions.competitions(_state()))
    (theme,) = view.themes
    assert theme.end_date == "2026-11-22" and theme.min_for_awards == 5
    assert [(e.alpha_id, e.matches) for e in theme.entries] == [
        ("glbPass", True),
        ("glbFail", False),
    ]
    assert theme.entries[1].missing == ["LOW_GLB_AMER_SHARPE"]
