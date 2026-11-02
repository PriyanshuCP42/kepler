"""What stood between strong alphas and the Desk: the Critic's decay ratio, a shortlist held by
blocked alphas, and plans that ignored pyramids one short and markets your pool is heavy in."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from alpha_harness.agents import critic
from alpha_harness.agents import strategist as st
from alpha_harness.api import desk
from alpha_harness.db.models import CriticVerdict
from alpha_harness.db.sqlite import Database

if TYPE_CHECKING:
    from pathlib import Path


def _row(train: float, test: float, limit: float = 1.58) -> dict[str, Any]:
    return {
        "sharpe": (train + test) / 2,
        "fitness": 2.0,
        "margin": 0.0009,
        "train_sharpe": train,
        "test_sharpe": test,
        "operator_count": 6,
        "checks": json.dumps(
            [{"name": "LOW_SHARPE", "result": "PASS", "limit": limit, "value": (train + test) / 2}]
        ),
    }


class TestDecay:
    def test_a_strong_test_sharpe_holds_up_against_a_stronger_train(self):
        judged = critic.judge(_row(train=4.92, test=2.61))
        assert judged.gates["decay"]["ok"] is True
        assert judged.verdict == "pass"

    def test_a_weak_test_sharpe_still_decays(self):
        judged = critic.judge(_row(train=2.4, test=1.2))
        assert judged.gates["decay"]["ok"] is False
        assert judged.verdict == "near_miss"

    def test_strength_does_not_excuse_a_collapse(self):
        assert critic.judge(_row(train=8.0, test=2.5)).gates["decay"]["ok"] is False

    def test_without_a_sharpe_limit_the_ratio_decides(self):
        row = _row(train=4.92, test=2.61)
        row["checks"] = json.dumps(
            [{"name": "LOW_FITNESS", "result": "PASS", "limit": 1.0, "value": 2.0}]
        )
        assert critic.judge(row).gates["decay"]["ok"] is False


async def test_blocked_alphas_wait_behind_every_open_one(monkeypatch):
    ids = [f"a{i}" for i in range(desk.CANDIDATES + 5)]
    blocked = set(ids[:10])

    async def prod(_state, alpha_ids, _checks):
        return {a: (0.8 if a in blocked else 0.4) for a in alpha_ids if a != "a12"}

    async def no_members(_state, _leaders, _stored):
        return {}

    monkeypatch.setattr(desk, "prod_correlations", prod)
    monkeypatch.setattr(desk, "_clean_members", no_members)
    chosen = [SimpleNamespace(alpha_id=a) for a in ids]
    stored = {a: {"checks": []} for a in ids}
    picked = [v.alpha_id for v in await desk._unblocked_first(None, chosen, stored)]
    assert len(picked) == desk.CANDIDATES
    assert picked[: len(ids) - 10] == ids[10:]
    assert "a12" in picked


@pytest.fixture
async def db(tmp_path: Path):
    database = Database(tmp_path / "t.db")
    await database.create_all()
    yield database
    await database.dispose()


async def test_a_blocked_leader_gives_its_place_to_a_clean_member(db, monkeypatch):
    measured = {"lead": 0.62, "close": 0.65, "clean": 0.59, "cleaner": 0.56, "other": 0.4}
    async with db.session() as s:
        for alpha_id, score in (("lead", 83), ("close", 82), ("clean", 80), ("cleaner", 70)):
            s.add(CriticVerdict(alpha_id=alpha_id, verdict="pass", score=score, family_id="lead"))
        s.add(CriticVerdict(alpha_id="other", verdict="pass", score=60, family_id="other"))
        await s.commit()

    async def prod(_state, alpha_ids, _checks):
        return {a: measured[a] for a in alpha_ids}

    class Alphas:
        async def by_ids(self, ids):
            return {a: {"alpha_id": a, "checks": None} for a in ids}

    monkeypatch.setattr(desk, "prod_correlations", prod)
    state = SimpleNamespace(db=db, alphas=Alphas())
    chosen = [SimpleNamespace(alpha_id="lead"), SimpleNamespace(alpha_id="other")]
    stored = {"lead": {"checks": None}, "other": {"checks": None}}
    picked = [v.alpha_id for v in await desk._unblocked_first(state, chosen, stored)]
    assert picked == ["clean", "other"]
    assert "clean" in stored


def _niche(region="ASI", delay=1, category="analyst", **kw):
    niche = st.Niche(region, delay, category, category.title(), synced=True)
    for key, value in kw.items():
        setattr(niche, key, value)
    return niche


def test_one_short_of_a_pyramid_is_worth_the_most():
    assert st.need(2) == st.LAST_ONE
    assert st.need(2) > 1.5 * st.need(1)


def test_a_market_heavy_in_the_pool_is_cooled():
    heavy = _niche(region="GLB", simulations=500, families=10, multiplier=1.2)
    light = _niche(region="KOR", simulations=500, families=10, multiplier=1.2)
    st.assess([heavy, light], random.Random(0), pool={("GLB", 1): 26, ("KOR", 1): 4})
    assert heavy.boost == st.POOL_HEAVY_FACTOR
    assert light.boost == 1.0
    assert any("already holds 26" in r for r in heavy.reasons)
