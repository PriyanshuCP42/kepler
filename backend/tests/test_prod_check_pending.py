"""A Production Correlation BRAIN is still working out is asked for again after the rest, once;
and a round of Alphas with no stored PnL reads as none rather than crashing the Family Clerk."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import select

from alpha_harness.api.lab_tasks import AlphaIds, check_prod_correlation
from alpha_harness.brain.errors import BrainPollTimeout
from alpha_harness.brain.jobs import BrainJobs
from alpha_harness.db.models import BrainCache
from alpha_harness.db.sqlite import Database
from alpha_harness.vault.store import _by_alpha

if TYPE_CHECKING:
    from pathlib import Path


def test_no_rows_is_no_alphas():
    assert _by_alpha([], [], []) == {}


@pytest.fixture
async def db(tmp_path: Path):
    database = Database(tmp_path / "t.db")
    await database.create_all()
    yield database
    await database.dispose()


class _Endpoints:
    def __init__(self, slow: dict[str, int]) -> None:
        self.slow = slow
        self.asked: list[str] = []

    async def correlations(self, alpha_id: str, kind: str, ceiling: float = 0) -> dict[str, Any]:
        self.asked.append(alpha_id)
        if self.slow.get(alpha_id, 0) > 0:
            self.slow[alpha_id] -= 1
            raise BrainPollTimeout("still pending")
        return {"max": 0.5}


class _Alphas:
    async def by_ids(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        return {a: {"alpha_id": a, "checks": None} for a in ids}


class _Tasks:
    async def update(self, task: Any, **_: Any) -> None:
        pass


class _Backfill:
    def __init__(self) -> None:
        self.detail = ""

    async def start_job(self, kind: str, label: str, work: Any) -> str:
        self.detail = await work(object())
        return "job"


def _state(db: Database, slow: dict[str, int]) -> Any:
    state = type("S", (), {})()
    state.db, state.endpoints, state.alphas = db, _Endpoints(slow), _Alphas()
    state.tasks, state.backfill = _Tasks(), _Backfill()
    state.jobs = BrainJobs()
    return state


async def _kept(db: Database) -> set[str]:
    async with db.session() as s:
        keys = await s.scalars(select(BrainCache.key))
        return {k.removeprefix("correlation:prod:") for k in keys}


async def test_a_pending_check_is_asked_again_after_the_rest(db):
    state = _state(db, {"a": 1})
    await check_prod_correlation(AlphaIds.model_validate({"alphaIds": ["a", "b", "c"]}), state)
    assert state.endpoints.asked == ["a", "b", "c", "a"]
    assert await _kept(db) == {"a", "b", "c"}
    assert "could not check" not in state.backfill.detail


async def test_one_still_pending_twice_is_given_up(db):
    state = _state(db, {"a": 5})
    await check_prod_correlation(AlphaIds.model_validate({"alphaIds": ["a", "b"]}), state)
    assert state.endpoints.asked == ["a", "b", "a"]
    assert await _kept(db) == {"b"}
    assert "could not check 1" in state.backfill.detail
