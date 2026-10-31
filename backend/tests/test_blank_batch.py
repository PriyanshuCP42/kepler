"""A batch BRAIN fails whole, with no alpha and no reason, is sent again once. No BRAIN access."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import select

from alpha_harness.db.models import SimStatus, SimulationRecord
from alpha_harness.db.sqlite import Database
from alpha_harness.engine.lifecycle import Outcome
from alpha_harness.engine.slots import BatchEngine

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
async def db(tmp_path: Path):
    database = Database(tmp_path / "t.db")
    await database.create_all()
    yield database
    await database.dispose()


class _Tracker:
    async def notify(self) -> None:
        pass

    def alpha_landed(self, alpha_id: str) -> None:
        pass


def _engine(db: Database, results: dict[str, tuple[dict[str, Any], Outcome]]) -> BatchEngine:
    engine = object.__new__(BatchEngine)
    engine.db = db
    engine.tracker = _Tracker()  # type: ignore[assignment]
    engine._child_results = results
    return engine


def _row(n: int, **extra: Any) -> SimulationRecord:
    return SimulationRecord(
        request_hash=f"h{n}",
        payload={"type": "REGULAR", "regular": f"rank(x{n})", "settings": {"region": "ASI"}},
        expression=f"rank(x{n})",
        region="ASI",
        delay=1,
        task="t",
        **extra,
    )


async def _batch(db: Database, *, requeued: int = 0) -> int:
    async with db.session() as s:
        parent = _row(0, is_batch=True, status=SimStatus.RUNNING, platform_status="ERROR")
        parent.child_ids = ["c1", "c2"]
        s.add(parent)
        await s.flush()
        for n in (1, 2):
            s.add(
                _row(
                    n,
                    status=SimStatus.RUNNING,
                    parent_record_id=parent.id,
                    orphan_requeues=requeued,
                )
            )
        await s.flush()
        return parent.id


def _blank() -> dict[str, tuple[dict[str, Any], Outcome]]:
    failed = Outcome("finished", SimStatus.FAILED)
    return {c: ({"id": c, "type": "REGULAR", "status": "FAIL"}, failed) for c in ("c1", "c2")}


async def _children(db: Database) -> list[SimulationRecord]:
    async with db.session() as s:
        return list(
            await s.scalars(select(SimulationRecord).where(SimulationRecord.is_batch.is_(False)))
        )


async def test_blank_batch_failure_is_queued_again(db) -> None:
    parent_id = await _batch(db)
    async with db.session() as s:
        parent = await s.get(SimulationRecord, parent_id)
    await _engine(db, _blank())._expand(parent)

    children = await _children(db)
    assert {c.status for c in children} == {SimStatus.QUEUED}
    assert all(c.parent_record_id is None and c.orphan_requeues == 1 for c in children)


async def test_blank_batch_failure_is_not_retried_twice(db) -> None:
    parent_id = await _batch(db, requeued=1)
    async with db.session() as s:
        parent = await s.get(SimulationRecord, parent_id)
    await _engine(db, _blank())._expand(parent)

    assert {c.status for c in await _children(db)} == {SimStatus.FAILED}
