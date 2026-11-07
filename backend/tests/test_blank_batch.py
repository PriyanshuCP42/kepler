"""A batch BRAIN ends ERROR sends its blank children again once. No BRAIN access.

Blank means no alpha and no reason: every child FAIL (BRAIN's hiccup), or the siblings BRAIN
CANCELLED alongside one child it refused by name. Expressions here are synthetic.
"""

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


async def _batch(
    db: Database, *, requeued: int = 0, children: int = 2, platform_status: str = "ERROR"
) -> int:
    async with db.session() as s:
        parent = _row(0, is_batch=True, status=SimStatus.RUNNING, platform_status=platform_status)
        parent.child_ids = [f"c{n}" for n in range(1, children + 1)]
        s.add(parent)
        await s.flush()
        for n in range(1, children + 1):
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


REFUSED = 'Attempted to use unknown variable "x9".'


def _one_refused(children: int = 3) -> dict[str, tuple[dict[str, Any], Outcome]]:
    """c1 refused by name; BRAIN cancels the rest with no alpha and no reason."""
    refused = Outcome("finished", SimStatus.ERROR, message=REFUSED)
    results = {"c1": ({"id": "c1", "status": "ERROR"}, refused)}
    cancelled = Outcome("finished", SimStatus.CANCELLED)
    for n in range(2, children + 1):
        results[f"c{n}"] = ({"id": f"c{n}", "status": "CANCELLED"}, cancelled)
    return results


async def _children(db: Database) -> list[SimulationRecord]:
    async with db.session() as s:
        return list(
            await s.scalars(
                select(SimulationRecord)
                .where(SimulationRecord.is_batch.is_(False))
                .order_by(SimulationRecord.id)
            )
        )


async def _expand(db: Database, parent_id: int, results: dict[str, Any]) -> None:
    async with db.session() as s:
        parent = await s.get(SimulationRecord, parent_id)
    assert parent is not None
    await _engine(db, results)._expand(parent)


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


async def test_siblings_of_a_refused_child_are_queued_again(db) -> None:
    parent_id = await _batch(db, children=3)
    await _expand(db, parent_id, _one_refused())

    refused, *siblings = await _children(db)
    # The refused child is finished with BRAIN's reason and never sent again.
    assert refused.status == SimStatus.ERROR
    assert REFUSED in (refused.message or "")
    assert refused.orphan_requeues == 0
    assert {c.status for c in siblings} == {SimStatus.QUEUED}
    assert all(c.parent_record_id is None and c.platform_id is None for c in siblings)
    assert all(c.orphan_requeues == 1 for c in siblings)
    assert all("refused" in (c.message or "") for c in siblings)


async def test_siblings_of_a_refused_child_are_not_retried_twice(db) -> None:
    parent_id = await _batch(db, children=3, requeued=1)
    await _expand(db, parent_id, _one_refused())

    refused, *siblings = await _children(db)
    assert refused.status == SimStatus.ERROR
    assert {c.status for c in siblings} == {SimStatus.CANCELLED}
    assert all(c.orphan_requeues == 1 for c in siblings)


async def test_requeued_sibling_is_finished_if_its_new_batch_fails_too(db) -> None:
    """Sent again beside another bad child, a sibling ends there instead of looping."""
    parent_id = await _batch(db, children=3)
    await _expand(db, parent_id, _one_refused())
    async with db.session() as s:
        again = await s.get(SimulationRecord, (await _children(db))[1].id)
        assert again is not None
        parent = _row(9, is_batch=True, status=SimStatus.RUNNING, platform_status="ERROR")
        parent.child_ids = ["d1", "d2"]
        s.add(parent)
        await s.flush()
        bad = _row(8, status=SimStatus.RUNNING, parent_record_id=parent.id)
        s.add(bad)
        again.status, again.parent_record_id = SimStatus.RUNNING, parent.id
        await s.flush()
        new_parent, again_id = parent.id, again.id
    # Matched by submission order: ``again`` (the older row) to d1, ``bad`` to d2.
    results = {
        "d1": ({"id": "d1"}, Outcome("finished", SimStatus.CANCELLED)),
        "d2": ({"id": "d2"}, Outcome("finished", SimStatus.ERROR, message=REFUSED)),
    }
    await _expand(db, new_parent, results)

    async with db.session() as s:
        row = await s.get(SimulationRecord, again_id)
    assert row is not None
    assert row.status == SimStatus.CANCELLED
    assert row.orphan_requeues == 1


async def test_completed_siblings_of_a_refused_child_are_kept(db) -> None:
    parent_id = await _batch(db, children=2)
    results = {
        "c1": ({"id": "c1"}, Outcome("finished", SimStatus.ERROR, message=REFUSED)),
        "c2": ({"id": "c2"}, Outcome("finished", SimStatus.COMPLETE, alpha_id="A2")),
    }
    await _expand(db, parent_id, results)

    statuses = [c.status for c in await _children(db)]
    assert statuses == [SimStatus.ERROR, SimStatus.COMPLETE]


async def test_cancelled_batch_is_not_queued_again(db) -> None:
    """A batch the user cancelled ends CANCELLED on BRAIN, not ERROR: nothing goes again."""
    parent_id = await _batch(db, children=2, platform_status="CANCELLED")
    cancelled = Outcome("finished", SimStatus.CANCELLED)
    await _expand(db, parent_id, {c: ({"id": c}, cancelled) for c in ("c1", "c2")})

    assert {c.status for c in await _children(db)} == {SimStatus.CANCELLED}


async def test_children_are_matched_in_the_order_the_batch_sent_them(db) -> None:
    """The packer can send a newer row first; BRAIN lists children in send order, not id order."""
    parent_id = await _batch(db, children=2)
    first, second = await _children(db)
    async with db.session() as s:
        parent = await s.get(SimulationRecord, parent_id)
        assert parent is not None
        parent.payload = {"children": [second.payload, first.payload]}
    refused = Outcome("finished", SimStatus.ERROR, message=REFUSED)
    results = {
        "c1": ({"id": "c1"}, refused),
        "c2": ({"id": "c2"}, Outcome("finished", SimStatus.CANCELLED)),
    }
    await _expand(db, parent_id, results)

    first, second = await _children(db)
    # c1 was the second row, sent first: it is the one refused, and the older row goes again.
    assert second.status == SimStatus.ERROR
    assert REFUSED in (second.message or "")
    assert first.status == SimStatus.QUEUED
