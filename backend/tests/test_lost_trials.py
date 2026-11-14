"""A task whose simulations BRAIN lost must still finish: old open trials are closed."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select

from alpha_harness.db.models import Study, StudyStatus, Trial, TrialState, utcnow
from alpha_harness.db.sqlite import Database
from alpha_harness.labs import scheduler

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
async def db(tmp_path: Path):
    database = Database(tmp_path / "t.db")
    await database.create_all()
    yield database
    await database.dispose()


async def test_only_old_open_trials_are_closed_as_lost(db) -> None:
    now = utcnow()
    async with db.session() as session:
        study = Study(name="t", template_source="x", sampler="template", status=StudyStatus.RUNNING)
        session.add(study)
        await session.flush()
        rows = [
            (0, TrialState.RUNNING, now - timedelta(hours=5)),  # lost
            (1, TrialState.QUEUED, now - timedelta(hours=3)),  # lost
            (2, TrialState.RUNNING, now - timedelta(minutes=10)),  # may still come back
            (3, TrialState.COMPLETE, now - timedelta(hours=5)),  # finished: untouched
        ]
        session.add_all(
            Trial(study_id=study.id, number=n, state=state, created_at=at) for n, state, at in rows
        )
        study_id = study.id

    closed = await scheduler.close_lost(SimpleNamespace(db=db), study_id)

    assert closed == 2
    async with db.session() as session:
        result = await session.execute(
            select(Trial.number, Trial.state).where(Trial.study_id == study_id)
        )
        states = dict(result.tuples().all())
        messages = list(
            await session.scalars(select(Trial.message).where(Trial.state == TrialState.FAIL))
        )
    assert states == {
        0: TrialState.FAIL,
        1: TrialState.FAIL,
        2: TrialState.RUNNING,
        3: TrialState.COMPLETE,
    }
    assert messages == [scheduler.LOST, scheduler.LOST]
