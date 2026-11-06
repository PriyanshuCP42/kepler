"""Slow BRAIN answers are polled once, survive a caller giving up, and are polled gently."""

import asyncio

import pytest

from alpha_harness.brain import client as brain_client
from alpha_harness.brain.client import BrainClient, BrainResponse
from alpha_harness.brain.errors import BrainPollTimeout
from alpha_harness.brain.jobs import BrainJobs


async def test_callers_asking_for_the_same_job_share_one_poll():
    polls: list[float] = []
    kept: list[dict] = []

    async def fetch(ceiling: float) -> dict:
        polls.append(ceiling)
        await asyncio.sleep(0.05)
        return {"max": 0.63}

    async def keep(body: dict) -> None:
        kept.append(body)

    jobs = BrainJobs(ceiling=60)
    first, second = await asyncio.gather(
        jobs.run("correlation:prod:a1", fetch, keep), jobs.run("correlation:prod:a1", fetch, keep)
    )
    assert first == second == {"max": 0.63}
    assert polls == [60] and kept == [{"max": 0.63}]
    assert not jobs.running("correlation:prod:a1")


async def test_a_caller_that_gives_up_leaves_the_job_running_and_its_answer_kept():
    kept: list[dict] = []
    release = asyncio.Event()

    async def fetch(_ceiling: float) -> dict:
        await release.wait()
        return {"max": 0.71}

    async def keep(body: dict) -> None:
        kept.append(body)

    jobs = BrainJobs(ceiling=60)
    with pytest.raises(BrainPollTimeout, match="keeps polling"):
        await jobs.run("correlation:prod:a2", fetch, keep, wait=0.01)
    assert jobs.running("correlation:prod:a2") and kept == []
    # Asked again while it runs: joined, not started again.
    later = asyncio.create_task(jobs.run("correlation:prod:a2", fetch, keep, wait=5))
    release.set()
    assert await later == {"max": 0.71}
    assert kept == [{"max": 0.71}]


async def test_a_job_that_stays_pending_is_polled_less_and_less_often(monkeypatch):
    waits: list[float] = []

    async def no_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(brain_client.asyncio, "sleep", no_sleep)
    answers = [BrainResponse(200, None, 1.0, None, None)] * 30
    answers.append(BrainResponse(200, {"max": 0.5}, None, None, None))

    class Stub:
        async def request_retrying(self, _method: str, _path: str) -> BrainResponse:
            return answers.pop(0)

    done = await BrainClient.poll(Stub(), "/alphas/a3/correlations/prod")  # type: ignore[arg-type]
    assert done.body == {"max": 0.5}
    # BRAIN asks for a poll a second; past the first few, Kepler waits longer, to a cap.
    assert waits[:3] == [1.0, 1.0, 1.0]
    assert waits[-1] == brain_client.MAX_POLL_DELAY
    assert len(waits) == 30 and sum(waits) > 30 * 4
