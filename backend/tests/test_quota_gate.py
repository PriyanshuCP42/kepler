"""QuotaGate: atomic reservation, ceilings, idempotency, uncertainty, recovery. No BRAIN access."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import update

from alpha_harness.db.models import QuotaReservation, utcnow
from alpha_harness.db.sqlite import Database
from alpha_harness.engine.budget import Denied, Outcome, QuotaGate

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
async def db(tmp_path: Path):
    database = Database(tmp_path / "t.db")
    await database.create_all()
    yield database
    await database.dispose()


def gate(db: Database, **kw) -> QuotaGate:
    args = {"hard_cap": 5000, "operating_ceiling": 1000, "protected_reserve": 100} | kw
    return QuotaGate(db, **args)


async def test_reserves_up_to_limit_then_refuses(db) -> None:
    g = gate(db)  # usable = 900
    await g.authorize(token="a", owner="x", units=900)
    with pytest.raises(Denied) as e:
        await g.authorize(token="b", owner="x", units=1)
    assert e.value.reason == "over_budget"


async def test_reserve_is_protected_unless_asked(db) -> None:
    g = gate(db)
    await g.authorize(token="a", owner="x", units=900)
    await g.authorize(token="b", owner="human", units=100, use_reserve=True)
    assert await g.available(use_reserve=True) == 0


@pytest.mark.parametrize("units", [0, -5])
async def test_invalid_units_fail_closed(db, units) -> None:
    with pytest.raises(Denied) as e:
        await gate(db).authorize(token="a", owner="x", units=units)
    assert e.value.reason == "invalid"


async def test_ceiling_cannot_exceed_hard_cap(db) -> None:
    with pytest.raises(ValueError):
        gate(db, operating_ceiling=6000)


async def test_concurrent_requests_never_oversubscribe(db) -> None:
    g = gate(db, operating_ceiling=500, protected_reserve=0)

    async def ask(i: int) -> bool:
        try:
            await g.authorize(token=f"t{i}", owner="x", units=10)
        except Denied:
            return False
        return True

    results = await asyncio.gather(*(ask(i) for i in range(200)))
    assert sum(results) == 50
    assert await g.committed() == 500


async def test_same_token_is_idempotent(db) -> None:
    g = gate(db)
    first = await g.authorize(token="a", owner="x", units=10)
    again = await g.authorize(token="a", owner="x", units=10)
    assert first.fresh and not again.fresh
    assert await g.committed() == 10


async def test_release_frees_units_and_token_cannot_be_reused(db) -> None:
    g = gate(db)
    await g.authorize(token="a", owner="x", units=10)
    await g.settle("a", Outcome.RELEASED)
    assert await g.committed() == 0
    with pytest.raises(Denied) as e:
        await g.authorize(token="a", owner="x", units=10)
    assert e.value.reason == "released"


async def test_confirmed_charges_actual_units(db) -> None:
    g = gate(db)
    await g.authorize(token="a", owner="x", units=10)
    await g.settle("a", Outcome.CONFIRMED, actual_units=4)
    assert await g.committed() == 4


async def test_uncertain_keeps_counting_and_can_be_reconciled(db) -> None:
    g = gate(db)
    await g.authorize(token="a", owner="x", units=10)
    await g.settle("a", Outcome.UNCERTAIN, note="timeout")
    assert await g.committed() == 10
    await g.settle("a", Outcome.CONFIRMED, actual_units=10)
    assert (await g.status())["confirmed"] == 10
    with pytest.raises(Denied):
        await g.settle("a", Outcome.RELEASED)  # confirmed spend cannot be un-spent


async def test_observed_usage_floors_the_ledger(db) -> None:
    g = gate(db)  # usable 900
    with pytest.raises(Denied):
        await g.authorize(token="a", owner="x", units=10, observed_used=895)
    await g.authorize(token="b", owner="x", units=5, observed_used=895)


async def test_owner_budget(db) -> None:
    g = gate(db, owner_budgets={"agent": 20})
    await g.authorize(token="a", owner="agent", units=20)
    with pytest.raises(Denied) as e:
        await g.authorize(token="b", owner="agent", units=1)
    assert e.value.reason == "owner_budget"
    await g.authorize(token="c", owner="other", units=50)


async def test_restart_turns_stale_reservations_uncertain_not_free(db) -> None:
    g = gate(db, stale_seconds=60)
    await g.authorize(token="a", owner="x", units=10)
    async with db.session() as s:
        await s.execute(update(QuotaReservation).values(created_at=utcnow() - timedelta(minutes=5)))
    assert await g.recover() == 1
    assert await g.committed() == 10
    assert (await g.status())["uncertain"] == 10


async def test_survives_new_gate_instance(db) -> None:
    await gate(db).authorize(token="a", owner="x", units=10)
    assert await gate(db).committed() == 10


# --- the send path ----------------------------------------------------------------------


def _request(kind: str = "REGULAR"):
    from alpha_harness.brain.schemas import SimulationRequest, SimulationSettings

    return SimulationRequest(
        type=kind,
        regular="rank(close)",
        settings=SimulationSettings(region="USA", delay=1, universe="TOP3000"),
    )


async def test_units_region_agnostic_and_batch(db) -> None:
    g = gate(db)
    assert g.units_for(_request()) == 1
    assert g.units_for(_request("REGION_AGNOSTIC")) == 4
    assert g.units_for([_request(), _request(), _request("REGION_AGNOSTIC")]) == 6


async def test_guard_confirms_on_success(db) -> None:
    g = gate(db)
    sent = []

    async def send(p):
        sent.append(p)
        return "ok"

    assert await g.guard([_request(), _request()], send) == "ok"
    assert await g.committed() == 2 and len(sent) == 1


async def test_guard_refuses_without_sending_at_the_ceiling(db) -> None:
    from alpha_harness.brain.errors import BrainDailyLimitReached

    g = gate(db, operating_ceiling=5, protected_reserve=0)
    called = False

    async def send(p):
        nonlocal called
        called = True

    with pytest.raises(BrainDailyLimitReached):
        await g.guard([_request()] * 6, send)
    assert not called
    assert await g.committed() == 0


async def test_guard_releases_on_definite_failure_and_holds_on_ambiguous(db) -> None:
    from alpha_harness.brain.errors import BrainServiceUnavailable, BrainTransportError

    g = gate(db)

    async def definite(p):
        raise BrainServiceUnavailable("503", status=503)

    with pytest.raises(BrainServiceUnavailable):
        await g.guard(_request(), definite)
    assert await g.committed() == 0

    async def ambiguous(p):
        raise BrainTransportError("read timed out", maybe_delivered=True)

    with pytest.raises(BrainTransportError):
        await g.guard(_request(), ambiguous)
    assert (await g.status())["uncertain"] == 1


async def test_guard_holds_on_unexpected_error(db) -> None:
    g = gate(db)

    async def boom(p):
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        await g.guard(_request(), boom)
    assert (await g.status())["uncertain"] == 1


async def test_endpoint_cannot_be_sent_around_the_gate(db) -> None:
    from alpha_harness.brain.endpoints import BrainEndpoints

    class Client:
        posts = 0

        async def request(self, *a, **k):
            Client.posts += 1
            raise AssertionError("must not reach the network in tests")

    g = gate(db, operating_ceiling=1, protected_reserve=0)
    ep = BrainEndpoints(Client())  # type: ignore[arg-type]
    ep.quota_guard = lambda payload, send: g.guard(payload, send)
    from alpha_harness.brain.errors import BrainDailyLimitReached

    with pytest.raises(BrainDailyLimitReached):
        await ep.create_simulation([_request(), _request()])
    assert Client.posts == 0


async def test_brain_count_replaces_a_higher_local_count(db) -> None:
    g = gate(db)  # usable = 900
    await g.authorize(token="a", owner="x", units=900)
    # BRAIN, read after that send, says only 300 were charged: errors cost nothing there.
    read_at = utcnow()
    await g.authorize(token="b", owner="x", units=500, platform_used=(300, read_at))
    # 300 by BRAIN plus 500 reserved since: 100 left.
    with pytest.raises(Denied):
        await g.authorize(token="c", owner="x", units=101, platform_used=(300, read_at))
    await g.authorize(token="d", owner="x", units=100, platform_used=(300, read_at))
