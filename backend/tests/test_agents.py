"""The research agents: the Critic's rules, the Conductor's watchdog, the bus, the families."""

import asyncio
import json
from datetime import date, timedelta

import numpy as np

from alpha_harness.agents import families
from alpha_harness.agents.bus import AgentBus
from alpha_harness.agents.conductor import is_infrastructure
from alpha_harness.agents.critic import judge
from alpha_harness.labs.ga import series_of


def _checks(*failing: tuple[str, float, float]) -> str:
    passing = [{"name": "LOW_SHARPE", "result": "PASS", "limit": 1.25, "value": 1.6}]
    return json.dumps(
        passing + [{"name": n, "result": "FAIL", "limit": lim, "value": v} for n, lim, v in failing]
    )


def _row(**over):
    row = {
        "alpha_id": "a1",
        "sharpe": 1.6,
        "fitness": 1.1,
        "margin": 0.0009,
        "train_sharpe": 1.7,
        "test_sharpe": 1.3,
        "operator_count": 4,
        "checks": _checks(),
        "simulation_mode": None,
    }
    return {**row, **over}


class TestCritic:
    def test_a_robust_alpha_passes_and_waits_for_the_pool_gate(self):
        result = judge(_row())
        assert result.verdict == "pass"
        assert result.gates["pool"]["ok"] is None
        assert 0 < result.score <= 100

    def test_decay_out_of_sample_fails(self):
        result = judge(_row(train_sharpe=2.0, test_sharpe=0.2))
        assert result.verdict == "fail"
        assert result.gates["decay"]["ok"] is False

    def test_moderate_decay_is_a_near_miss(self):
        assert judge(_row(train_sharpe=2.0, test_sharpe=0.9)).verdict == "near_miss"

    def test_a_check_missed_narrowly_is_a_near_miss(self):
        result = judge(_row(checks=_checks(("HIGH_TURNOVER", 0.7, 0.78))))
        assert result.verdict == "near_miss"

    def test_a_check_missed_widely_fails(self):
        assert judge(_row(checks=_checks(("HIGH_TURNOVER", 0.7, 1.5)))).verdict == "fail"

    def test_thin_margin_is_flagged(self):
        result = judge(_row(margin=0.00015))
        assert result.gates["cost"]["ok"] is False
        assert result.verdict == "fail"

    def test_quick_mode_is_never_submittable(self):
        assert judge(_row(simulation_mode="QUICK")).verdict == "fail"

    def test_no_checks_is_unjudged(self):
        assert judge(_row(checks=None)).verdict == "unjudged"

    def test_simpler_scores_higher(self):
        assert judge(_row(operator_count=3)).score > judge(_row(operator_count=8)).score


def test_watchdog_restarts_only_infrastructure_faults():
    assert is_infrastructure("(sqlite3.OperationalError) database is locked")
    assert is_infrastructure("QueuePool limit of size 5 overflow 10 reached")
    assert is_infrastructure("BRAIN answered 503 Service Unavailable")
    assert not is_infrastructure("The template is empty.")
    assert not is_infrastructure(None)


async def test_bus_coalesces_a_burst_into_one_parcel():
    sent = []

    class Hub:
        async def broadcast(self, topic, payload, *, replay=True):
            sent.append((topic, payload))

    bus = AgentBus(Hub())
    for _ in range(5):
        bus.handoff("writers", "validator", "candidates", 10)
    bus.handoff("writers", "nobody", "candidates", 3)  # unknown agents are ignored
    await asyncio.sleep(2.0)
    parcels = [p for t, p in sent if t == "agent-handoff"]
    assert len(parcels) == 1
    assert parcels[0]["count"] == 50
    snapshot = bus.snapshot()
    writers = next(a for a in snapshot["agents"] if a["id"] == "writers")
    assert writers["state"] == "working"
    await bus.stop()


def _days(values):
    start = date(2020, 1, 1)
    return {start + timedelta(days=i): float(v) for i, v in enumerate(values)}


def test_family_clerk_places_repeats_and_new_ideas():
    rng = np.random.default_rng(0)
    base = rng.normal(size=600)
    pool_member = base + rng.normal(scale=0.2, size=600)
    clerk = families.FamilyClerk.__new__(families.FamilyClerk)
    clerk._leaders = {}
    pool = {"submitted": series_of(_days(pool_member))}

    repeat = clerk._place("r", _days(base), pool)
    assert repeat["verdict"] == "near_miss"

    fresh = rng.normal(size=600)
    first = clerk._place("f1", _days(fresh), pool)
    assert first["family_id"] == "f1"
    sibling = clerk._place("f2", _days(fresh + rng.normal(scale=0.1, size=600)), pool)
    assert sibling["family_id"] == "f1"
    other = clerk._place("o", _days(rng.normal(size=600)), pool)
    assert other["family_id"] == "o"
