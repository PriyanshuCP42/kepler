"""Setting Osmosis points when BRAIN refuses some Alphas: they are left out and the scope set
again, so its points still add up to a valid allocation."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np

from alpha_harness.agents import osmosis
from alpha_harness.api import genius
from alpha_harness.brain.errors import BrainTransportError, BrainValidationError


def _series(seed: int) -> dict:
    noise = np.random.default_rng(seed).normal(0, 1, 800)
    start = date(2021, 1, 4)
    return {start + timedelta(days=i): (float(0.15 + noise[i]), 0.1) for i in range(800)}


class _Metadata:
    def __init__(self) -> None:
        self.refused: set[str] = set()

    async def osmosis_refused(self) -> set[str]:
        return set(self.refused)

    async def add_osmosis_refused(self, alpha_ids: list[str]) -> None:
        self.refused |= set(alpha_ids)


class _Alphas:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    async def submitted_members(self) -> list[dict]:
        return [{"alpha_id": a, "region": "USA", "delay": 1} for a in self.ids]

    async def series(self, ids: list[str]) -> dict:
        return {a: _series(int(a[1:])) for a in ids}


class _Endpoints:
    """BRAIN: holds points, but refuses them on the old Alphas."""

    def __init__(self, old: set[str], flaky: set[str] | None = None) -> None:
        self.old = old
        self.flaky = flaky or set()
        self.points: dict[str, int] = {}

    async def osmosis_points(self) -> dict[str, int]:
        return dict(self.points)

    async def update_alpha(self, alpha_id: str, properties: dict) -> dict:
        if alpha_id in self.old:
            raise BrainValidationError(
                f"PATCH /alphas/{alpha_id}: rejected by the platform", fields={}
            )
        if alpha_id in self.flaky:
            raise BrainTransportError(f"PATCH /alphas/{alpha_id}: connection reset")
        self.points[alpha_id] = properties["osmosisPoints"]
        return {}


def _state(brain: _Endpoints, metadata: _Metadata) -> SimpleNamespace:
    return SimpleNamespace(
        auth=SimpleNamespace(session=SimpleNamespace(authenticated=True)),
        alphas=_Alphas([f"a{i}" for i in range(13)]),
        endpoints=brain,
        metadata=metadata,
    )


def test_refused_alphas_are_left_out_and_the_scope_still_adds_up():
    brain = _Endpoints(old={"a0", "a1"})
    metadata = _Metadata()
    state = _state(brain, metadata)

    applied = asyncio.run(genius.osmosis_apply(genius.OsmosisApply(scopes=["USA/1"]), state))

    assert set(applied.failed) <= {"a0", "a1"} and applied.failed
    assert metadata.refused == set(applied.failed)
    held = {a: p for a, p in brain.points.items() if p}
    assert sum(held.values()) == osmosis.POINTS
    assert len(held) >= osmosis.MIN_ALPHAS
    assert not held.keys() & {"a0", "a1"}
    usa = next(s for s in applied.view.scopes if s.region == "USA")
    assert usa.currently_valid


def test_a_network_blip_is_reported_but_not_remembered_as_refused():
    brain = _Endpoints(old=set(), flaky={"a2"})
    metadata = _Metadata()

    applied = asyncio.run(
        genius.osmosis_apply(genius.OsmosisApply(scopes=["USA/1"]), _state(brain, metadata))
    )

    assert applied.failed == ["a2"]
    assert not metadata.refused
