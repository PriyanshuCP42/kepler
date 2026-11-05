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


def _series(seed: int, drift: float = 0.15) -> dict:
    noise = np.random.default_rng(seed).normal(0, 1, 800)
    start = date(2021, 1, 4)
    return {start + timedelta(days=i): (float(drift + noise[i]), 0.1) for i in range(800)}


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
        return {a: _series(int(a[1:]), -0.3 if a.startswith("w") else 0.15) for a in ids}


class _Endpoints:
    """BRAIN: holds 1 or more points, clears on null, refuses 0 and non-compensated Alphas."""

    def __init__(
        self, old: set[str], flaky: set[str] | None = None, held: dict[str, int] | None = None
    ) -> None:
        self.old = old
        self.flaky = flaky or set()
        self.points: dict[str, int] = dict(held or {})

    async def osmosis_points(self) -> dict[str, int]:
        return dict(self.points)

    async def update_alpha(self, alpha_id: str, properties: dict) -> dict:
        value = properties["osmosisPoints"]
        if alpha_id in self.old:
            raise BrainValidationError(
                f"PATCH /alphas/{alpha_id}: rejected by the platform",
                fields={
                    "osmosisPoints": ["Cannot update Osmosis points for non-compensated alpha."]
                },
            )
        if value is not None and value < 1:
            raise BrainValidationError(
                f"PATCH /alphas/{alpha_id}: rejected by the platform",
                fields={"osmosisPoints": ["Ensure this value is greater than or equal to 1."]},
            )
        if alpha_id in self.flaky:
            raise BrainTransportError(f"PATCH /alphas/{alpha_id}: connection reset")
        if value is None:
            self.points.pop(alpha_id, None)
        else:
            self.points[alpha_id] = value
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


def test_an_alpha_left_out_is_cleared_with_null_not_zero():
    # "w0" loses money recently, so the proposal leaves it out; it holds points today.
    brain = _Endpoints(old=set(), held={"w0": 2_000, "a0": 98_000})
    metadata = _Metadata()
    state = _state(brain, metadata)
    state.alphas.ids.append("w0")

    applied = asyncio.run(genius.osmosis_apply(genius.OsmosisApply(scopes=["USA/1"]), state))

    assert not applied.failed and not metadata.refused
    assert "w0" not in brain.points
    assert sum(brain.points.values()) == osmosis.POINTS
