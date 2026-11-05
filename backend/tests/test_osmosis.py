"""Osmosis: a valid allocation per scope, exactly 100,000 points over at least ten Alphas."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np

from alpha_harness.agents import osmosis


def _series(seed: int, drift: float, *, turnover: float = 0.1, base=None) -> dict:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, 800) if base is None else base + rng.normal(0, 0.05, 800)
    start = date(2021, 1, 4)
    return {start + timedelta(days=i): (float(drift + noise[i]), turnover) for i in range(800)}


def _pool(n: int, drift: float = 0.15) -> dict:
    return {f"a{i}": _series(i, drift) for i in range(n)}


def test_a_scope_gets_exactly_the_points_over_at_least_ten():
    series = _pool(14)
    scope = osmosis.allocate("USA", 1, list(series), series, {})
    held = [p for p in scope.picks if p.points]
    assert sum(p.points for p in scope.picks) == osmosis.POINTS
    assert len(held) >= osmosis.MIN_ALPHAS
    assert max(p.points for p in held) <= osmosis.MAX_SHARE * osmosis.POINTS + 1


def test_too_few_alphas_is_no_proposal():
    series = _pool(6)
    scope = osmosis.allocate("EUR", 1, list(series), series, {})
    assert not scope.eligible and not scope.picks


def test_a_twin_is_left_out_and_the_ten_made_up():
    series = _pool(10)
    series["twin"] = _series(99, 0.15, base=np.array([v[0] for v in series["a0"].values()]))
    scope = osmosis.allocate("GLB", 1, list(series), series, {})
    roles = {p.alpha_id: p.role for p in scope.picks}
    assert sum(r == "left_out" for r in roles.values()) == 1
    assert "twin" in roles and (roles["twin"] == "left_out") != (roles["a0"] == "left_out")
    assert sum(p.points for p in scope.picks) == osmosis.POINTS


def test_weak_alphas_only_make_up_the_number():
    series = _pool(8) | {f"weak{i}": _series(50 + i, -0.05) for i in range(4)}
    scope = osmosis.allocate("USA", 1, list(series), series, {})
    roles = {p.alpha_id: p for p in scope.picks}
    fillers = [p for p in roles.values() if p.role == "filler"]
    assert len(fillers) == 2
    assert all(
        p.points < min(q.points for q in roles.values() if q.role == "chosen") for p in fillers
    )


def test_current_validity_is_read_from_brain_points():
    series = _pool(12)
    valid = osmosis.allocate("GLB", 1, list(series), series, {f"a{i}": 10_000 for i in range(10)})
    lopsided = osmosis.allocate("USA", 1, list(series), series, {"a0": 100_000, "a1": 100_000})
    assert valid.currently_valid
    assert not lopsided.currently_valid


def test_scopes_skip_ra_parents_and_put_eligible_first():
    series = _pool(12)
    members = [{"alpha_id": a, "region": "USA", "delay": 1} for a in series]
    members += [{"alpha_id": "p", "region": "ALL", "delay": 1}]
    members += [{"alpha_id": "e1", "region": "EUR", "delay": 1}]
    scopes = osmosis.plan(members, series, {})
    assert [(s.region, s.eligible) for s in scopes] == [("USA", True), ("EUR", False)]


def test_the_lock_is_the_next_sunday_night_eastern():
    friday = datetime(2026, 11, 6, 15, 0, tzinfo=UTC)
    lock = osmosis.next_lock(friday)
    assert lock == datetime(2026, 11, 9, 4, 59, tzinfo=UTC)  # Sun 8 Nov 23:59 EST
    assert osmosis.next_lock(lock + timedelta(minutes=1)) == lock + timedelta(days=7)


def _held(region: str, delay: int, points: int, alphas: int) -> osmosis.Scope:
    scope = osmosis.Scope(region, delay, alphas)
    scope.current_points, scope.current_alphas = points, alphas
    return scope


def test_the_shortlist_flags_points_that_do_not_total_100000():
    gaps = osmosis.shortlist_gaps([_held("GLB", 1, 106_000, 25)])
    assert any("106,000 points" in g for g in gaps)


def test_the_shortlist_counts_regions_not_delays():
    usa = [_held("USA", 1, 100_000, 25), _held("USA", 0, 100_000, 25), _held("GLB", 1, 100_000, 25)]
    gaps = osmosis.shortlist_gaps(usa)
    assert any("cover 2 (GLB, USA)" in g for g in gaps)
    three = usa[:1] + usa[2:] + [_held("EUR", 1, 100_000, 25)]
    assert not any("regions" in g for g in osmosis.shortlist_gaps(three))


def test_a_scope_thinner_than_the_qualified_average_is_named():
    gaps = osmosis.shortlist_gaps([_held("USA", 1, 100_000, 17)])
    assert any("USA D1 holds 17" in g for g in gaps)
    assert any("160 were above 2.0" in g for g in gaps)
