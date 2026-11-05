"""Osmosis: the points you allocate across your submitted Alphas, and an allocation to propose.

BRAIN builds a personal combo, Osmosis, from points allocated per scope (region x delay). Its
daily PnL sets a 1-2x multiplier on the daily base payment (the Daily Osmosis Rank), and its
after-cost Sharpe over a quarter is Combined Osmosis Performance, one of the four performances
of which the best counts towards a Genius level. The rules, from BRAIN's Osmosis guide:

- at least :data:`MIN_SCOPES` scopes, each holding exactly :data:`POINTS` points over at least
  :data:`MIN_ALPHAS` Alphas; no minimum per Alpha;
- the allocation as of Sunday 23:59 EST is used from the following Monday week;
- Combined Osmosis Performance needs a valid allocation for 6 weeks of the last 3 months.

BRAIN's own advice drives the proposal: start from your best and most distinct Alphas, keep
volatility down, prefer turnover under :data:`TURNOVER_CAP`, and drop pairs that move together.
Out-of-sample PnL is hidden from consultants, so the last backtest years stand in for it, as
they do on the Submission Desk.

Applying points writes to BRAIN, so it is done only when you ask, from the Genius page.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from . import guard

POINTS = 100_000
MIN_ALPHAS = 10
MIN_SCOPES = 3
#: Daily PnL correlation above which a second Alpha adds little: the weaker one is left out.
CORRELATED = 0.7
#: Turnover over this is held to half its weight, per BRAIN's Osmosis advice.
TURNOVER_CAP = 0.2
#: No Alpha holds more than this share of a scope: one bad month must not sink it.
MAX_SHARE = 0.2
#: An Alpha weaker than this in the recent years is used only to make up the ten.
MIN_RECENT_SHARPE = 0.5
#: Share of a filler's weight, against a chosen Alpha's.
FILLER_WEIGHT = 0.1
_EST = ZoneInfo("America/New_York")

# -- WorldQuant's full-time shortlist ----------------------------------------------------
#
# From WorldQuant's slide "Osmosis Score: Criteria to shortlist consultants for full-time
# roles" (BRAIN webinar, November 2026). To qualify: at least 10 Alphas, exactly 100,000
# points and at least 3 *regions*. Of 5,799 who qualified, Combined Osmosis Sharpe was above
# 0.5 for 480, above 1.0 for 345 and above 2.0 for 160. Qualified consultants averaged 4.20
# scopes and 20.6 Alphas per scope. 3,182 were disqualified for points not totalling
# 100,000, 804 for fewer than 3 scopes, 782 for fewer than 10 Alphas in a scope.

#: Distinct regions the shortlist asks for, on top of BRAIN's three valid scopes.
SHORTLIST_REGIONS = 3
#: Combined Osmosis Sharpe of the shortlist's top tier (160 of 5,799).
SHORTLIST_SHARPE = 2.0
#: What a qualified consultant holds on average.
PEER_SCOPES = 4.2
PEER_ALPHAS_PER_SCOPE = 20.6


def shortlist_gaps(scopes: list[Scope]) -> list[str]:
    """What stands between this allocation and WorldQuant's full-time shortlist, in plain
    words: the disqualifications first, then the averages of those who qualified."""
    gaps: list[str] = []
    for s in scopes:
        if s.current_points and s.current_points != POINTS:
            gaps.append(
                f"{s.region} D{s.delay} holds {s.current_points:,} points, not exactly "
                f"{POINTS:,}: the most common shortlist disqualification (3,182 consultants)."
            )
        if s.current_points and s.current_alphas < MIN_ALPHAS:
            gaps.append(
                f"{s.region} D{s.delay} spreads its points over {s.current_alphas} Alphas; the "
                f"shortlist needs at least {MIN_ALPHAS} in every scope."
            )
    valid = [s for s in scopes if s.currently_valid]
    regions = sorted({s.region for s in valid})
    if len(regions) < SHORTLIST_REGIONS:
        held = ", ".join(regions) if regions else "none"
        gaps.append(
            f"The full-time shortlist counts at least {SHORTLIST_REGIONS} regions; your valid "
            f"allocations cover {len(regions)} ({held}). Add a scope in a new region, not "
            "another delay of one you hold."
        )
    thin = [s for s in valid if s.current_alphas < PEER_ALPHAS_PER_SCOPE]
    if thin:
        gaps.append(
            f"Qualified consultants average {PEER_SCOPES} scopes and {PEER_ALPHAS_PER_SCOPE} "
            "Alphas per scope; "
            + ", ".join(f"{s.region} D{s.delay} holds {s.current_alphas}" for s in thin)
            + "."
        )
    gaps.append(
        f"The shortlist ranks Combined Osmosis Sharpe: of 5,799 who qualified, 160 were above "
        f"{SHORTLIST_SHARPE}. Proposals here are steadiest by recent backtest, which stands in "
        "for the out-of-sample PnL BRAIN keeps hidden."
    )
    return gaps


@dataclass(slots=True)
class Pick:
    alpha_id: str
    points: int
    recent_sharpe: float | None
    turnover: float | None
    current: int
    #: ``chosen``, ``filler`` (to make up the ten) or ``left_out``.
    role: str
    reason: str = ""


@dataclass(slots=True)
class Scope:
    region: str
    delay: int
    alphas: int
    picks: list[Pick] = field(default_factory=list)
    #: Recent-years Sharpe of the combo as proposed, and as currently allocated.
    proposed_sharpe: float | None = None
    current_sharpe: float | None = None
    current_points: int = 0
    current_alphas: int = 0

    @property
    def eligible(self) -> bool:
        return self.alphas >= MIN_ALPHAS

    @property
    def currently_valid(self) -> bool:
        return self.current_points == POINTS and self.current_alphas >= MIN_ALPHAS


def next_lock(now: datetime | None = None) -> datetime:
    """The next Sunday 23:59 US Eastern, when allocations lock (UTC)."""
    now = (now or datetime.now(UTC)).astimezone(_EST)
    days = (6 - now.weekday()) % 7
    lock = (now + timedelta(days=days)).replace(hour=23, minute=59, second=0, microsecond=0)
    if lock <= now:
        lock += timedelta(days=7)
    return lock.astimezone(UTC)


def _recent(days: dict[date, tuple[float, float]]) -> tuple[np.ndarray, float | None, float | None]:
    """The recent years' daily PnL, their annualised Sharpe and mean turnover."""
    recent = sorted(d for d in days if d >= guard.RECENT_START)
    pnl = np.array([days[d][0] for d in recent], dtype=float)
    traded = np.array([days[d][1] for d in recent], dtype=float)
    if pnl.size < 60 or not pnl.std():
        return pnl, None, None
    return pnl, float(pnl.mean() / pnl.std() * math.sqrt(252)), float(traded.mean())


def _combo_sharpe(
    series: dict[str, dict[date, tuple[float, float]]], points: dict[str, int]
) -> float | None:
    """Recent-years Sharpe of the points-weighted sum of these Alphas' daily PnL."""
    held = {a: p for a, p in points.items() if p > 0 and series.get(a)}
    if not held:
        return None
    days = sorted({d for a in held for d in series[a] if d >= guard.RECENT_START})
    if len(days) < 60:
        return None
    total = np.zeros(len(days))
    for alpha_id, p in held.items():
        total += p * np.array([series[alpha_id].get(d, (0.0, 0.0))[0] for d in days])
    return float(total.mean() / total.std() * math.sqrt(252)) if total.std() else None


def _round(weights: dict[str, float]) -> dict[str, int]:
    """Weights as integer points summing to exactly :data:`POINTS` (largest remainder)."""
    total = sum(weights.values())
    raw = {a: POINTS * w / total for a, w in weights.items()}
    points = {a: math.floor(v) for a, v in raw.items()}
    short = POINTS - sum(points.values())
    for alpha_id in sorted(raw, key=lambda a: raw[a] - points[a], reverse=True)[:short]:
        points[alpha_id] += 1
    return points


def _capped(weights: dict[str, float]) -> dict[str, float]:
    """No share over :data:`MAX_SHARE`, the excess handed to the rest in proportion."""
    out = dict(weights)
    for _ in range(len(out)):
        total = sum(out.values())
        over = {a for a, w in out.items() if w > MAX_SHARE * total + 1e-12}
        if not over or len(out) * MAX_SHARE < 1:
            break
        under = [a for a in out if a not in over]
        excess = sum(out[a] - MAX_SHARE * total for a in over)
        for a in over:
            out[a] = MAX_SHARE * total
        base = sum(out[a] for a in under)
        for a in under:
            out[a] += excess * out[a] / base if base else excess / len(under)
    return out


def allocate(
    region: str,
    delay: int,
    members: list[str],
    series: dict[str, dict[date, tuple[float, float]]],
    current: dict[str, int],
) -> Scope:
    """A proposed allocation for one scope. Pure.

    Alphas are taken strongest first by recent Sharpe; one that moves with an Alpha already
    taken (daily PnL correlation over :data:`CORRELATED`) is left out. Each kept Alpha weighs
    its recent Sharpe over its PnL volatility (risk-balanced, so a loud Alpha cannot dominate),
    halved over :data:`TURNOVER_CAP` turnover, capped at :data:`MAX_SHARE`. When fewer than
    :data:`MIN_ALPHAS` remain, the best of the rest make up the number at a small weight.
    """
    scope = Scope(region, delay, len(members))
    scope.current_points = sum(current.get(a, 0) for a in members)
    scope.current_alphas = sum(1 for a in members if current.get(a, 0) > 0)
    scope.current_sharpe = _combo_sharpe(series, {a: current.get(a, 0) for a in members})
    if not scope.eligible:
        return scope

    measured: dict[str, tuple[np.ndarray, float | None, float | None]] = {
        a: _recent(series.get(a) or {}) for a in members
    }
    ranked = sorted(members, key=lambda a: measured[a][1] or float("-inf"), reverse=True)
    chosen: list[str] = []
    reasons: dict[str, str] = {}
    for alpha_id in ranked:
        pnl, sharpe, _ = measured[alpha_id]
        if sharpe is None:
            reasons[alpha_id] = "No recent PnL stored."
            continue
        if sharpe < MIN_RECENT_SHARPE:
            reasons[alpha_id] = f"Recent Sharpe {sharpe:.2f} is under {MIN_RECENT_SHARPE}."
            continue
        twin = next(
            (
                b
                for b in chosen
                if pnl.size == measured[b][0].size
                and float(np.corrcoef(pnl, measured[b][0])[0, 1]) > CORRELATED
            ),
            None,
        )
        if twin is not None:
            reasons[alpha_id] = f"Moves with {twin} (correlation over {CORRELATED})."
            continue
        chosen.append(alpha_id)

    fillers = [a for a in ranked if a not in chosen and measured[a][1] is not None][
        : max(0, MIN_ALPHAS - len(chosen))
    ]
    weights: dict[str, float] = {}
    for alpha_id in chosen:
        pnl, sharpe, turnover = measured[alpha_id]
        weight = max(sharpe or 0.0, 0.0) / float(pnl.std())
        if turnover is not None and turnover > TURNOVER_CAP:
            weight *= 0.5
        weights[alpha_id] = weight
    floor = FILLER_WEIGHT * (min(weights.values()) if weights else 1.0)
    for alpha_id in fillers:
        weights[alpha_id] = floor
    if len(weights) < MIN_ALPHAS:
        return scope  # too few with any PnL to make a valid scope

    points = _round(_capped(weights))
    scope.proposed_sharpe = _combo_sharpe(series, points)
    for alpha_id in ranked:
        _, sharpe, turnover = measured[alpha_id]
        role = "chosen" if alpha_id in chosen else "filler" if alpha_id in fillers else "left_out"
        reason = (
            "Makes up the ten Alphas a scope needs, at a small weight." if role == "filler" else ""
        )
        scope.picks.append(
            Pick(
                alpha_id,
                points.get(alpha_id, 0),
                round(sharpe, 2) if sharpe is not None else None,
                round(turnover, 3) if turnover is not None else None,
                current.get(alpha_id, 0),
                role,
                reasons.get(alpha_id, reason),
            )
        )
    return scope


def plan(
    members: list[dict[str, Any]],
    series: dict[str, dict[date, tuple[float, float]]],
    current: dict[str, int],
) -> list[Scope]:
    """Every scope you have submitted to, eligible ones first, strongest proposal first."""
    by_scope: dict[tuple[str, int], list[str]] = {}
    for row in members:
        region, delay = row.get("region"), row.get("delay")
        if not region or region == "ALL" or delay is None:
            continue  # BRAIN takes RA children, never the parent
        by_scope.setdefault((str(region), int(delay)), []).append(str(row["alpha_id"]))
    scopes = [allocate(r, d, ids, series, current) for (r, d), ids in by_scope.items()]
    return sorted(
        scopes,
        key=lambda s: (s.eligible, s.proposed_sharpe or float("-inf"), s.alphas),
        reverse=True,
    )
