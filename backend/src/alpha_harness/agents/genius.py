"""The Genius tracker: BRAIN's real performance bars against the Submission Desk's estimate.

BRAIN scores Combined Alpha Performance on the days after the backtest, which it keeps hidden,
so the Desk estimates it from the last backtest years (:data:`.guard.RECENT_YEARS`). Whether
that estimate is any good can only be told against BRAIN's own figures. They are published on
the Genius page alone, so you type them in now and then; this lines each reading up with what
the estimate said that day, worked out again from the alphas submitted by then.

Between two readings, the alphas submitted in between moved both: BRAIN's figure by what BRAIN
says, the estimate by what Kepler says. Once there are :data:`MIN_PAIRS` such steps, how often
the two moved the same way, and how closely, says how far to trust the Desk's ranking.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from sqlalchemy import select

from ..brain.platform import POWER_POOL_CLASS
from ..db.models import GeniusReading
from ..tools import portfolio
from ..vault.store import json_list
from . import guard

#: Genius's bars by level: Expert, Master, Grandmaster.
BARS: dict[str, tuple[float, float, float]] = {
    "signals": (20, 120, 220),
    "pyramids": (10, 30, 60),
    "combined": (0.5, 1.0, 2.0),
    "power_pool": (0.5, 1.0, 2.0),
}
LEVELS = ("Expert", "Master", "Grandmaster")
#: Steps between readings needed before the estimate is judged.
MIN_PAIRS = 3


@dataclass(slots=True)
class Member:
    alpha_id: str
    submitted: date | None
    power: bool


@dataclass(slots=True)
class Estimate:
    combined: float | None
    power_pool: float | None
    alphas: int
    #: The whole pool's recent figures, which its combined-performance score is made of.
    figures: dict[str, Any] | None = None


@dataclass(slots=True)
class Step:
    """From one reading to the next: what was submitted, and how each figure moved."""

    since: date
    until: date
    submitted: list[str]
    brain_combined: float | None
    brain_power_pool: float | None
    estimate_combined: float | None
    estimate_power_pool: float | None
    #: How much the estimate's combined-performance score moved, as a share: Sharpe, Fitness,
    #: margin, turnover and drawdown together (see :func:`..tools.portfolio.quality_change`).
    estimate_quality: float | None = None


@dataclass(slots=True)
class Agreement:
    pairs: int
    #: Steps where BRAIN's figure and the estimate moved the same way.
    same_direction: int = 0
    #: Pearson correlation of the two changes; ``None`` below :data:`MIN_PAIRS` or when flat.
    correlation: float | None = None
    verdict: str = ""


@dataclass(slots=True)
class Report:
    readings: list[dict[str, Any]] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    agreement: Agreement = field(default_factory=lambda: Agreement(0))
    today: Estimate | None = None


def level_of(metric: str, value: float | None) -> str | None:
    """The highest level this one figure clears, or ``None`` below Expert."""
    if value is None:
        return None
    reached = [name for name, bar in zip(LEVELS, BARS[metric], strict=True) if value >= bar]
    return reached[-1] if reached else None


def _quality(before: dict[str, Any] | None, after: dict[str, Any] | None) -> float | None:
    found = portfolio.quality_change(before, after)
    return float(found["total"]) if found else None


def _change(after: float | None, before: float | None) -> float | None:
    return None if after is None or before is None else after - before


def agreement(steps: list[Step]) -> Agreement:
    """How well the estimate's changes track BRAIN's, over the steps where both are known."""
    # The combined-performance score where known, which is what the Desk ranks by.
    moves = [
        (s.estimate_quality if s.estimate_quality is not None else s.estimate_combined, s)
        for s in steps
    ]
    # A step where neither moved says nothing about whether they move together.
    pairs = [
        (move, s.brain_combined)
        for move, s in moves
        if move is not None and s.brain_combined is not None and (move or s.brain_combined)
    ]
    found = Agreement(len(pairs))
    found.same_direction = sum(1 for e, b in pairs if e * b > 0)
    if len(pairs) < MIN_PAIRS:
        found.verdict = (
            f"{len(pairs)} of {MIN_PAIRS} steps so far: add a reading every few days, "
            "especially after submitting, to judge the estimate."
        )
        return found
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx > 0 and sy > 0:
        found.correlation = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / (
            sx * sy
        )
    share = found.same_direction / len(pairs)
    if found.correlation is not None and found.correlation >= 0.5 and share >= 0.6:
        found.verdict = "The estimate moves with BRAIN's figure: trust the Desk's ranking."
    elif share >= 0.5:
        found.verdict = (
            "The estimate gets the direction right about half the time or more: use it as a "
            "rough guide."
        )
    else:
        found.verdict = (
            "The estimate often moves against BRAIN's figure: lean on BRAIN's own before-and-"
            "after on each alpha's page, and submit fewer, stronger alphas."
        )
    return found


def estimate(
    series: dict[str, dict[date, tuple[float, float]]],
    meta: dict[str, dict[str, Any]],
    members: list[Member],
    on: date | None,
) -> Estimate:
    """The Desk's estimate of both bars from the alphas submitted by the end of ``on``."""
    held = [m for m in members if on is None or (m.submitted is not None and m.submitted <= on)]
    ids = [m.alpha_id for m in held]
    power = [m.alpha_id for m in held if m.power]
    whole = portfolio.combined(series, meta, ids, guard.RECENT_START)["recent"]
    pooled = portfolio.combined(series, meta, power, guard.RECENT_START)["recent"]
    return Estimate(
        combined=whole["sharpe"] if whole else None,
        power_pool=pooled["sharpe"] if pooled else None,
        alphas=len(ids),
        figures=whole,
    )


def build(
    readings: list[GeniusReading],
    members: list[Member],
    series: dict[str, dict[date, tuple[float, float]]],
    meta: dict[str, dict[str, Any]],
) -> Report:
    """Every reading with the estimate on its day, and each step between two readings."""
    ordered = sorted(readings, key=lambda r: (r.read_on, r.id))
    report = Report(today=estimate(series, meta, members, None))
    estimates = [estimate(series, meta, members, r.read_on) for r in ordered]
    for reading, found in zip(ordered, estimates, strict=True):
        report.readings.append(
            {
                "id": reading.id,
                "readOn": reading.read_on.isoformat(),
                "combined": reading.combined,
                "powerPool": reading.power_pool,
                "signals": reading.signals,
                "pyramids": reading.pyramids,
                "note": reading.note,
                "estimateCombined": found.combined,
                "estimatePowerPool": found.power_pool,
                "alphas": found.alphas,
            }
        )
    for i in range(1, len(ordered)):
        a, b, ea, eb = ordered[i - 1], ordered[i], estimates[i - 1], estimates[i]
        if a.read_on == b.read_on:
            continue  # two readings of one day: a correction, not a step
        report.steps.append(
            Step(
                since=a.read_on,
                until=b.read_on,
                submitted=[
                    m.alpha_id
                    for m in members
                    if m.submitted is not None and a.read_on < m.submitted <= b.read_on
                ],
                brain_combined=_change(b.combined, a.combined),
                brain_power_pool=_change(b.power_pool, a.power_pool),
                estimate_combined=_change(eb.combined, ea.combined),
                estimate_power_pool=_change(eb.power_pool, ea.power_pool),
                estimate_quality=_quality(ea.figures, eb.figures),
            )
        )
    report.agreement = agreement(report.steps)
    return report


def members_of(rows: list[dict[str, Any]]) -> list[Member]:
    out: list[Member] = []
    for row in rows:
        when = row.get("date_submitted")
        if isinstance(when, datetime):
            day: date | None = when.date()
        else:
            day = when if isinstance(when, date) else None
        out.append(
            Member(
                alpha_id=str(row["alpha_id"]),
                submitted=day,
                power=POWER_POOL_CLASS in json_list(row.get("classifications")),
            )
        )
    return out


async def report(state: Any) -> Report:
    """Read every reading and the submitted pool, and build the report off the event loop."""
    async with state.db.session() as session:
        readings = list(await session.scalars(select(GeniusReading)))
    rows = await state.alphas.submitted_members()
    members = members_of(rows)
    ids = [m.alpha_id for m in members]
    series = await state.alphas.series(ids)
    meta = await state.alphas.by_ids(ids)
    return await asyncio.to_thread(build, readings, members, series, meta)
