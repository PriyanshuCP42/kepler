"""Consultant challenges: contests BRAIN announces in webinars, not in its competitions list.

BRAIN's competitions page carries rank and tags, but a challenge like the USA/GLB November
2026 Theme Challenge is only announced on a slide. Its rules live here, by hand, and where you
stand is worked out from what Kepler already holds: the submitted Alphas (with BRAIN's
production correlation for each) and the simulations sent on each BRAIN day. Nothing here
calls BRAIN, so the standing still shows while the session is signed out, which is exactly
when the daily simulation rule is at risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from ..brain.platform import PLATFORM_TIMEZONE

PLATFORM_TZ = ZoneInfo(PLATFORM_TIMEZONE)

#: Why a submitted Alpha does not count.
OUTSIDE_WINDOW = "OUTSIDE_WINDOW"
WRONG_REGION = "WRONG_REGION"
PROD_TOO_HIGH = "PROD_CORRELATION"
NOT_MEASURED = "PROD_NOT_MEASURED"


@dataclass(frozen=True, slots=True)
class Challenge:
    id: str
    name: str
    #: First and last BRAIN day (America/New_York), both inclusive.
    start: date
    end: date
    regions: frozenset[str]
    #: An Alpha counts only with a production correlation below this.
    max_prod: float = 0.7
    min_alphas: int = 0
    min_daily_simulations: int = 0
    min_daily_submissions: int = 0
    #: The Power Pool theme whose matching Alphas it also asks for, by market.
    theme_region: str | None = None
    theme_delay: int | None = None
    min_theme_alphas: int = 0
    #: How the ranking is decided, in the slide's words.
    ranking: str = ""
    tie_breaker: str = ""
    prizes: tuple[str, ...] = ()


CHALLENGES: tuple[Challenge, ...] = (
    Challenge(
        id="usa-glb-nov-2026",
        name="USA/GLB November 2026 Theme Challenge",
        start=date(2026, 11, 7),
        end=date(2026, 11, 24),
        regions=frozenset({"USA", "GLB"}),
        max_prod=0.7,
        min_alphas=30,
        min_daily_simulations=1000,
        min_daily_submissions=1,
        theme_region="GLB",
        theme_delay=1,
        min_theme_alphas=5,
        ranking=(
            "Average production correlation of your eligible Alphas, lowest first, to two decimals"
        ),
        tie_breaker="Number of eligible Alphas",
        prizes=("$50", "$30", "$20"),
    ),
)


def platform_day(value: Any) -> date | None:
    """The BRAIN day of a timestamp. One without an offset is BRAIN's own local time."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        at = value
    elif isinstance(value, date):
        return value
    else:
        try:
            at = datetime.fromisoformat(str(value))
        except ValueError:
            return None
    if at.tzinfo is None:
        return at.date()
    return at.astimezone(PLATFORM_TZ).date()


@dataclass(slots=True)
class Submitted:
    """One submitted Alpha as the challenge sees it."""

    alpha_id: str
    region: str | None
    day: date | None
    prod_correlation: float | None
    #: Whether it matches the challenge's Power Pool theme (None when the theme is unknown).
    theme_match: bool | None = None
    sharpe: float | None = None


@dataclass(slots=True)
class Entry:
    alpha_id: str
    region: str | None
    day: date | None
    prod_correlation: float | None
    eligible: bool
    reasons: list[str]
    theme_match: bool | None
    sharpe: float | None


@dataclass(slots=True)
class Day:
    day: date
    simulations: int
    submissions: int
    eligible: int
    #: The day met both daily rules. ``None`` for today, which is still open.
    met: bool | None


@dataclass(slots=True)
class Standing:
    challenge: Challenge
    today: date
    started: bool
    ended: bool
    days_left: int
    entries: list[Entry]
    days: list[Day]
    eligible: int
    average_prod: float | None
    theme_alphas: int
    unmeasured: int
    missed_days: list[date]
    simulations_today: int
    simulations_needed_today: int
    submissions_needed_today: int
    alphas_needed: int
    alphas_per_day: int
    warnings: list[str] = field(default_factory=list)


def judge(challenge: Challenge, alpha: Submitted) -> Entry:
    """Whether one submitted Alpha counts, and if not, every reason it does not."""
    reasons: list[str] = []
    if alpha.day is None or not challenge.start <= alpha.day <= challenge.end:
        reasons.append(OUTSIDE_WINDOW)
    if alpha.region not in challenge.regions:
        reasons.append(WRONG_REGION)
    if alpha.prod_correlation is None:
        reasons.append(NOT_MEASURED)
    elif alpha.prod_correlation >= challenge.max_prod:
        reasons.append(PROD_TOO_HIGH)
    return Entry(
        alpha_id=alpha.alpha_id,
        region=alpha.region,
        day=alpha.day,
        prod_correlation=alpha.prod_correlation,
        eligible=not reasons,
        reasons=reasons,
        theme_match=alpha.theme_match,
        sharpe=alpha.sharpe,
    )


def standing(
    challenge: Challenge,
    submitted: list[Submitted],
    simulations: dict[date, int],
    today: date,
) -> Standing:
    """Where you stand in ``challenge`` on BRAIN day ``today``.

    A day counts as met when it had at least the daily simulations and at least the daily
    submissions in the challenge's regions. Today is never counted as missed: it is still open,
    and what it still needs is reported instead.
    """
    in_window = [
        a for a in submitted if a.day is not None and challenge.start <= a.day <= challenge.end
    ]
    entries = sorted(
        (judge(challenge, a) for a in in_window),
        key=lambda e: (e.day or date.min, e.alpha_id),
        reverse=True,
    )
    counted = [e for e in entries if e.eligible]
    prods = [e.prod_correlation for e in counted if e.prod_correlation is not None]
    average = sum(prods) / len(prods) if prods else None

    last = min(today, challenge.end)
    days: list[Day] = []
    missed: list[date] = []
    if today >= challenge.start:
        for n in range((last - challenge.start).days + 1):
            d = date.fromordinal(challenge.start.toordinal() + n)
            sims = simulations.get(d, 0)
            subs = [e for e in entries if e.day == d and e.region in challenge.regions]
            met = (
                None
                if d == today
                else sims >= challenge.min_daily_simulations
                and len(subs) >= challenge.min_daily_submissions
            )
            if met is False:
                missed.append(d)
            days.append(
                Day(
                    day=d,
                    simulations=sims,
                    submissions=len(subs),
                    eligible=sum(1 for e in subs if e.eligible),
                    met=met,
                )
            )

    open_today = challenge.start <= today <= challenge.end
    sims_today = simulations.get(today, 0) if open_today else 0
    subs_today = sum(1 for e in entries if e.day == today and e.region in challenge.regions)
    days_left = (challenge.end - today).days + 1 if today <= challenge.end else 0
    days_left = min(days_left, (challenge.end - challenge.start).days + 1)
    alphas_needed = max(0, challenge.min_alphas - len(counted))
    theme = sum(1 for e in counted if e.theme_match and e.region == challenge.theme_region)

    warnings: list[str] = []
    if missed:
        warnings.append(
            f"{len(missed)} day(s) missed a daily rule: " + ", ".join(d.isoformat() for d in missed)
        )
    return Standing(
        challenge=challenge,
        today=today,
        started=today >= challenge.start,
        ended=today > challenge.end,
        days_left=max(0, days_left),
        entries=entries,
        days=list(reversed(days)),
        eligible=len(counted),
        average_prod=average,
        theme_alphas=theme,
        unmeasured=sum(1 for e in entries if NOT_MEASURED in e.reasons),
        missed_days=missed,
        simulations_today=sims_today,
        simulations_needed_today=(
            max(0, challenge.min_daily_simulations - sims_today) if open_today else 0
        ),
        submissions_needed_today=(
            max(0, challenge.min_daily_submissions - subs_today) if open_today else 0
        ),
        alphas_needed=alphas_needed,
        alphas_per_day=math.ceil(alphas_needed / days_left) if days_left > 0 else 0,
        warnings=warnings,
    )


def average_after(current: list[float], candidate: float) -> float:
    """The average production correlation once ``candidate`` is added: the ranking metric."""
    values = [*current, candidate]
    return sum(values) / len(values)


__all__ = [
    "CHALLENGES",
    "NOT_MEASURED",
    "OUTSIDE_WINDOW",
    "PROD_TOO_HIGH",
    "WRONG_REGION",
    "Challenge",
    "Day",
    "Entry",
    "Standing",
    "Submitted",
    "average_after",
    "judge",
    "platform_day",
    "standing",
]
