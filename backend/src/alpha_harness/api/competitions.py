"""Competitions: what BRAIN is running, and where you stand in each.

BRAIN lists every competition with your status in it, but your rank only comes with one
competition at a time, so only the ongoing ones are read in full.
"""

import asyncio
from collections import defaultdict
from datetime import UTC, date, datetime
from typing import Any

import structlog
from fastapi import APIRouter
from pydantic import Field
from sqlalchemy import func, select

from ..agents import challenges as challenge_rules
from ..agents import themes as theme_rules
from ..brain.platform import POWER_POOL_CLASS
from ..db.models import QuotaReservation, QuotaSnapshot, utcnow
from ..schemas import Out
from ..vault.store import json_list
from ..vault.yields import checks_of
from .deps import State

router = APIRouter(prefix="/api/competitions", tags=["competitions"])
log = structlog.get_logger(__name__)


class CompetitionStanding(Out):
    rank: int | None
    alphas: int | None


class Entry(Out):
    """One of your submitted Alphas in a competition or theme."""

    alpha_id: str
    region: str | None
    delay: int | None
    universe: str | None
    sharpe: float | None
    fitness: float | None
    date_submitted: str | None
    #: For a theme: whether it meets every condition the theme asks; the checks it misses.
    matches: bool | None = None
    missing: list[str] = Field(default_factory=list)


class Competition(Out):
    id: str
    name: str
    #: BRAIN's word for your place in it, e.g. ``ACCEPTED`` or ``EXCLUDED``.
    status: str
    enrolled: bool
    ongoing: bool
    scoring: str | None
    start_date: str | None
    end_date: str | None
    sign_up_end_date: str | None
    faq: str | None
    #: Only for an ongoing competition you are enrolled in.
    standing: CompetitionStanding | None
    #: Your submitted Alphas tagged with its id (an Alpha Quality Project cycle's tag).
    entries: list[Entry] = Field(default_factory=list)


class ThemeTab(Out):
    """A Power Pool theme: its market, what it asks, and your tagged Power Pool Alphas in it."""

    id: str
    name: str
    multiplier: float
    region: str | None
    delay: int | None
    qualifier: str | None
    requires: list[str]
    start_date: str | None
    end_date: str | None
    min_for_awards: int
    #: Power Pool Alphas tagged PowerPoolSelected in the theme's market, matching or not.
    entries: list[Entry]


class Competitions(Out):
    competitions: list[Competition]
    themes: list[ThemeTab] = Field(default_factory=list)


def _entry(row: dict[str, Any], **extra: Any) -> Entry:
    return Entry(
        alpha_id=str(row["alpha_id"]),
        region=row.get("region"),
        delay=row.get("delay"),
        universe=row.get("universe"),
        sharpe=row.get("sharpe"),
        fitness=row.get("fitness"),
        date_submitted=str(row["date_submitted"]) if row.get("date_submitted") else None,
        **extra,
    )


#: An Alpha submitted before its theme began, so it does not count towards it.
BEFORE_THEME = "SUBMITTED_BEFORE_THEME"


async def _themes(state: State, members: list[dict[str, Any]]) -> list[ThemeTab]:
    source = getattr(state, "themes", None)
    found = await source.current() if source is not None else []
    if not found:
        return []
    tagged = [
        m
        for m in members
        if POWER_POOL_CLASS in json_list(m.get("classifications"))
        and "PowerPoolSelected" in json_list(m.get("tags"))
    ]
    stored = await state.alphas.by_ids([str(m["alpha_id"]) for m in tagged])
    tabs: list[ThemeTab] = []
    for theme in found:
        window = theme_rules.WINDOWS.get(theme.name)
        entries: list[Entry] = []
        for m in tagged:
            if m.get("region") != theme.region or m.get("delay") != theme.delay:
                continue
            checks = checks_of((stored.get(str(m["alpha_id"])) or {}).get("checks"))
            missing = theme_rules.missed(checks, theme.requires)
            # BRAIN's FAQ: from the theme's start date, a matching Alpha "can be submitted and
            # tagged and will contribute"; one submitted before it does not.
            submitted = str(m.get("date_submitted") or "")[:10]
            if window and submitted and submitted < window[0]:
                missing = [*missing, BEFORE_THEME]
            matches = (
                theme.covers(m.get("region"), m.get("delay"), m.get("universe"), checks)
                and BEFORE_THEME not in missing
            )
            entries.append(_entry(m, matches=matches, missing=missing))
        tabs.append(
            ThemeTab(
                id=theme.id,
                name=theme.name,
                multiplier=theme.multiplier,
                region=theme.region,
                delay=theme.delay,
                qualifier=theme.qualifier,
                requires=list(theme.requires),
                start_date=window[0] if window else None,
                end_date=window[1] if window else None,
                min_for_awards=theme_rules.MIN_FOR_AWARDS,
                entries=sorted(entries, key=lambda e: (not e.matches, -(e.sharpe or 0))),
            )
        )
    return tabs


def _when(value: Any) -> datetime | None:
    """A BRAIN timestamp as an aware datetime. BRAIN sends an offset; one without is read as
    UTC rather than left naive, which could not be compared with ``now``."""
    if not value:
        return None
    try:
        at = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return at if at.tzinfo is not None else at.replace(tzinfo=UTC)


def _is_ongoing(c: dict[str, Any], now: datetime) -> bool:
    start, end = _when(c.get("startDate")), _when(c.get("endDate"))
    return start is not None and end is not None and start <= now <= end


@router.get("")
async def competitions(state: State) -> Competitions:
    listed = await state.endpoints.competitions()
    now = datetime.now(UTC)
    ongoing = [c for c in listed if _is_ongoing(c, now)]
    # One competition BRAIN will not show (restricted, or a passing error) costs only its rank,
    # not the page.
    details = await asyncio.gather(
        *(state.endpoints.competition(str(c["id"])) for c in ongoing), return_exceptions=True
    )
    boards: dict[str, Any] = {}
    for c, d in zip(ongoing, details, strict=True):
        if isinstance(d, BaseException):
            log.warning("competitions.detail_failed", competition=c.get("id"), error=str(d))
        else:
            boards[str(d.get("id"))] = d.get("leaderboard")

    members = await state.alphas.submitted_members()

    def tagged(cid: str) -> list[Entry]:
        wanted = cid.upper()
        return [
            _entry(m) for m in members if wanted in {t.upper() for t in json_list(m.get("tags"))}
        ]

    def row(c: dict[str, Any]) -> Competition:
        cid = str(c["id"])
        board = boards.get(cid)
        return Competition(
            id=cid,
            name=str(c.get("name") or cid),
            status=str(c.get("status") or ""),
            enrolled=c.get("signUpDate") is not None,
            ongoing=_is_ongoing(c, now),
            scoring=c.get("scoring"),
            start_date=c.get("startDate"),
            end_date=c.get("endDate"),
            sign_up_end_date=c.get("signUpEndDate"),
            faq=c.get("faq"),
            standing=CompetitionStanding(rank=board.get("rank"), alphas=board.get("alphas"))
            if isinstance(board, dict)
            else None,
            entries=tagged(cid) if _is_ongoing(c, now) else [],
        )

    rows = [row(c) for c in listed]

    # Ongoing first, soonest to end; then the rest, most recent first. By the moment, not the
    # text: BRAIN's offsets change with daylight saving, and text sorts ignore them.
    def ends(r: Competition) -> datetime:
        return _when(r.end_date) or datetime.min.replace(tzinfo=UTC)

    live = sorted((r for r in rows if r.ongoing), key=ends)
    past = sorted((r for r in rows if not r.ongoing), key=ends, reverse=True)
    return Competitions(competitions=live + past, themes=await _themes(state, members))


# --- consultant challenges ---------------------------------------------------------------


class ChallengeEntry(Out):
    """One Alpha submitted inside a challenge's window, and whether it counts."""

    alpha_id: str
    region: str | None
    day: str | None
    prod_correlation: float | None
    sharpe: float | None
    eligible: bool
    #: Why it does not count: OUTSIDE_WINDOW, WRONG_REGION, PROD_CORRELATION, PROD_NOT_MEASURED.
    reasons: list[str]
    theme_match: bool | None


class ChallengeDay(Out):
    day: str
    simulations: int
    submissions: int
    eligible: int
    #: Both daily rules met; ``None`` for today, which is still open.
    met: bool | None


class ChallengeRules(Out):
    regions: list[str]
    max_prod: float
    min_alphas: int
    min_daily_simulations: int
    min_daily_submissions: int
    theme_region: str | None
    theme_delay: int | None
    min_theme_alphas: int
    ranking: str
    tie_breaker: str
    prizes: list[str]


class ChallengeTab(Out):
    id: str
    name: str
    start_date: str
    end_date: str
    today: str
    started: bool
    ended: bool
    days_left: int
    rules: ChallengeRules
    eligible: int
    #: The ranking metric: the mean production correlation of the eligible Alphas.
    average_prod: float | None
    theme_alphas: int
    unmeasured: int
    missed_days: list[str]
    simulations_today: int
    simulations_needed_today: int
    submissions_needed_today: int
    alphas_needed: int
    alphas_per_day: int
    #: Kepler is signed in to BRAIN: while it is not, nothing simulates.
    signed_in: bool | None
    warnings: list[str]
    days: list[ChallengeDay]
    entries: list[ChallengeEntry]


class Challenges(Out):
    challenges: list[ChallengeTab]


async def _simulations_by_day(state: State) -> dict[date, int]:
    """Simulations sent on each BRAIN day: the larger of Kepler's own ledger and BRAIN's count.

    BRAIN's count (``limit - remaining`` from the quota headers) includes simulations run by
    hand on the website; the ledger covers days whose last reading was taken early.
    """
    counted: dict[date, int] = defaultdict(int)
    async with state.db.session() as session:
        ledger = await session.execute(
            select(
                QuotaReservation.day,
                func.sum(func.coalesce(QuotaReservation.actual_units, QuotaReservation.units)),
            )
            .where(QuotaReservation.state.in_(("CONFIRMED", "UNCERTAIN")))
            .group_by(QuotaReservation.day)
        )
        for day, units in ledger.all():
            parsed = challenge_rules.platform_day(day)
            if parsed is not None:
                counted[parsed] = max(counted[parsed], int(units or 0))
        readings = await session.execute(
            select(QuotaSnapshot.observed_at, QuotaSnapshot.limit_total, QuotaSnapshot.remaining)
        )
        for observed, limit, remaining in readings.all():
            if limit is None or remaining is None:
                continue
            at = observed if observed.tzinfo else observed.replace(tzinfo=UTC)
            day = challenge_rules.platform_day(at)
            if day is not None:
                counted[day] = max(counted[day], int(limit) - int(remaining))
    return dict(counted)


async def _theme_matches(
    state: State, challenge: challenge_rules.Challenge, members: list[dict[str, Any]]
) -> dict[str, bool]:
    """Which submitted Power Pool Alphas match the challenge's theme, by Alpha id."""
    if challenge.theme_region is None:
        return {}
    source = getattr(state, "themes", None)
    found = await source.current() if source is not None else []
    theme = next(
        (
            t
            for t in found
            if t.region == challenge.theme_region
            and t.delay == challenge.theme_delay
            and "power pool" in t.name.lower()
        ),
        None,
    )
    if theme is None:
        return {}
    pool = [m for m in members if POWER_POOL_CLASS in json_list(m.get("classifications"))]
    stored = await state.alphas.by_ids([str(m["alpha_id"]) for m in pool])
    return {
        str(m["alpha_id"]): theme.covers(
            m.get("region"),
            m.get("delay"),
            m.get("universe"),
            checks_of((stored.get(str(m["alpha_id"])) or {}).get("checks")),
        )
        for m in pool
    }


def _prod_of(member: dict[str, Any]) -> float | None:
    value = member.get("prod_correlation")
    return float(value) if isinstance(value, int | float) else None


@router.get("/challenges")
async def challenges(state: State) -> Challenges:
    """Each consultant challenge Kepler knows the rules of, with where you stand in it.

    Read from Kepler's own records only, so it answers while the session is signed out.
    """
    members = await state.alphas.submitted_members()
    simulations = await _simulations_by_day(state)
    today = utcnow().astimezone(challenge_rules.PLATFORM_TZ).date()
    engine = getattr(state, "engine", None)
    lost: bool | None = engine.session_lost if engine is not None else None
    tabs: list[ChallengeTab] = []
    for ch in challenge_rules.CHALLENGES:
        themed = await _theme_matches(state, ch, members)
        submitted = [
            challenge_rules.Submitted(
                alpha_id=str(m["alpha_id"]),
                region=m.get("region"),
                day=challenge_rules.platform_day(m.get("date_submitted")),
                prod_correlation=_prod_of(m),
                theme_match=themed.get(str(m["alpha_id"])) if themed else None,
                sharpe=m.get("sharpe"),
            )
            for m in members
        ]
        st = challenge_rules.standing(ch, submitted, simulations, today)
        warnings = list(st.warnings)
        if lost is True and not st.ended:
            warnings.insert(
                0, "Kepler is signed out of BRAIN: nothing simulates until you sign in again."
            )
        tabs.append(
            ChallengeTab(
                id=ch.id,
                name=ch.name,
                start_date=ch.start.isoformat(),
                end_date=ch.end.isoformat(),
                today=today.isoformat(),
                started=st.started,
                ended=st.ended,
                days_left=st.days_left,
                rules=ChallengeRules(
                    regions=sorted(ch.regions),
                    max_prod=ch.max_prod,
                    min_alphas=ch.min_alphas,
                    min_daily_simulations=ch.min_daily_simulations,
                    min_daily_submissions=ch.min_daily_submissions,
                    theme_region=ch.theme_region,
                    theme_delay=ch.theme_delay,
                    min_theme_alphas=ch.min_theme_alphas,
                    ranking=ch.ranking,
                    tie_breaker=ch.tie_breaker,
                    prizes=list(ch.prizes),
                ),
                eligible=st.eligible,
                average_prod=st.average_prod,
                theme_alphas=st.theme_alphas,
                unmeasured=st.unmeasured,
                missed_days=[d.isoformat() for d in st.missed_days],
                simulations_today=st.simulations_today,
                simulations_needed_today=st.simulations_needed_today,
                submissions_needed_today=st.submissions_needed_today,
                alphas_needed=st.alphas_needed,
                alphas_per_day=st.alphas_per_day,
                signed_in=None if lost is None else not lost,
                warnings=warnings,
                days=[
                    ChallengeDay(
                        day=d.day.isoformat(),
                        simulations=d.simulations,
                        submissions=d.submissions,
                        eligible=d.eligible,
                        met=d.met,
                    )
                    for d in st.days
                ],
                entries=[
                    ChallengeEntry(
                        alpha_id=e.alpha_id,
                        region=e.region,
                        day=e.day.isoformat() if e.day else None,
                        prod_correlation=e.prod_correlation,
                        sharpe=e.sharpe,
                        eligible=e.eligible,
                        reasons=e.reasons,
                        theme_match=e.theme_match,
                    )
                    for e in st.entries
                ],
            )
        )
    return Challenges(challenges=tabs)
