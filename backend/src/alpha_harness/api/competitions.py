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
from ..brain.platform import POWER_POOL_CLASS, POWER_POOL_MAX_FIELDS, POWER_POOL_MAX_OPERATORS
from ..db.models import QuotaReservation, QuotaSnapshot, utcnow
from ..schemas import Out
from ..vault.store import json_list
from ..vault.yields import checks_of
from .deps import State, refuse

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


# --- challenge candidates ------------------------------------------------------------------

#: How many of the lowest-correlation candidates are measured against your pools.
CANDIDATE_DEPTH = 40
WHOLE_POOL = "all"
POWER_POOL = "Power Pool"


class PoolEffect(Out):
    """A pool's Sharpe before and after the candidate joins it."""

    members: int
    before: float | None
    after: float | None


class ChallengeCandidate(Out):
    alpha_id: str
    region: str | None
    universe: str | None
    expression: str | None
    sharpe: float | None
    fitness: float | None
    turnover: float | None
    prod_correlation: float
    #: Your ranking metric if this one is submitted next.
    average_after: float
    operators: int | None
    fields: int | None
    power_pool_correlation: float | None
    #: Within Power Pool's limits: at most 8 operators and 3 fields, Power Pool correlation
    #: under 0.5 where measured.
    power_pool_ok: bool
    #: Clears the regular Sharpe and Fitness limits, so no theme is needed.
    regular: bool
    #: Matches the month's Power Pool theme in its market (None where there is none).
    theme_match: bool | None
    whole: PoolEffect | None
    power: PoolEffect | None
    #: Lifts both your combined performance and your Power Pool performance.
    improves: bool | None
    #: Meets every condition: the challenge, Power Pool, and both pools improve.
    qualifies: bool
    #: BRAIN's full check not yet run on it.
    pending: bool
    reasons: list[str]


class ChallengeCandidates(Out):
    challenge_id: str
    average_prod: float | None
    candidates: list[ChallengeCandidate]


def _sharpe_of(figures: dict[str, Any] | None) -> float | None:
    value = (figures or {}).get("sharpe")
    return float(value) if isinstance(value, int | float) else None


def _effect(raw: dict[str, Any] | None) -> PoolEffect | None:
    if not raw:
        return None
    return PoolEffect(
        members=int(raw.get("members") or 0),
        before=_sharpe_of(raw.get("before")),
        after=_sharpe_of(raw.get("after")),
    )


def _lifts(effect: PoolEffect | None) -> bool | None:
    if effect is None or effect.after is None:
        return None
    return effect.before is None or effect.after >= effect.before


def assess(
    *,
    power_ok: bool,
    regular: bool,
    theme_match: bool | None,
    whole: PoolEffect | None,
    power: PoolEffect | None,
) -> tuple[bool | None, list[str]]:
    """Whether a candidate lifts both pools, and every reason it falls short of qualifying.

    It qualifies within Power Pool's limits, as a regular Alpha or one matching the month's
    theme (a pure Power Pool Alpha needs the theme), lifting combined and Power Pool Sharpe.
    """
    lifts = [_lifts(whole), _lifts(power)]
    improves = None if None in lifts else all(lifts)
    reasons: list[str] = []
    if not power_ok:
        reasons.append("OUTSIDE_POWER_POOL_LIMITS")
    if not regular and not theme_match:
        reasons.append("NEEDS_REGULAR_OR_THEME")
    if improves is False:
        reasons.append("LOWERS_A_POOL")
    elif improves is None:
        reasons.append("EFFECT_NOT_MEASURED")
    return improves, reasons


@router.get("/challenges/{challenge_id}/candidates")
async def challenge_candidates(challenge_id: str, state: State) -> ChallengeCandidates:
    """The submittable Alphas that best serve a challenge, Power Pool and both pools at once.

    Of the submittable Alphas in the challenge's regions with a measured production
    correlation under its ceiling, the lowest few are sized as Power Pool counts them, matched
    against the month's theme, and measured against your combined and Power Pool pools.
    Those meeting every condition come first, lowest production correlation first.
    """
    from ..agents import guard
    from ..tools import portfolio
    from .lab_tasks import submittable_alphas
    from .template_lab import sizes

    ch = next((c for c in challenge_rules.CHALLENGES if c.id == challenge_id), None)
    if ch is None:
        raise refuse(404, "no_challenge", f"No challenge called {challenge_id}.")
    members = await state.alphas.submitted_members()
    today = utcnow().astimezone(challenge_rules.PLATFORM_TZ).date()
    submitted = [
        challenge_rules.Submitted(
            alpha_id=str(m["alpha_id"]),
            region=m.get("region"),
            day=challenge_rules.platform_day(m.get("date_submitted")),
            prod_correlation=_prod_of(m),
        )
        for m in members
    ]
    standing = challenge_rules.standing(ch, submitted, {}, today)
    counted = [
        e.prod_correlation
        for e in standing.entries
        if e.eligible and e.prod_correlation is not None
    ]

    seen: set[str] = set()
    picked = []
    for a in await submittable_alphas(state):
        region = str((a.settings or {}).get("region") or "")
        if (
            not a.alpha_id
            or a.alpha_id in seen
            or region not in ch.regions
            or a.prod_correlation is None
            or a.prod_correlation >= ch.max_prod
        ):
            continue
        seen.add(a.alpha_id)
        picked.append(a)
    picked.sort(key=lambda a: a.prod_correlation or 1.0)
    picked = picked[:CANDIDATE_DEPTH]
    ids = [str(a.alpha_id) for a in picked]
    stored = await state.alphas.by_ids(ids)

    # Sized per market, as Power Pool counts them.
    sized: dict[str, dict[str, Any]] = {}
    by_market: dict[tuple[str, int], list[str]] = defaultdict(list)
    for a in picked:
        settings = a.settings or {}
        by_market[(str(settings.get("region")), int(settings.get("delay") or 0))].append(
            str(a.alpha_id)
        )
    for (region, delay), market_ids in by_market.items():
        texts = [str((stored.get(i) or {}).get("expression") or "") for i in market_ids]
        stats = await sizes(state, texts, region, delay)
        sized.update(zip(market_ids, stats, strict=True))

    source = getattr(state, "themes", None)
    month = await source.current() if source is not None else []

    scopes: dict[str, list[str]] = {
        WHOLE_POOL: [str(m["alpha_id"]) for m in members],
        POWER_POOL: [
            str(m["alpha_id"])
            for m in members
            if POWER_POOL_CLASS in json_list(m.get("classifications"))
        ],
    }
    joins = {i: [WHOLE_POOL, POWER_POOL] for i in ids}
    series = await state.alphas.series([*scopes[WHOLE_POOL], *ids])
    meta = await state.alphas.by_ids(scopes[WHOLE_POOL]) | stored
    effects = await asyncio.to_thread(
        portfolio.what_if, series, meta, scopes, joins, guard.RECENT_START
    )

    out: list[ChallengeCandidate] = []
    for a in picked:
        alpha_id = str(a.alpha_id)
        row = stored.get(alpha_id) or {}
        checks = checks_of(row.get("checks"))
        by_name = {c.get("name"): c for c in checks}
        ppc_check = by_name.get("POWER_POOL_CORRELATION") or {}
        ppc = ppc_check.get("value")
        ppc = float(ppc) if isinstance(ppc, int | float) else None
        stat = sized.get(alpha_id) or {}
        operators = (stat.get("operators") or [None])[-1]
        fields = (stat.get("fields") or [None])[-1]
        limits = a.limits or {}
        regular = (a.sharpe or 0) >= float(limits.get("LOW_SHARPE", 99)) and (
            a.fitness or 0
        ) >= float(limits.get("LOW_FITNESS", 99))
        theme = next(
            (t for t in month if t.region == row.get("region") and t.delay == row.get("delay")),
            None,
        )
        theme_match = (
            theme.covers(row.get("region"), row.get("delay"), row.get("universe"), checks)
            if theme is not None
            else None
        )
        power_ok = (
            operators is not None
            and operators <= POWER_POOL_MAX_OPERATORS
            and fields is not None
            and fields <= POWER_POOL_MAX_FIELDS
            and (ppc is None or ppc < 0.5)
        )
        whole = _effect(effects.get(alpha_id, {}).get(WHOLE_POOL))
        power = _effect(effects.get(alpha_id, {}).get(POWER_POOL))
        improves, reasons = assess(
            power_ok=power_ok, regular=regular, theme_match=theme_match, whole=whole, power=power
        )
        prod = float(a.prod_correlation or 0.0)
        out.append(
            ChallengeCandidate(
                alpha_id=alpha_id,
                region=row.get("region"),
                universe=row.get("universe"),
                expression=row.get("expression"),
                sharpe=a.sharpe,
                fitness=a.fitness,
                turnover=a.turnover,
                prod_correlation=prod,
                average_after=challenge_rules.average_after(counted, prod),
                operators=operators,
                fields=fields,
                power_pool_correlation=ppc,
                power_pool_ok=power_ok,
                regular=regular,
                theme_match=theme_match,
                whole=whole,
                power=power,
                improves=improves,
                qualifies=not reasons,
                pending=bool(a.pending),
                reasons=reasons,
            )
        )
    out.sort(key=lambda c: (not c.qualifies, c.prod_correlation))
    return ChallengeCandidates(
        challenge_id=ch.id,
        average_prod=standing.average_prod,
        candidates=out,
    )
