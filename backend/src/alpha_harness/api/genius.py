"""The Genius tracker: your Genius figures, typed in, against the Submission Desk's estimate."""

from dataclasses import asdict
from datetime import UTC, date, datetime

import structlog
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..agents import genius, master, osmosis
from ..brain.errors import BrainValidationError
from ..db.models import GeniusReading
from ..schemas import Out
from .deps import State, refuse

router = APIRouter(prefix="/api/genius", tags=["genius"])
log = structlog.get_logger(__name__)


class GeniusReadingOut(Out):
    id: int
    read_on: str
    combined: float | None
    power_pool: float | None
    signals: int | None
    pyramids: int | None
    note: str | None
    #: ``brain`` when synced from BRAIN's Genius page, ``you`` when typed in.
    source: str
    level: str | None
    rank: int | None
    #: What the Desk's estimate said that day, from the alphas submitted by then.
    estimate_combined: float | None
    estimate_power_pool: float | None
    alphas: int


class GeniusStep(Out):
    since: str
    until: str
    submitted: list[str]
    brain_combined: float | None
    brain_power_pool: float | None
    estimate_combined: float | None
    estimate_power_pool: float | None
    estimate_quality: float | None


class GeniusAgreement(Out):
    pairs: int
    same_direction: int
    correlation: float | None
    verdict: str


class GeniusBar(Out):
    metric: str
    label: str
    #: Expert, Master, Grandmaster.
    bars: list[float]
    value: float | None
    level: str | None


class GeniusToday(Out):
    combined: float | None
    power_pool: float | None
    alphas: int
    figures: dict[str, float | int | None] | None


class GeniusView(Out):
    readings: list[GeniusReadingOut]
    steps: list[GeniusStep]
    agreement: GeniusAgreement
    bars: list[GeniusBar]
    levels: list[str]
    today: GeniusToday | None
    min_pairs: int
    #: The last time the figures were read from BRAIN, and why the last try failed.
    synced_at: str | None
    sync_error: str | None
    sync_hours: float


class NewReading(BaseModel):
    read_on: date = Field(alias="readOn")
    combined: float | None = None
    power_pool: float | None = Field(default=None, alias="powerPool")
    signals: int | None = Field(default=None, ge=0)
    pyramids: int | None = Field(default=None, ge=0)
    note: str | None = Field(default=None, max_length=500)


_LABELS = {
    "signals": "Signals",
    "pyramids": "Pyramids completed",
    "combined": "Combined Alpha Performance",
    "power_pool": "Combined Power Pool Alpha Performance",
}


@router.get("")
async def view(state: State) -> GeniusView:
    found = await genius.report(state)
    latest = found.readings[-1] if found.readings else {}
    keys = {"signals": "signals", "pyramids": "pyramids", "combined": "combined"}
    keys["power_pool"] = "powerPool"

    def value(metric: str) -> float | None:
        raw = latest.get(keys[metric])
        return float(raw) if isinstance(raw, int | float) else None

    return GeniusView(
        readings=[GeniusReadingOut.model_validate(r) for r in found.readings],
        steps=[
            GeniusStep.model_validate(
                asdict(s) | {"since": s.since.isoformat(), "until": s.until.isoformat()}
            )
            for s in found.steps
        ],
        agreement=GeniusAgreement.model_validate(asdict(found.agreement)),
        bars=[
            GeniusBar(
                metric=metric,
                label=_LABELS[metric],
                bars=list(bars),
                value=value(metric),
                level=genius.level_of(metric, value(metric)),
            )
            for metric, bars in genius.BARS.items()
        ],
        levels=list(genius.LEVELS),
        today=GeniusToday.model_validate(asdict(found.today)) if found.today else None,
        min_pairs=genius.MIN_PAIRS,
        synced_at=state.genius.synced_at.isoformat() if state.genius.synced_at else None,
        sync_error=state.genius.error,
        sync_hours=genius.SYNC_HOURS,
    )


@router.post("/sync")
async def sync(state: State) -> GeniusView:
    """Read your Genius figures from BRAIN now, as its Genius page does, as today's reading."""
    if not state.auth.session.authenticated:
        raise refuse(409, "signed_out", "Sign in to BRAIN to sync your Genius figures.")
    try:
        await state.genius.sync()
    except Exception as exc:
        raise refuse(502, "sync_failed", f"BRAIN did not give your Genius figures: {exc}") from exc
    return await view(state)


@router.post("", status_code=201)
async def add(body: NewReading, state: State) -> GeniusView:
    """Store what your Genius page shows today. Any figure may be left out."""
    if all(v is None for v in (body.combined, body.power_pool, body.signals, body.pyramids)):
        raise refuse(422, "empty", "Enter at least one figure from your Genius page.")
    async with state.db.session() as session:
        session.add(
            GeniusReading(
                read_on=body.read_on,
                combined=body.combined,
                power_pool=body.power_pool,
                signals=body.signals,
                pyramids=body.pyramids,
                note=(body.note or "").strip() or None,
            )
        )
    return await view(state)


@router.delete("/{reading_id}")
async def remove(reading_id: int, state: State) -> GeniusView:
    async with state.db.session() as session:
        row = await session.get(GeniusReading, reading_id)
        if row is None:
            raise refuse(404, "not_found", f"No reading {reading_id}.")
        await session.delete(row)
    return await view(state)


# -- Osmosis -----------------------------------------------------------------


class OsmosisPick(Out):
    alpha_id: str
    points: int
    current: int
    recent_sharpe: float | None
    turnover: float | None
    role: str
    reason: str


class OsmosisScope(Out):
    region: str
    delay: int
    alphas: int
    eligible: bool
    #: Alphas still to submit here before the scope can hold an allocation.
    short_of: int
    currently_valid: bool
    current_points: int
    current_alphas: int
    current_sharpe: float | None
    proposed_sharpe: float | None
    picks: list[OsmosisPick]


class OsmosisView(Out):
    #: The next Sunday 23:59 US Eastern, when the allocation locks.
    locks_at: str
    min_scopes: int
    min_alphas: int
    points: int
    valid_now: int
    eligible: int
    scopes: list[OsmosisScope]
    #: What stands between you and a valid allocation, in plain words.
    advice: list[str]


class OsmosisApply(BaseModel):
    #: ``REGION/DELAY`` of each scope to set as proposed.
    scopes: list[str] = Field(min_length=1, max_length=20)


#: Times a scope is planned and set again after BRAIN refuses points on some of its Alphas.
OSMOSIS_ROUNDS = 3
#: BRAIN's words for an Alpha that can never hold Osmosis points, measured: "Cannot update
#: Osmosis points for non-compensated alpha."
NON_COMPENSATED = "non-compensated"


class OsmosisApplied(Out):
    changed: int
    failed: list[str]
    view: OsmosisView


async def _osmosis(state: State) -> tuple[list[osmosis.Scope], dict[str, int]]:
    refused = await state.metadata.osmosis_refused()
    members = [m for m in await state.alphas.submitted_members() if m["alpha_id"] not in refused]
    series = await state.alphas.series([str(m["alpha_id"]) for m in members])
    current = await state.endpoints.osmosis_points()
    return osmosis.plan(members, series, current), current


def _osmosis_view(scopes: list[osmosis.Scope]) -> OsmosisView:
    eligible = [s for s in scopes if s.eligible]
    valid_now = sum(1 for s in scopes if s.currently_valid)
    advice: list[str] = []
    if len(eligible) < osmosis.MIN_SCOPES:
        need = osmosis.MIN_SCOPES - len(eligible)
        # Delay 1 first: BRAIN allows 5 Delay 0 submissions a month, at a higher bar.
        closest = sorted(
            (s for s in scopes if not s.eligible),
            key=lambda s: (s.delay == 1, s.alphas),
            reverse=True,
        )[:need]
        advice.append(
            f"Only {len(eligible)} scope{'' if len(eligible) == 1 else 's'} hold the "
            f"{osmosis.MIN_ALPHAS} Alphas an allocation needs; BRAIN asks for "
            f"{osmosis.MIN_SCOPES}. Closest: "
            + ", ".join(
                f"{s.region} D{s.delay} ({osmosis.MIN_ALPHAS - s.alphas} more)" for s in closest
            )
            + "."
        )
    if valid_now < osmosis.MIN_SCOPES:
        advice.append(
            f"{valid_now} of your scopes hold a valid allocation now, so the Daily Osmosis Rank "
            "is not generated and Combined Osmosis Performance cannot build."
        )
    advice += osmosis.shortlist_gaps(scopes)
    return OsmosisView(
        locks_at=osmosis.next_lock().isoformat(),
        min_scopes=osmosis.MIN_SCOPES,
        min_alphas=osmosis.MIN_ALPHAS,
        points=osmosis.POINTS,
        valid_now=valid_now,
        eligible=len(eligible),
        scopes=[
            OsmosisScope(
                region=s.region,
                delay=s.delay,
                alphas=s.alphas,
                eligible=s.eligible,
                short_of=max(0, osmosis.MIN_ALPHAS - s.alphas),
                currently_valid=s.currently_valid,
                current_points=s.current_points,
                current_alphas=s.current_alphas,
                current_sharpe=s.current_sharpe,
                proposed_sharpe=s.proposed_sharpe,
                picks=[OsmosisPick.model_validate(asdict(p)) for p in s.picks],
            )
            for s in scopes
        ],
        advice=advice,
    )


@router.get("/osmosis")
async def osmosis_plan(state: State) -> OsmosisView:
    """Your Osmosis points as BRAIN holds them, and a valid allocation to propose. Reads only."""
    if not state.auth.session.authenticated:
        raise refuse(409, "signed_out", "Sign in to BRAIN to read your Osmosis points.")
    scopes, _ = await _osmosis(state)
    return _osmosis_view(scopes)


@router.post("/osmosis/apply")
async def osmosis_apply(body: OsmosisApply, state: State) -> OsmosisApplied:
    """Set these scopes' Osmosis points on BRAIN as proposed: each Alpha whose points differ
    is edited, those left out cleared. Only on your click; never submits anything.

    BRAIN holds points only on compensated Alphas. One it refuses as non-compensated is
    remembered and left out, and the scope is planned and set again without it, so its points
    still add up. Any other refusal is reported, not remembered.
    """
    if not state.auth.session.authenticated:
        raise refuse(409, "signed_out", "Sign in to BRAIN to set Osmosis points.")
    wanted = {name.upper() for name in body.scopes}
    changed, failed = 0, []
    for attempt in range(OSMOSIS_ROUNDS):
        scopes, current = await _osmosis(state)
        chosen = [s for s in scopes if f"{s.region}/{s.delay}" in wanted and s.eligible and s.picks]
        if not chosen:
            if attempt == 0:
                raise refuse(
                    422, "nothing_to_apply", "None of those scopes has a proposal to apply."
                )
            break
        refused: list[str] = []
        for scope in chosen:
            for pick in scope.picks:
                if current.get(pick.alpha_id, 0) == pick.points:
                    continue
                try:
                    # BRAIN takes 1 or more, or null to clear: never 0.
                    await state.endpoints.update_alpha(
                        pick.alpha_id, {"osmosisPoints": pick.points or None}
                    )
                    changed += 1
                except BrainValidationError as exc:
                    detail = exc.fields or exc.body
                    log.warning("osmosis.refused", alpha_id=pick.alpha_id, detail=detail)
                    if NON_COMPENSATED in str(detail).lower():
                        refused.append(pick.alpha_id)
                    else:
                        failed.append(pick.alpha_id)
                except Exception as exc:  # noqa: BLE001 - a blip: reported, not remembered
                    log.warning("osmosis.apply_failed", alpha_id=pick.alpha_id, error=str(exc))
                    failed.append(pick.alpha_id)
        if not refused:
            break
        failed += refused
        await state.metadata.add_osmosis_refused(refused)
    scopes, _ = await _osmosis(state)
    return OsmosisApplied(changed=changed, failed=failed, view=_osmosis_view(scopes))


# -- Path to a level ---------------------------------------------------------


class LevelRow(Out):
    key: str
    label: str
    you: float | None
    bar: float | None
    peers: float | None
    lower_better: bool
    met: bool | None
    note: str


class LevelPlan(Out):
    level: str
    quarter_end: str
    days_left: int
    performance_cutoff: str
    bars: list[LevelRow]
    tie_breakers: list[LevelRow]
    performances: dict[str, float | None]
    #: Consultants at the level on the leaderboard, whose medians are the peer column.
    peers_counted: int
    pace: list[str]


def _row(row: master.Row) -> LevelRow:
    return LevelRow.model_validate(asdict(row) | {"met": row.met})


@router.get("/level")
async def level_plan(state: State, level: str = "Master") -> LevelPlan:
    """Your figures this quarter against ``level``'s bars and the consultants already at it."""
    if level not in genius.LEVELS:
        raise refuse(422, "unknown_level", f"Pick one of {', '.join(genius.LEVELS)}.")
    if not state.auth.session.authenticated:
        raise refuse(409, "signed_out", "Sign in to BRAIN to read your Genius figures.")
    current, board = await state.master.figures()
    found = master.plan(current, board, datetime.now(UTC).date(), level)
    return LevelPlan(
        level=found.level,
        quarter_end=found.quarter_end.isoformat(),
        days_left=found.days_left,
        performance_cutoff=found.performance_cutoff.isoformat(),
        bars=[_row(r) for r in found.bars],
        tie_breakers=[_row(r) for r in found.tie_breakers],
        performances=found.performances,
        peers_counted=found.peers_counted,
        pace=found.pace,
    )
