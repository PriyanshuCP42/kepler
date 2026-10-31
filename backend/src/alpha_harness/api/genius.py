"""The Genius tracker: your Genius figures, typed in, against the Submission Desk's estimate."""

from dataclasses import asdict
from datetime import date

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..agents import genius
from ..db.models import GeniusReading
from ..schemas import Out
from .deps import State, refuse

router = APIRouter(prefix="/api/genius", tags=["genius"])


class GeniusReadingOut(Out):
    id: int
    read_on: str
    combined: float | None
    power_pool: float | None
    signals: int | None
    pyramids: int | None
    note: str | None
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


class GeniusView(Out):
    readings: list[GeniusReadingOut]
    steps: list[GeniusStep]
    agreement: GeniusAgreement
    bars: list[GeniusBar]
    levels: list[str]
    today: GeniusToday | None
    min_pairs: int


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
    )


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
