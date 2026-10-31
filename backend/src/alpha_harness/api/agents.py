"""Research HQ: the agents, what they are doing, and the Critic's shortlist."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from ..agents.critic import reasons_text
from ..db.models import CriticVerdict, Study
from ..schemas import Out
from ..tasks import spawn
from .deps import State, refuse

router = APIRouter(prefix="/api/agents", tags=["agents"])


class AgentRoster(Out):
    agents: list[dict[str, Any]]
    handoffs: list[dict[str, Any]]


class CriticSummary(Out):
    #: Alphas by verdict: pass, near_miss, fail, unjudged.
    verdicts: dict[str, int]
    #: Passing alphas the Family Clerk has placed, and how many distinct families they form.
    placed: int
    families: int


class ShortlistEntry(Out):
    alpha_id: str
    score: float
    region: str | None
    delay: int | None
    universe: str | None
    pool_correlation: float | None
    family_size: int
    gates: dict[str, Any]
    reasons: list[str]


class Shortlist(Out):
    alphas: list[ShortlistEntry]


class Started(Out):
    started: bool


@router.get("")
async def roster(state: State) -> AgentRoster:
    """Every agent, its state and counters, and the latest handoffs between them."""
    return AgentRoster.model_validate(state.agents.snapshot())


@router.get("/critic")
async def critic_summary(state: State) -> CriticSummary:
    async with state.db.session() as session:
        placed = await session.scalar(
            select(func.count())
            .select_from(CriticVerdict)
            .where(CriticVerdict.family_checked_at.is_not(None))
        )
        families = await session.scalar(
            select(func.count())
            .select_from(CriticVerdict)
            .where(
                CriticVerdict.verdict == "pass", CriticVerdict.family_id == CriticVerdict.alpha_id
            )
        )
    return CriticSummary(
        verdicts=await state.critic.counts(), placed=int(placed or 0), families=int(families or 0)
    )


@router.post("/critic/rescore", status_code=202)
async def rescore(state: State) -> Started:
    """Score every alpha in the vault again. Runs in the background; costs no quota."""
    if state.critic_busy:
        raise refuse(409, "busy", "The Critic is already scoring the vault.")

    state.critic_busy = True

    async def run() -> None:
        try:
            await state.critic.rescore(state.tasks)
            state.families.nudge()
        finally:
            state.critic_busy = False

    spawn(run(), name="critic-rescore")
    return Started(started=True)


@router.get("/shortlist")
async def shortlist(state: State, limit: Annotated[int, Query(ge=1, le=100)] = 30) -> Shortlist:
    """Family leaders that passed every gate, best first: one per distinct idea."""
    size = (
        select(CriticVerdict.family_id, func.count().label("n"))
        .where(CriticVerdict.verdict == "pass", CriticVerdict.family_id.is_not(None))
        .group_by(CriticVerdict.family_id)
        .subquery()
    )
    async with state.db.session() as session:
        rows = (
            await session.execute(
                select(CriticVerdict, size.c.n)
                .join(size, size.c.family_id == CriticVerdict.alpha_id)
                .where(CriticVerdict.verdict == "pass")
                .order_by(CriticVerdict.score.desc())
                .limit(limit)
            )
        ).all()
    return Shortlist(
        alphas=[
            ShortlistEntry(
                alpha_id=row.alpha_id,
                score=row.score,
                region=row.region,
                delay=row.delay,
                universe=row.universe,
                pool_correlation=row.pool_correlation,
                family_size=int(n),
                gates=row.gates or {},
                reasons=reasons_text(row.reasons),
            )
            for row, n in rows
        ]
    )


class RepairMoveStat(Out):
    failure: str
    move: str
    name: str
    tries: int
    wins: int


class RepairSummary(Out):
    #: Near misses not tried yet.
    waiting: int
    #: Repairs written but not yet judged, and how the judged ones came out.
    pending: int
    fixed: int
    not_fixed: int
    errors: int
    #: What the Repair Engineer has learned, best first.
    moves: list[RepairMoveStat]


class RepairRequest(BaseModel):
    #: Near misses to repair, best Critic score first.
    alphas: int = Field(default=100, ge=1, le=500)
    per_alpha: int = Field(default=3, ge=1, le=6, alias="perAlpha")
    cores: int = Field(default=4, ge=1, le=8)

    model_config = {"populate_by_name": True}


class RepairTaskAdded(Out):
    id: int
    name: str
    simulations: int


@router.get("/repair")
async def repair_summary(state: State) -> RepairSummary:
    """Near misses waiting, repairs out, and which fixes have worked."""
    return RepairSummary.model_validate(await state.repair.summary())


@router.post("/repair/tasks", status_code=201)
async def repair_task(body: RepairRequest, state: State) -> RepairTaskAdded:
    """Write fixes for the best near misses as one task in Tasks, not started."""
    added = await state.repair.add_task(body.alphas, body.per_alpha, body.cores)
    if added is None:
        raise refuse(422, "nothing_to_repair", "No near miss is waiting for a repair.")
    async with state.db.session() as session:
        row = await session.get(Study, added.id)
        simulations = row.max_trials if row is not None else 0
    return RepairTaskAdded(id=added.id, name=added.name, simulations=simulations)
