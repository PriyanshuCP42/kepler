"""Autopilot: today's research funnel, the Pre-screen, and the Critic's shortlist."""

from collections import defaultdict
from datetime import timedelta
from typing import Any

from fastapi import APIRouter
from sqlalchemy import func, select

from ..brain.filters import platform_midnight
from ..db.models import CriticVerdict, SimulationRecord, Study
from ..labs.launch import lab_of
from ..labs.objectives import OBJECTIVES
from ..schemas import Out
from .agents import ShortlistEntry, shortlist
from .deps import State, refuse

router = APIRouter(prefix="/api/autopilot", tags=["autopilot"])


class Stage(Out):
    key: str
    label: str
    count: int | None
    note: str


class PreScreen(Out):
    status: str
    active: bool
    reason: str
    trained_at: str | None
    samples: int
    positives: int
    auc: float | None
    lift: float | None
    target: str
    oversample: int
    explore_share: float
    screened_today: int
    kept_today: int


class LabYield(Out):
    lab: str
    objective: str
    tasks: int
    #: Finished simulations the Critic has judged.
    simulations: int
    passed: int
    near_misses: int
    #: Critic passes per 1,000 judged simulations.
    per_thousand: float


class Autopilot(Out):
    funnel: list[Stage]
    prescreen: PreScreen
    yields: list[LabYield]
    shortlist: list[ShortlistEntry]


class Started(Out):
    started: bool


@router.get("")
async def overview(state: State) -> Autopilot:
    """Each stage of today's funnel, counted since the platform's day began."""
    since = platform_midnight()
    landed = CriticVerdict.landed_at >= since
    checks_ok = func.json_extract(CriticVerdict.gates, "$.checks.ok") == 1
    async with state.db.session() as session:

        async def count(*where: Any) -> int:
            found = await session.scalar(
                select(func.count()).select_from(CriticVerdict).where(*where)
            )
            return int(found or 0)

        judged = await count(landed)
        checks = await count(landed, checks_ok)
        passed = await count(landed, CriticVerdict.verdict == "pass")
        families = await count(
            landed,
            CriticVerdict.verdict == "pass",
            CriticVerdict.family_id == CriticVerdict.alpha_id,
        )
        leaders = await count(
            CriticVerdict.verdict == "pass", CriticVerdict.family_id == CriticVerdict.alpha_id
        )
    screener = state.screener
    funnel = [
        Stage(
            key="drawn",
            label="Candidates written",
            count=screener.drawn_today,
            note="Template and Search Lab points, before the Pre-screen",
        ),
        Stage(
            key="sent",
            label="Sent to BRAIN",
            count=await state.tracker.used_today(),
            note="Simulations charged to today's quota, every lab",
        ),
        Stage(key="judged", label="Finished and judged", count=judged, note="Scored by the Critic"),
        Stage(key="checks", label="Passed BRAIN's checks", count=checks, note="Before correlation"),
        Stage(
            key="critic",
            label="Passed the Critic",
            count=passed,
            note="Holds up out of sample, covers its costs",
        ),
        Stage(
            key="families",
            label="New distinct families",
            count=families,
            note="Below 0.6 with your pool and each other",
        ),
        Stage(
            key="shortlist", label="Shortlist", count=leaders, note="One alpha per family, all time"
        ),
    ]
    listed = await shortlist(state, limit=30)
    return Autopilot(
        funnel=funnel,
        prescreen=PreScreen.model_validate(screener.summary()),
        yields=await lab_yields(state),
        shortlist=listed.alphas,
    )


@router.post("/prescreen/train", status_code=202)
async def train(state: State) -> Started:
    """Retrain the Pre-screen on the vault now. Runs in the background; costs no quota."""
    if not state.screener.retrain_in_background():
        raise refuse(409, "busy", "The Pre-screen is already training.")
    return Started(started=True)


#: How far back the lab yields look.
YIELD_DAYS = 14


async def lab_yields(state: Any) -> list[LabYield]:
    """Critic passes per 1,000 simulations, by lab and by what its tasks maximised.

    Split by objective so a lab scored on the Critic Score can be set against the same lab
    scored on Train Sharpe. Passes count after the Family Clerk: a pool repeat is not one.
    """
    since = platform_midnight() - timedelta(days=YIELD_DAYS)
    async with state.db.session() as session:
        rows = (
            await session.execute(
                select(SimulationRecord.task, Study.objectives, CriticVerdict.verdict)
                .join(CriticVerdict, CriticVerdict.alpha_id == SimulationRecord.alpha_id)
                .join(Study, Study.task == SimulationRecord.task)
                .where(SimulationRecord.created_at >= since)
            )
        ).all()
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"tasks": set(), "simulations": 0, "passed": 0, "near_misses": 0}
    )
    for task, objectives, verdict in rows:
        lab = lab_of(task)
        if lab is None:
            continue
        key = (objectives or [None])[0]
        objective = OBJECTIVES[key].label if key in OBJECTIVES else str(key)
        group = groups[lab, objective]
        group["tasks"].add(task)
        group["simulations"] += 1
        group["passed"] += verdict == "pass"
        group["near_misses"] += verdict == "near_miss"
    out = [
        LabYield(
            lab=lab,
            objective=objective,
            tasks=len(g["tasks"]),
            simulations=g["simulations"],
            passed=g["passed"],
            near_misses=g["near_misses"],
            per_thousand=round(1000 * g["passed"] / g["simulations"], 1),
        )
        for (lab, objective), g in groups.items()
    ]
    return sorted(out, key=lambda y: y.per_thousand, reverse=True)
