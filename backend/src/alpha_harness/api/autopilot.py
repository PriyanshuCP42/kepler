"""Autopilot: today's research funnel, the Pre-screen, and the Critic's shortlist."""

from typing import Any

from fastapi import APIRouter
from sqlalchemy import func, select

from ..brain.filters import platform_midnight
from ..db.models import CriticVerdict
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
    oversample: int
    explore_share: float
    screened_today: int
    kept_today: int


class Autopilot(Out):
    funnel: list[Stage]
    prescreen: PreScreen
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
        shortlist=listed.alphas,
    )


@router.post("/prescreen/train", status_code=202)
async def train(state: State) -> Started:
    """Retrain the Pre-screen on the vault now. Runs in the background; costs no quota."""
    if not state.screener.retrain_in_background():
        raise refuse(409, "busy", "The Pre-screen is already training.")
    return Started(started=True)
