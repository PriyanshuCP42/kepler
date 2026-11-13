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


# --- Today's Picks -------------------------------------------------------------------------


class Pick(Out):
    """One submittable Alpha as the Submit Gate judged it."""

    alpha_id: str
    region: str | None
    delay: int | None
    universe: str | None
    sharpe: float | None
    fitness: float | None
    returns: float | None
    drawdown: float | None
    return_to_drawdown: float | None
    margin: float | None
    turnover: float | None
    test_sharpe: float | None
    prod_correlation: float | None
    self_correlation: float | None
    power_pool_correlation: float | None
    operators: int | None
    fields: int | None
    power_pool: bool
    regular: bool
    theme_match: bool | None
    #: Change in combined and in Power Pool Sharpe if it joins.
    whole_delta: float | None
    power_delta: float | None
    #: The recent submission whose dataset it repeats in the same market.
    lookalike_of: str | None
    #: The checked Alpha whose daily PnL this one moves with, and BRAIN's production
    #: correlation for it: what this one's check will most likely say.
    twin_of: str | None = None
    twin_prod: float | None = None
    datasets: list[str]
    #: READY, CHECK, REPAIR or REJECT.
    verdict: str
    reasons: list[str]
    score: float


class Picks(Out):
    measured: int
    ready: list[Pick]
    check: list[Pick]
    repair: list[Pick]
    rejected: int
    #: Why candidates were rejected, most common first.
    rejected_by: dict[str, int]


def to_pick(m: Any) -> Pick:
    row, j = m.row, m.judgement
    settings = row.settings or {}
    return Pick(
        alpha_id=str(row.alpha_id),
        region=settings.get("region"),
        delay=settings.get("delay"),
        universe=settings.get("universe"),
        sharpe=row.sharpe,
        fitness=row.fitness,
        returns=row.returns,
        drawdown=row.drawdown,
        return_to_drawdown=j.return_to_drawdown,
        margin=row.margin,
        turnover=row.turnover,
        test_sharpe=row.test_sharpe,
        prod_correlation=m.facts.prod_correlation,
        self_correlation=m.facts.self_correlation,
        power_pool_correlation=m.power_pool_correlation,
        operators=m.operators,
        fields=m.fields,
        power_pool=j.power_pool,
        regular=j.regular,
        theme_match=m.theme_match,
        whole_delta=m.whole_delta,
        power_delta=m.power_delta,
        lookalike_of=m.lookalike_of,
        twin_of=m.twin.alpha_id if m.twin else None,
        twin_prod=m.twin.prod_correlation if m.twin else None,
        datasets=sorted(m.datasets),
        verdict=str(j.verdict),
        reasons=j.reasons,
        score=j.score,
    )


async def todays_picks(state: Any) -> Picks:
    """The submittable Alphas the Submit Gate would submit, check or repair, best first."""
    from ..agents import picks
    from ..agents.gate import Verdict
    from .lab_tasks import submittable_alphas

    measured = await picks.measure(state, picks.shortlist(await submittable_alphas(state)))
    grouped: dict[Verdict, list[Pick]] = defaultdict(list)
    rejected_by: dict[str, int] = defaultdict(int)
    for m in measured:
        grouped[m.judgement.verdict].append(to_pick(m))
        if m.judgement.verdict is Verdict.REJECT and m.judgement.reasons:
            rejected_by[m.judgement.reasons[0]] += 1
    for found in grouped.values():
        found.sort(key=lambda p: -p.score)
    return Picks(
        measured=len(measured),
        ready=grouped[Verdict.READY],
        check=grouped[Verdict.CHECK],
        repair=grouped[Verdict.REPAIR],
        rejected=len(grouped[Verdict.REJECT]),
        rejected_by=dict(sorted(rejected_by.items(), key=lambda kv: -kv[1])),
    )


@router.get("/picks")
async def picks_today(state: State) -> Picks:
    """Today's Picks: submittable Alphas the Submit Gate calls Ready, ones to check with BRAIN
    first, and near misses to repair. Kepler never submits; you do."""
    return await todays_picks(state)


class AutopilotAlert(Out):
    kind: str
    title: str
    text: str


class AutopilotStatus(Out):
    enabled: bool
    daily_target: int
    #: The last round: when, the day's figures, what it did and why.
    last: dict[str, Any]
    alerts: list[AutopilotAlert]


@router.get("/status")
async def autopilot_status(state: State) -> AutopilotStatus:
    """What the Daily Autopilot last did, and the alerts it raised most recently."""
    pilot = state.autopilot
    return AutopilotStatus(
        enabled=pilot.enabled,
        daily_target=pilot.daily_target,
        last=pilot.last,
        alerts=[
            AutopilotAlert(kind=k, title=t, text=x)
            for k, t, x in reversed(pilot.alerts.raised[-10:])
        ],
    )
