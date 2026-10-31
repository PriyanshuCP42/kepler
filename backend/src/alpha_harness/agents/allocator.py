"""The learning layer: which labs get today's simulations, learned from what each one yields.

Every lab and agent that writes simulations is an **arm**: the Search Lab (on the pyramids the
Strategist plans), the Repair Engineer, the Hypothesis Writer and the Template Lab. What each
arm is worth is how many **good alphas** it makes per simulation, over the last
:data:`WINDOW_DAYS` days: alphas the Critic passes (BRAIN's checks, out-of-sample decay, cost,
size, then the Family Clerk's pool check) that also hold up in the last backtest years
(:func:`.guard.recent_form`), the nearest to the days BRAIN scores after submission.

The yield is a Beta posterior per arm, starting from a prior of :data:`PRIOR_GOOD` good alphas
in :data:`PRIOR_SIMULATIONS` simulations so an arm with little history is neither trusted nor
written off. Each day's split is a Thompson draw from those posteriors, seeded by the platform
day so the plan shown is the plan written, with :data:`EXPLORE` of the budget kept for each arm
that can take it so a lab that improves gets noticed. An arm takes no more than it can use (a
repair task is as big as the near misses waiting); what it cannot use goes to the Search Lab,
which can always take more. The Template Lab needs your template, so its share is a suggestion.

Budget: the day's operating ceiling, less what was sent today, what running tasks will still
send, and what tasks written today but not started would. Neighbour checks for the Submission
Desk come out of it first: they decide what is Ready.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from sqlalchemy import func, select

from ..brain.filters import platform_midnight
from ..db.models import (
    CriticVerdict,
    Hypothesis,
    SimulationRecord,
    Study,
    StudyStatus,
    Trial,
    TrialState,
)
from ..labs.launch import lab_of
from . import guard

if TYPE_CHECKING:
    from .bus import AgentBus

log = structlog.get_logger(__name__)

WINDOW_DAYS = 14
#: The prior: half a good alpha per hundred simulations, worth this many simulations of data.
PRIOR_GOOD = 1.0
PRIOR_SIMULATIONS = 200
#: Budget share kept for each arm that can take it, so none is starved of a chance.
EXPLORE = 0.1
#: Near-miss fixes per near miss, as the Repair Engineer writes them.
REPAIR_PER_ALPHA = 3
#: Simulations per neighbour check: one per neighbour.
GUARD_PER_ALPHA = 6
#: How often the plan is worked out again for Research HQ.
REFRESH_SECONDS = 1800
#: Candidates checked per plan.
GUARD_LIMIT = 20
#: The least the Strategist gives one pyramid's Search Lab task.
SEARCH_PER_NICHE = 100

SEARCH, REPAIR, HYPOTHESIS, TEMPLATE = "search", "repair", "hypothesis", "template"
#: Each arm, the lab name its tasks carry, and what it says in the plan.
ARMS: dict[str, tuple[str, str]] = {
    SEARCH: ("Search Lab", "Search on the pyramids the Strategist plans"),
    REPAIR: ("Repair Engineer", "Fix near misses by what failed"),
    HYPOTHESIS: ("Hypothesis Writer", "Test new hypotheses from your LLM"),
    TEMPLATE: ("Template Lab", "Your own templates"),
}
ACTIVE_STATES = (StudyStatus.RUNNING, StudyStatus.QUEUED)
SPENT = (TrialState.QUEUED, TrialState.RUNNING, TrialState.COMPLETE, TrialState.FAIL)


@dataclass(slots=True)
class Arm:
    key: str
    lab: str
    action: str
    simulations: int = 0
    passed: int = 0
    good: int = 0
    #: Good alphas per 1,000 simulations: the posterior mean and its 5th and 95th percentile.
    expected: float = 0.0
    low: float = 0.0
    high: float = 0.0
    #: Today's draw, the share of the budget it gets, and those simulations.
    draw: float = 0.0
    share: float = 0.0
    planned: int = 0
    #: The most it can use today; ``None`` for no limit.
    capacity: int | None = None
    #: Written by you rather than by the plan.
    manual: bool = False
    note: str | None = None


@dataclass(slots=True)
class Plan:
    ceiling: int
    used: int
    pending: int
    budget: int
    guard: int
    arms: list[Arm] = field(default_factory=list)


def posterior(good: int, simulations: int) -> tuple[float, float]:
    """The Beta posterior's parameters for a yield of ``good`` in ``simulations``."""
    alpha = PRIOR_GOOD + good
    beta = PRIOR_SIMULATIONS - PRIOR_GOOD + max(0, simulations - good)
    return alpha, beta


def split(arms: list[Arm], budget: int, rng: random.Random) -> None:
    """Share ``budget`` between the arms by a Thompson draw, within what each can use.

    Every arm that can take simulations is first given :data:`EXPLORE` of the budget; the rest
    goes by the draws. An arm over its capacity gives the excess back, to be shared again among
    those still under theirs, and the Search Lab takes whatever is left at the end.
    """
    for arm in arms:
        a, b = posterior(arm.good, arm.simulations)
        arm.draw = rng.betavariate(a, b)
    open_arms = [a for a in arms if a.capacity is None or a.capacity > 0]
    if not open_arms or budget <= 0:
        return
    floor = EXPLORE * budget
    rest = budget - floor * len(open_arms)
    total = sum(a.draw for a in open_arms) or 1.0
    wanted = {a.key: floor + max(0.0, rest) * a.draw / total for a in open_arms}
    # Settle the capped arms, then share what they gave back among the others.
    left = float(budget)
    free = list(open_arms)
    while True:
        over = [a for a in free if a.capacity is not None and wanted[a.key] > a.capacity]
        if not over:
            break
        for arm in over:
            wanted[arm.key] = float(arm.capacity or 0)
            left -= wanted[arm.key]
            free.remove(arm)
        weight = sum(wanted[a.key] for a in free) or 1.0
        scale = left / weight
        for arm in free:
            wanted[arm.key] *= scale
    for arm in open_arms:
        arm.planned = int(wanted[arm.key])
    spare = budget - sum(a.planned for a in open_arms)
    search = next((a for a in arms if a.key == SEARCH), None)
    if search is not None and spare > 0:
        search.planned += spare
    for arm in arms:
        arm.share = arm.planned / budget if budget else 0.0


class Allocator:
    def __init__(self, state: Any, bus: AgentBus) -> None:
        self.state = state
        self.bus = bus
        self._task: asyncio.Task[None] | None = None

    async def plan(self) -> Plan:
        ceiling = int(self.state.settings.simulation_operating_ceiling)
        used = int(await self.state.tracker.used_today())
        pending = await self._pending()
        guard_ids = await self._unchecked()
        guard_sims = min(len(guard_ids) * GUARD_PER_ALPHA, max(0, ceiling - used - pending))
        budget = max(0, ceiling - used - pending - guard_sims)

        found = await self._yields()
        arms = []
        for key, (lab, action) in ARMS.items():
            sims, passed, good = found.get(lab, (0, 0, 0))
            a, b = posterior(good, sims)
            draws = np.random.default_rng(0).beta(a, b, 4000)
            arms.append(
                Arm(
                    key=key,
                    lab=lab,
                    action=action,
                    simulations=sims,
                    passed=passed,
                    good=good,
                    expected=round(1000 * a / (a + b), 2),
                    low=round(1000 * float(np.quantile(draws, 0.05)), 2),
                    high=round(1000 * float(np.quantile(draws, 0.95)), 2),
                )
            )
        await self._capacities(arms)
        rng = random.Random(platform_midnight().date().toordinal())
        split(arms, budget, rng)
        plan = Plan(ceiling, used, pending, budget, guard_sims, arms)
        self._report(plan)
        return plan

    def _report(self, plan: Plan) -> None:
        """What the Learning Analyst shows in Research HQ."""
        learned = [a for a in plan.arms if a.simulations]
        best = max(learned, key=lambda a: a.expected, default=None)
        if best is not None:
            self.bus.set_stat(
                "learning", "best lab", f"{best.lab}: {best.expected:.1f} good per 1,000"
            )
        self.bus.set_stat("learning", "simulations to plan", plan.budget)
        top = max(plan.arms, key=lambda a: a.planned, default=None)
        if top is not None and plan.budget:
            self.bus.say(
                "learning",
                f"Today's {plan.budget:,} simulations: most to the {top.lab} "
                f"({top.planned:,}), learned from {WINDOW_DAYS} days of results.",
            )
        elif not plan.budget:
            self.bus.say("learning", "Today's simulations are all spoken for.")

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="learning-analyst")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        """Keep what was learned fresh in Research HQ, even with Autopilot closed."""
        await asyncio.sleep(90)
        while True:
            try:
                await self.plan()
            except Exception:
                log.warning("learning.plan_failed", exc_info=True)
            await asyncio.sleep(REFRESH_SECONDS)

    async def write(self, cores: int, start: bool) -> dict[str, Any]:
        """Write today's tasks as planned; with ``start``, queue them to run."""
        # Imported here: the API layer builds on the agents.
        from ..api.lab_tasks import requeue

        plan = await self.plan()
        written: list[dict[str, Any]] = []
        notes: list[str] = []
        by = {a.key: a for a in plan.arms}

        if plan.guard:
            ids = (await self._unchecked())[: plan.guard // GUARD_PER_ALPHA or 1]
            added = await self.state.guard.add_task(ids, cores) if ids else None
            if added is not None:
                written.append({"id": added.id, "name": added.name, "arm": "guard"})

        repair = by[REPAIR]
        if repair.planned >= REPAIR_PER_ALPHA:
            added = await self.state.repair.add_task(
                repair.planned // REPAIR_PER_ALPHA, REPAIR_PER_ALPHA, cores
            )
            if added is not None:
                written.append({"id": added.id, "name": added.name, "arm": REPAIR})

        hypothesis = by[HYPOTHESIS]
        if hypothesis.planned > 0:
            tested = await self.state.hypotheses.add_task(cores)
            if tested is not None:
                task, _ = tested
                written.append({"id": task.id, "name": task.name, "arm": HYPOTHESIS})
        if not hypothesis.capacity and self.state.hypotheses.write_in_background():
            notes.append("No hypotheses were waiting, so the Hypothesis Writer is writing some.")

        search = by[SEARCH]
        if search.planned >= SEARCH_PER_NICHE:
            # The Strategist gives each pyramid at least that many, so fewer for a small share;
            # but two or more, so one pyramid it cannot search does not waste the whole share.
            niches = max(2, min(5, search.planned // SEARCH_PER_NICHE))
            result = await self.state.strategist.add_tasks(niches, search.planned, cores)
            written.extend(
                {"id": t["id"], "name": t["niche"], "arm": SEARCH} for t in result["added"]
            )
            notes.extend(f"{s['niche']}: {s['reason']}" for s in result["skipped"])

        template = by[TEMPLATE]
        if template.planned:
            notes.append(
                f"Template Lab: about {template.planned:,} simulations are left for a template "
                "task of your own."
            )
        if start and written:
            await requeue(self.state, [t["id"] for t in written])
        for arm, target in ((SEARCH, "strategist"), (REPAIR, "repair"), (HYPOTHESIS, "hypothesis")):
            tasks = sum(1 for t in written if t["arm"] == arm)
            self.bus.handoff("learning", target, "plan", tasks, "today's share")
        self.bus.say(
            "learning",
            f"Wrote today's plan: {len(written)} tasks"
            + (", running." if start else "; run them from Tasks."),
        )
        return {"tasks": written, "notes": notes, "started": bool(start and written)}

    # -- what was learned ----------------------------------------------------

    async def _yields(self) -> dict[str, tuple[int, int, int]]:
        """Per lab name: judged simulations, Critic passes, and passes that hold up recently."""
        since = platform_midnight() - timedelta(days=WINDOW_DAYS)
        async with self.state.db.session() as session:
            rows = (
                await session.execute(
                    select(SimulationRecord.task, CriticVerdict.alpha_id, CriticVerdict.verdict)
                    .join(CriticVerdict, CriticVerdict.alpha_id == SimulationRecord.alpha_id)
                    .where(SimulationRecord.created_at >= since)
                )
            ).all()
        sims: dict[str, int] = defaultdict(int)
        passes: dict[str, list[str]] = defaultdict(list)
        for task, alpha_id, verdict in rows:
            lab = lab_of(task)
            if lab is None:
                continue
            sims[lab] += 1
            if verdict == "pass":
                passes[lab].append(str(alpha_id))
        every = [a for ids in passes.values() for a in ids]
        series = await self.state.alphas.series(every) if every else {}
        out: dict[str, tuple[int, int, int]] = {}
        for lab, n in sims.items():
            good = 0
            for alpha_id in passes[lab]:
                form = guard.recent_form(series.get(alpha_id) or {})
                if form.recent is not None and not form.fading and not form.weak_years:
                    good += 1
            out[lab] = (n, len(passes[lab]), good)
        return out

    async def _pending(self) -> int:
        """Simulations running tasks will still send today, and tasks written today not started:
        a plan written but not yet run must not be planned again."""
        async with self.state.db.session() as session:
            studies = list(
                await session.scalars(
                    select(Study).where(
                        Study.status.in_(ACTIVE_STATES)
                        | (
                            (Study.status == StudyStatus.IDLE)
                            & (Study.created_at >= platform_midnight())
                        )
                    )
                )
            )
            if not studies:
                return 0
            counted = await session.execute(
                select(Trial.study_id, func.count())
                .where(Trial.study_id.in_([s.id for s in studies]), Trial.state.in_(SPENT))
                .group_by(Trial.study_id)
            )
            spent = {int(study_id): int(n) for study_id, n in counted.all()}
        return sum(max(0, int(s.max_trials or 0) - int(spent.get(s.id, 0))) for s in studies)

    async def _capacities(self, arms: list[Arm]) -> None:
        by = {a.key: a for a in arms}
        waiting = len(await self.state.repair.waiting())
        by[REPAIR].capacity = waiting * REPAIR_PER_ALPHA
        by[REPAIR].note = f"{waiting:,} near misses waiting"
        async with self.state.db.session() as session:
            ready = list(
                await session.scalars(
                    select(Hypothesis).where(
                        Hypothesis.enabled.is_(True), Hypothesis.study_id.is_(None)
                    )
                )
            )
        cap = sum(len(h.expressions or []) * max(1, len(h.neutralizations or [])) for h in ready)
        by[HYPOTHESIS].capacity = cap
        by[HYPOTHESIS].note = (
            f"{len(ready)} hypotheses waiting"
            if ready
            else "None waiting: the plan asks the Hypothesis Writer for more"
        )
        by[TEMPLATE].manual = True
        by[TEMPLATE].note = "Needs your template, so its share is a suggestion"

    async def _unchecked(self) -> list[str]:
        """Desk candidates worth a neighbour check that have not had one."""
        from ..api.desk import desk

        found = await desk(self.state)
        return [
            c.alpha_id
            for c in found.candidates
            if c.readiness in ("ready", "check") and c.guard.status == "untested"
        ][:GUARD_LIMIT]
