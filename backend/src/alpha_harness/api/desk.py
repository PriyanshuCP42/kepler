"""The Submission Desk: what each shortlisted Alpha would do to your pool if you submitted it.

Every candidate is a Critic pass that leads its family: it passed BRAIN's checks, held up in
the test years, covers its costs, and sits below 0.6 with your pool. The desk adds what
decides whether to submit it:

- **Combined performance, before and after**, worked out here from stored daily PnL as
  BRAIN combines a pool: equal weight over the Alphas trading each day. Shown for your whole
  pool and for the partition it would join (region, delay, Power Pool or Regular), the one
  BRAIN's own before-and-after figure reproduces to the dollar.
- **Production Correlation**, which only BRAIN can measure, against every user's Alphas. Read
  from BRAIN's answers already kept; the desk can ask BRAIN for the missing ones.
- **The pyramid it would count towards**, from the category of its first data field.

Then a verdict: ready, check first, or blocked. Submitting stays a click on BRAIN.
"""

import asyncio
from dataclasses import asdict
from datetime import date
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..agents import guard
from ..brain.platform import (
    POWER_POOL_CLASS,
    PROD_CORRELATION_CEILING,
    PYRAMID_MIN_ALPHAS,
    SELF_CORRELATION_CEILING,
)
from ..db.models import CriticVerdict
from ..labs.fastexpr import ParseError, data_fields, parse
from ..schemas import Out
from ..tools import portfolio, power_pool
from ..vault.store import json_list
from ..vault.yields import checks_of, is_submitted
from .deps import State, refuse
from .lab_tasks import AlphaIds, WorkflowStarted, check_prod_correlation, prod_correlations

router = APIRouter(prefix="/api/desk", tags=["desk"])

#: Candidates the desk weighs, best Critic score first.
CANDIDATES = 40
WHOLE = "all"

Readiness = Literal["ready", "check", "hurts", "fading", "overfit", "blocked"]


class Figures(Out):
    sharpe: float | None
    fitness: float | None
    returns: float
    drawdown: float
    turnover: float
    margin: float


class Effect(Out):
    #: ``all`` or the partition, e.g. ``USA · D1 · Regular``.
    scope: str
    members: int
    before: Figures | None
    after: Figures | None
    #: The same over :data:`..agents.guard.RECENT_YEARS`, the nearest days to those BRAIN's
    #: Combined Alpha Performance is measured on.
    recent_before: Figures | None = None
    recent_after: Figures | None = None


class YearSharpe(Out):
    year: int
    sharpe: float | None


class Pyramid(Out):
    name: str
    submitted: int
    #: Submitting this one makes it :data:`PYRAMID_MIN_ALPHAS`.
    completes: bool
    multiplier: float | None


class NeighbourOut(Out):
    kind: str
    name: str
    #: queued | done | failed
    state: str
    sharpe: float | None
    test_sharpe: float | None
    holds: bool | None


class GuardView(Out):
    """The Overfitting Guard on one candidate: see :mod:`..agents.guard`."""

    status: guard.Status
    held: int
    measured: int
    neighbours: list[NeighbourOut]
    #: Simulations of the same idea in its region, and the test Sharpe chance alone reaches.
    siblings: int
    luck_bar: float
    years: int
    positive_years: int
    test_days: int | None
    test_without_spike: float | None
    #: Its Sharpe in each recent year, over them together, and over its whole history.
    recent_years: list[YearSharpe] = Field(default_factory=list)
    recent_sharpe: float | None = None
    full_sharpe: float | None = None


class Candidate(Out):
    alpha_id: str
    region: str | None
    delay: int | None
    universe: str | None
    expression: str | None
    power_pool: bool
    critic_score: float
    sharpe: float | None
    fitness: float | None
    train_sharpe: float | None
    test_sharpe: float | None
    #: Highest with your submitted Alphas, from the Family Clerk.
    pool_correlation: float | None
    #: BRAIN's, against every user's Alphas; ``None`` until asked.
    prod_correlation: float | None
    effects: list[Effect]
    guard: GuardView
    pyramid: Pyramid | None
    readiness: Readiness
    reasons: list[str]


class Desk(Out):
    candidates: list[Candidate]
    submitted: int
    #: Candidates without stored daily PnL, so without an effect yet.
    unmeasured: int
    prod_ceiling: float
    self_ceiling: float
    #: Your whole pool's combined Sharpe in each backtest year.
    pool_years: list[YearSharpe]
    recent_from: int
    recent_to: int


def partition_of(region: Any, delay: Any, power: bool) -> str:
    return f"{region} · D{delay} · {'Power Pool' if power else 'Regular'}"


def overfitting(view: GuardView, test_sharpe: float | None) -> list[str]:
    """Why the Overfitting Guard doubts it; empty when it does not."""
    doubts: list[str] = []
    if view.status == "fragile":
        doubts.append(
            f"Only {view.held} of {view.measured} neighbours kept {guard.HOLD:.0%} of its Sharpe."
        )
    if test_sharpe is not None and test_sharpe < view.luck_bar:
        doubts.append(
            f"Test Sharpe {test_sharpe:.2f} is under {view.luck_bar:.2f}, what the best of "
            f"{view.siblings} variants of this idea reaches by chance."
        )
    if view.test_days is not None and view.test_days < guard.MIN_TEST_DAYS:
        doubts.append(
            f"Trades on only {view.test_days} days of the test years; "
            f"at least {guard.MIN_TEST_DAYS} are needed to trust its Sharpe."
        )
    if view.years and view.positive_years < guard.POSITIVE_YEARS * view.years:
        doubts.append(
            f"Made money in only {view.positive_years} of the {view.years} years it traded in."
        )
    if view.test_without_spike is not None and view.test_without_spike <= 0:
        doubts.append(f"The test years lose money without their best {guard.SPIKE_DAYS} days.")
    return doubts


def fading(view: GuardView) -> list[str]:
    """Why it looks to be dying out in the last backtest years; empty when it does not."""
    span = guard.RECENT_SPAN
    found: list[str] = []
    weak = [
        y for y in view.recent_years if y.sharpe is not None and y.sharpe < guard.MIN_YEAR_SHARPE
    ]
    if weak:
        years = ", ".join(f"{y.year} ({y.sharpe:.2f})" for y in weak)
        found.append(
            f"Weak in {years}: every year of {span} needs a Sharpe of {guard.MIN_YEAR_SHARPE}."
        )
    recent, full = view.recent_sharpe, view.full_sharpe
    if recent is not None and full is not None and full > 0 and recent < guard.KEEP_RECENT * full:
        found.append(
            f"Fading: {span} Sharpe {recent:.2f} keeps only {recent / full:.0%} of its "
            f"whole-history {full:.2f}."
        )
    return found


_GUARD_PENDING = {
    "untested": "Neighbour check not run yet.",
    "running": "Neighbour check still running.",
    "inconclusive": "Too few neighbours came back to judge; run it again later.",
}


def judge(
    prod: float | None,
    pool: float | None,
    effects: list[Effect],
    view: GuardView | None = None,
    test_sharpe: float | None = None,
) -> tuple[Readiness, list[str]]:
    """Ready, check first, lowers the pool, overfit, or blocked, with why."""
    if prod is not None and prod >= PROD_CORRELATION_CEILING:
        return "blocked", [
            f"Production Correlation {prod:.2f} is over BRAIN's {PROD_CORRELATION_CEILING}."
        ]
    if pool is not None and pool >= SELF_CORRELATION_CEILING:
        return "blocked", [
            f"Correlates {pool:.2f} with your pool; BRAIN's limit is {SELF_CORRELATION_CEILING}."
        ]
    reasons: list[str] = []
    hurts = False
    span = guard.RECENT_SPAN
    for effect in effects:
        # Judged on the recent years where they are known: the nearest to what BRAIN scores.
        recent = effect.recent_before is not None and effect.recent_after is not None
        before, after = (
            (effect.recent_before, effect.recent_after) if recent else (effect.before, effect.after)
        )
        if before is None or after is None or before.sharpe is None or after.sharpe is None:
            continue
        change = after.sharpe - before.sharpe
        where = "your pool" if effect.scope == WHOLE else effect.scope
        when = f" in {span}" if recent else ""
        reasons.append(
            f"Combined Sharpe of {where}{when} {before.sharpe:.2f} → {after.sharpe:.2f}."
        )
        hurts = hurts or change < 0
    doubts = overfitting(view, test_sharpe) if view is not None else []
    if doubts:
        return "overfit", [*doubts, *reasons]
    dying = fading(view) if view is not None else []
    if dying:
        return "fading", [*dying, *reasons]
    if hurts:
        return "hurts", reasons
    pending = [] if prod is not None else ["Production Correlation not checked yet."]
    if view is not None and view.status in _GUARD_PENDING:
        pending.append(_GUARD_PENDING[view.status])
    if pending:
        return "check", [*pending, *reasons]
    held = [] if view is None else [f"{view.held} of {view.measured} neighbours held."]
    return "ready", [f"Production Correlation {prod:.2f}.", *held, *reasons]


def _figures(raw: dict[str, Any] | None) -> Figures | None:
    return None if raw is None else Figures.model_validate(raw)


def _first_category(expression: str | None, categories: dict[str, str]) -> str | None:
    if not expression:
        return None
    try:
        names = data_fields(parse(expression))
    except ParseError:
        return None
    return next((categories[n] for n in names if n in categories), None)


async def _pyramids(state: Any) -> tuple[dict[str, str], dict[tuple[str, int, str], Any]]:
    """Field to category, and each pyramid's niche: submitted this quarter and multiplier."""
    categories = {
        str(r["field_id"]): str(r["c"]).lower()
        for r in await state.catalog.query(
            "SELECT field_id, any_value(category_id) AS c FROM data_field "
            "WHERE category_id IS NOT NULL GROUP BY field_id"
        )
    }
    # Signed out, the grid falls back to the vault; a failure leaves the pyramid blank.
    try:
        niches = {n.key: n for n in await state.strategist.niche_map()}
    except Exception:  # noqa: BLE001
        niches = {}
    return categories, niches


@router.get("")
async def desk(state: State) -> Desk:
    """The best unsubmitted Critic passes, one per family, with their effect on your pool."""
    async with state.db.session() as session:
        verdicts = (
            await session.scalars(
                select(CriticVerdict)
                .where(
                    CriticVerdict.verdict == "pass",
                    CriticVerdict.family_id == CriticVerdict.alpha_id,
                )
                .order_by(CriticVerdict.score.desc())
                .limit(CANDIDATES * 2)
            )
        ).all()
    stored = await state.alphas.by_ids([v.alpha_id for v in verdicts])
    chosen = [v for v in verdicts if v.alpha_id in stored and not is_submitted(stored[v.alpha_id])]
    chosen = chosen[:CANDIDATES]

    members = await state.alphas.submitted_members()
    scopes: dict[str, list[str]] = {WHOLE: [str(m["alpha_id"]) for m in members]}
    for m in members:
        power = POWER_POOL_CLASS in json_list(m.get("classifications"))
        name = partition_of(m["region"], m["delay"], power)
        scopes.setdefault(name, []).append(str(m["alpha_id"]))

    powered = {v.alpha_id: power_pool.is_power_pool(stored[v.alpha_id]) for v in chosen}
    joins = {
        v.alpha_id: [
            WHOLE,
            partition_of(
                stored[v.alpha_id]["region"], stored[v.alpha_id]["delay"], powered[v.alpha_id]
            ),
        ]
        for v in chosen
    }
    # A partition none of yours is in yet starts with this one alone.
    for joined in joins.values():
        scopes.setdefault(joined[1], [])
    ids = [*scopes[WHOLE], *joins]
    series = await state.alphas.series(ids)
    meta = await state.alphas.by_ids(scopes[WHOLE]) | stored
    effects = await asyncio.to_thread(
        portfolio.what_if, series, meta, scopes, joins, guard.RECENT_START
    )
    pool = await asyncio.to_thread(portfolio.compute, series, meta, scopes[WHOLE], 0.0)

    rows = {v.alpha_id: stored[v.alpha_id] for v in chosen}
    verdicts_of = await state.guard.verdicts(rows)
    siblings = await state.guard.siblings(rows)

    checks = {v.alpha_id: checks_of(stored[v.alpha_id].get("checks")) for v in chosen}
    prod = await prod_correlations(state, list(checks), checks)
    categories, niches = await _pyramids(state)

    out: list[Candidate] = []
    for v in chosen:
        row = stored[v.alpha_id]
        found = [
            Effect(
                scope=scope,
                members=e["members"],
                before=_figures(e["before"]),
                after=_figures(e["after"]),
                recent_before=_figures(e["recent_before"]),
                recent_after=_figures(e["recent_after"]),
            )
            for scope, e in effects.get(v.alpha_id, {}).items()
        ]
        pyramid = None
        category = _first_category(row.get("expression"), categories)
        if category is not None:
            niche = niches.get((str(row["region"]), int(row["delay"] or 0), category))
            submitted = niche.submitted if niche else 0
            label = (niche.category_name if niche else category).upper()
            pyramid = Pyramid(
                name=f"{row['region']}/D{row['delay']}/{label}",
                submitted=submitted,
                completes=submitted + 1 == PYRAMID_MIN_ALPHAS,
                multiplier=niche.multiplier if niche else None,
            )
        view = _guard_view(
            verdicts_of[v.alpha_id], siblings.get(v.alpha_id, 1), series.get(v.alpha_id), row
        )
        readiness, reasons = judge(
            prod.get(v.alpha_id), v.pool_correlation, found, view, row.get("test_sharpe")
        )
        out.append(
            Candidate(
                alpha_id=v.alpha_id,
                region=row.get("region"),
                delay=row.get("delay"),
                universe=row.get("universe"),
                expression=row.get("expression"),
                power_pool=powered[v.alpha_id],
                critic_score=v.score,
                sharpe=row.get("sharpe"),
                fitness=row.get("fitness"),
                train_sharpe=row.get("train_sharpe"),
                test_sharpe=row.get("test_sharpe"),
                pool_correlation=v.pool_correlation,
                prod_correlation=prod.get(v.alpha_id),
                effects=found,
                guard=view,
                pyramid=pyramid,
                readiness=readiness,
                reasons=reasons,
            )
        )
    out.sort(key=_rank, reverse=True)
    return Desk(
        candidates=out,
        submitted=len(scopes[WHOLE]),
        unmeasured=sum(1 for c in out if not c.effects),
        prod_ceiling=PROD_CORRELATION_CEILING,
        self_ceiling=SELF_CORRELATION_CEILING,
        pool_years=[
            YearSharpe(year=int(y["year"]), sharpe=y.get("sharpe")) for y in pool["yearly"]
        ],
        recent_from=guard.RECENT_YEARS[0],
        recent_to=guard.RECENT_YEARS[-1],
    )


def _guard_view(
    found: guard.Verdict,
    siblings: int,
    days: dict[date, tuple[float, float]] | None,
    row: dict[str, Any],
) -> GuardView:
    test_start = row.get("test_start")
    steady = guard.consistency(
        {d: p for d, (p, _) in (days or {}).items()},
        test_start if isinstance(test_start, date) else None,
    )
    form = guard.recent_form(days or {})
    return GuardView(
        status=found.status,
        held=found.held,
        measured=found.measured,
        neighbours=[NeighbourOut.model_validate(asdict(n)) for n in found.neighbours],
        siblings=siblings,
        luck_bar=round(guard.luck_bar(siblings), 2),
        years=steady.years,
        positive_years=steady.positive_years,
        test_days=steady.test_days,
        test_without_spike=steady.test_without_spike,
        recent_years=[YearSharpe(year=y, sharpe=v) for y, v in form.years.items()],
        recent_sharpe=form.recent,
        full_sharpe=form.full,
    )


_ORDER: dict[Readiness, int] = {
    "ready": 5,
    "check": 4,
    "hurts": 3,
    "fading": 2,
    "overfit": 1,
    "blocked": 0,
}


def _rank(candidate: Candidate) -> tuple[int, float]:
    """Ready first, then by how much it lifts your whole pool's combined Sharpe in the recent
    years, the nearest to what BRAIN scores."""
    change = sharpe_change(candidate.effects, WHOLE, recent=True)
    if change is None:
        change = sharpe_change(candidate.effects, WHOLE)
    return _ORDER[candidate.readiness], change or 0.0


def sharpe_change(effects: list[Effect], scope: str, *, recent: bool = False) -> float | None:
    """How much joining moves this scope's combined Sharpe, over every year or the recent ones."""
    for e in effects:
        old, new = (e.recent_before, e.recent_after) if recent else (e.before, e.after)
        before = old.sharpe if old else None
        after = new.sharpe if new else None
        if e.scope == scope and before is not None and after is not None:
            return after - before
    return None


@router.post("/prod-correlation", status_code=202)
async def check_candidates(state: State) -> WorkflowStarted:
    """Ask BRAIN for the Production Correlation of every candidate it has not given one for.
    BRAIN limits these per hour; no simulation quota."""
    found = await desk(state)
    wanted = [c.alpha_id for c in found.candidates if c.prod_correlation is None]
    if not wanted:
        raise refuse(409, "nothing_to_check", "Every candidate has its Production Correlation.")
    return await check_prod_correlation(AlphaIds.model_validate({"alphaIds": wanted}), state)


class GuardRequest(BaseModel):
    #: Candidates checked, best first among those not ruled out already.
    limit: int = Field(default=20, ge=1, le=40)
    cores: int = Field(default=4, ge=1, le=8)


class GuardTask(Out):
    id: int
    name: str
    finalists: int


@router.post("/guard", status_code=201)
async def guard_task(body: GuardRequest, state: State) -> GuardTask:
    """Write the neighbours of the best candidates not yet checked as one task in Tasks, not
    started. Skips the blocked, the overfit and those that lower your pool."""
    found = await desk(state)
    wanted = [
        c.alpha_id
        for c in found.candidates
        if c.readiness in ("ready", "check") and c.guard.status == "untested"
    ][: body.limit]
    added = await state.guard.add_task(wanted, body.cores) if wanted else None
    if added is None:
        raise refuse(409, "nothing_to_check", "Every candidate worth checking has its neighbours.")
    return GuardTask(id=added.id, name=added.name, finalists=len(wanted))
