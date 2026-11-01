"""The Overfitting Guard: whether a finalist is a real signal or a lucky fit.

Passing the test years is not proof by itself. Thousands of variants are simulated, and a few
of them pass the test years by chance; the more a single idea is tried, the more that happens.
So before a finalist is called ready, three checks:

- **Neighbours.** The same idea re-run with small changes: its windows shorter and longer, a
  little more decay, tighter truncation, the next coarser neutralization, the next larger
  universe. A real signal
  sits on a plateau, and most neighbours keep most of its Sharpe; a fitted one sits on a spike,
  and its neighbours fall away. Costs a few simulations per finalist, written as one task.
  Three neighbours go further, each with its own bar: ``rank`` of it (no scale, no outliers),
  ``sign`` of it (direction only: an edge that lives in a few extreme values loses it all),
  and the same expression in another market, the strongest out-of-sample test there is:
  an idea fitted to one market's history rarely carries to another's.
- **The luck bar.** How high a test Sharpe the best of ``n`` variants of one idea reaches by
  chance alone: about ``sqrt(2 ln n)`` standard errors, where a Sharpe measured over ``T``
  years has a standard error of about ``1 / sqrt(T)``. Variants of one idea move together, so
  half of that is charged. ``n`` counts every simulation in the vault of the same expression
  with its numbers blanked out, in the same region.
- **Recent form.** BRAIN measures what counts, Combined Alpha Performance, on the days after
  the backtest ends (from 2024), and keeps those days hidden. The nearest days anyone can see
  are the last backtest years, so an Alpha must still be strong there: every one of
  :data:`RECENT_YEARS` at a Sharpe of :data:`MIN_YEAR_SHARPE` or more, and those years together
  keeping :data:`KEEP_RECENT` of its whole-history Sharpe. One that is fading into them is
  likely still fading after them.
- **Consistency.** It traded on enough days of the test years for its Sharpe to mean
  something, most years it traded in made money, and the test years still made money without
  their best days, so one lucky stretch is not carrying it.

Nothing here costs quota except the neighbours, and those only when asked for.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import structlog
from sqlalchemy import select

from ..brain.platform import MULTI_SIMULATION_SIZE, TEST_PERIOD_YEARS
from ..db.models import GuardCheck, Trial, TrialState, utcnow
from ..labs.fastexpr import ParseError
from ..labs.launch import AddedTask, add_study
from ..labs.params import GUARD_SAMPLER, GuardParams
from ..tools import correlation_breaker, settings_sampler
from ..vault import metrics
from .repair import (
    Options,
    Settings,
    Transform,
    base_settings,
    checked_request,
    neutralization_step,
    options_for,
    setting_move,
    universe_step,
)

if TYPE_CHECKING:
    from ..brain.schemas import SimulationRequest
    from .bus import AgentBus

log = structlog.get_logger(__name__)

#: A neighbour holds when it keeps this share of the finalist's Sharpe.
HOLD = 0.7
#: ``sign`` throws away every magnitude, so it is held to less.
SIGN_HOLD = 0.5
#: Another market's level differs; carrying the idea at all is what is asked.
OTHER_MARKET_HOLD = 0.4
#: Where an idea is tried out of its own market: a sibling with different stocks.
OTHER_MARKET = {
    "DEU": "GBR",
    "GBR": "DEU",
    "EUR": "USA",
    "USA": "EUR",
    "GLB": "USA",
    "ASI": "USA",
    "AMR": "USA",
    "KOR": "HKG",
    "HKG": "KOR",
    "CHN": "HKG",
    "IND": "ASI",
}
#: Share of measured neighbours that must hold for a plateau.
PLATEAU = 2 / 3
#: Neighbours that must come back measured before a verdict.
MIN_MEASURED = 2
#: Share of the best-of-n chance Sharpe charged: variants of one idea are not independent.
LUCK_WEIGHT = 0.5
#: Share of whole years that must make money.
POSITIVE_YEARS = 0.6
#: A year counts once the Alpha traded on this many of its days.
ACTIVE_YEAR_DAYS = 100
#: The test years must still make money without this many of their best days.
SPIKE_DAYS = 10
#: Days of the test years the Alpha must trade on, about one in five: a Sharpe from a
#: handful of trades is luck, however high.
MIN_TEST_DAYS = 100

#: The last backtest years, the nearest to the hidden days Combined Alpha Performance is
#: measured on.
RECENT_YEARS = (2021, 2022, 2023)
RECENT_START = date(RECENT_YEARS[0], 1, 1)
RECENT_SPAN = f"{RECENT_YEARS[0]}-{RECENT_YEARS[-1]}"
#: Each recent year must reach this Sharpe.
MIN_YEAR_SHARPE = 0.5
#: The recent years together must keep this share of the whole-history Sharpe.
KEEP_RECENT = 0.75

Status = Literal["untested", "running", "robust", "fragile", "inconclusive"]

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
#: An integer window passed as a later argument: ``ts_mean(x, 20)``. Keyword values
#: (``std=4``) and names with digits (``adv20``) are left alone.
_WINDOW = re.compile(r"(,\s*)(\d+)(?=\s*[,)])")


# -- neighbours ----------------------------------------------------------------


def scale_windows(expression: str, factor: float) -> str | None:
    """Every integer window of 3 or more scaled by ``factor``; ``None`` when there is none."""
    changed = False

    def scale(match: re.Match[str]) -> str:
        nonlocal changed
        value = int(match.group(2))
        if value < 3:
            return match.group(0)
        new = max(2, round(value * factor))
        changed = changed or new != value
        return f"{match.group(1)}{new}"

    out = _WINDOW.sub(scale, expression)
    return out if changed else None


def _decay_up(settings: Settings, _: Options) -> Settings | None:
    return settings | {"decay": int(settings.get("decay") or 0) + 3}


def _truncation_tighter(settings: Settings, _: Options) -> Settings | None:
    current = float(settings.get("truncation") or 0.08)
    return settings | {"truncation": round(max(0.01, current * 0.75), 3)}


def _windows(factor: float) -> Transform:
    def apply(expression: str, settings: Settings, _: Options) -> tuple[str, Settings] | None:
        scaled = scale_windows(expression, factor)
        return None if scaled is None else (scaled, settings)

    return apply


tighter = setting_move(_truncation_tighter)
coarser = setting_move(neutralization_step(-1))


_TIGHTER = setting_move(_truncation_tighter)
_COARSER = setting_move(neutralization_step(-1))


def _around(tail: str) -> Transform:
    """The finalist bound to ``alpha`` and ``tail`` taken of it."""

    def apply(expression: str, settings: Settings, _: Options) -> tuple[str, Settings] | None:
        try:
            return correlation_breaker.compress(expression).program(tail), settings
        except ParseError:
            return None

    return apply


@dataclass(frozen=True, slots=True)
class Neighbour:
    id: str
    name: str
    apply: Transform | None
    #: Share of the finalist's Sharpe it must keep to hold.
    keep: float = HOLD


OTHER_MARKET_ID = "other_market"
NEIGHBOURS = (
    Neighbour("windows_down", "Windows 30% shorter", _windows(0.7)),
    Neighbour("windows_up", "Windows 40% longer", _windows(1.4)),
    Neighbour("decay_up", "Decay 3 days longer", setting_move(_decay_up)),
    Neighbour("tighter_truncation", "Truncation a quarter tighter", _TIGHTER),
    Neighbour("coarser_neutralization", "Next coarser neutralization", _COARSER),
    Neighbour("wider_universe", "Next larger universe", setting_move(universe_step(1))),
    Neighbour("rank", "Ranked", _around("rank(alpha)")),
    Neighbour("sign", "Direction only (sign)", _around("sign(alpha)"), SIGN_HOLD),
    # Built in :func:`neighbours_of`, which has the schema another market's options come from.
    Neighbour(OTHER_MARKET_ID, "Same idea in another market", None, OTHER_MARKET_HOLD),
)
NAMES: dict[str, str] = {n.id: n.name for n in NEIGHBOURS}
KEEPS: dict[str, float] = {n.id: n.keep for n in NEIGHBOURS}


def _largest(universes: tuple[str, ...]) -> str | None:
    """The broadest ``TOPn`` universe offered, else the first."""
    sized = sorted(
        (
            (int(m.group()), u)
            for u in universes
            if u.upper().startswith("TOP") and (m := re.search(r"\d+", u))
        ),
        reverse=True,
    )
    return sized[0][1] if sized else (universes[0] if universes else None)


def other_market(settings: Settings, schema: dict[str, Any]) -> Settings | None:
    """The finalist's settings moved to :data:`OTHER_MARKET`'s pick for its region: that
    market's broadest universe, its neutralization kept where offered, else industry."""
    region = OTHER_MARKET.get(str(settings.get("region")))
    if region is None:
        return None
    moved = settings | {"region": region}
    universe = _largest(options_for(schema, moved).universes)
    if universe is None:
        return None
    moved["universe"] = universe
    if moved.get("neutralization") not in options_for(schema, moved).neutralizations:
        moved["neutralization"] = "INDUSTRY"
    return moved


def neighbours_of(
    row: dict[str, Any], schema: dict[str, Any]
) -> list[tuple[str, SimulationRequest]]:
    """The finalist's neighbours BRAIN would accept, one per change that applies."""
    expression = row.get("expression")
    if not expression:
        return []
    base = base_settings(row)
    options = options_for(schema, base)
    out: list[tuple[str, SimulationRequest]] = []
    for neighbour in NEIGHBOURS:
        if neighbour.apply is None:
            continue
        built = neighbour.apply(expression, base, options)
        request = None if built is None else checked_request(*built, schema)
        if request is not None:
            out.append((neighbour.id, request))
    # A field the other market lacks makes BRAIN refuse it; the neighbour then tells nothing.
    moved = other_market(base, schema)
    request = None if moved is None else checked_request(expression, moved, schema)
    if request is not None:
        out.append((OTHER_MARKET_ID, request))
    return out


# -- the bars ------------------------------------------------------------------


def skeleton(expression: str) -> str:
    """The expression with its numbers blanked and spaces dropped: one idea, every variant."""
    return _NUMBER.sub("#", re.sub(r"\s+", "", expression))


def luck_bar(variants: int, years: float = TEST_PERIOD_YEARS) -> float:
    """The test Sharpe the best of ``variants`` tries reaches by chance, charged at
    :data:`LUCK_WEIGHT`. Zero for a single try."""
    if variants < 2 or years <= 0:
        return 0.0
    return LUCK_WEIGHT * math.sqrt(2 * math.log(variants)) / math.sqrt(years)


@dataclass(slots=True)
class Consistency:
    years: int = 0
    positive_years: int = 0
    #: Days of the test years with any PnL; ``None`` without test years.
    test_days: int | None = None
    #: Test-years PnL without its best :data:`SPIKE_DAYS` days; ``None`` without test years.
    test_without_spike: float | None = None

    @property
    def steady(self) -> bool:
        return self.years == 0 or self.positive_years >= POSITIVE_YEARS * self.years

    @property
    def unspiked(self) -> bool:
        return self.test_without_spike is None or self.test_without_spike > 0

    @property
    def active(self) -> bool:
        return self.test_days is None or self.test_days >= MIN_TEST_DAYS


def consistency(days: dict[date, float], test_start: date | None) -> Consistency:
    """Years it traded in that made money, its trading days in the test years, and the test
    years without their best days. Days without PnL are days it did not trade: before its data
    starts, or between the rare events it trades on."""
    by_year: dict[int, list[float]] = {}
    for day, pnl in days.items():
        if pnl != 0:
            by_year.setdefault(day.year, []).append(pnl)
    traded = [sum(v) for v in by_year.values() if len(v) >= ACTIVE_YEAR_DAYS]
    found = Consistency(years=len(traded), positive_years=sum(1 for v in traded if v > 0))
    if test_start is not None and days:
        test = sorted((p for d, p in days.items() if d >= test_start and p != 0), reverse=True)
        found.test_days = len(test)
        if len(test) >= SPIKE_DAYS * 4:
            found.test_without_spike = sum(test[SPIKE_DAYS:])
    return found


@dataclass(slots=True)
class RecentForm:
    #: Sharpe in each of :data:`RECENT_YEARS`; ``None`` for a year it has no days in.
    years: dict[int, float | None] = field(default_factory=dict)
    #: Sharpe over :data:`RECENT_YEARS` together, and over its whole history.
    recent: float | None = None
    full: float | None = None

    @property
    def weak_years(self) -> list[int]:
        return [y for y, s in self.years.items() if s is not None and s < MIN_YEAR_SHARPE]

    @property
    def fading(self) -> bool:
        return (
            self.recent is not None
            and self.full is not None
            and self.full > 0
            and self.recent < KEEP_RECENT * self.full
        )


def recent_form(series: dict[date, tuple[float, float]]) -> RecentForm:
    """Its Sharpe in each recent year and over them together, against its whole history."""
    if not series:
        return RecentForm()
    days = sorted(series)
    pnl = np.array([series[d][0] for d in days])
    traded = np.array([series[d][1] for d in days])
    by_year = {y: s.sharpe for y, _, s in metrics.yearly(days, pnl, traded, None)}
    late = np.array(days, dtype="datetime64[D]") >= np.datetime64(RECENT_START, "D")
    recent = metrics.stats(pnl[late], traded[late], trim=False) if late.any() else None
    full = metrics.stats(pnl, traded)
    return RecentForm(
        years={y: by_year.get(y) for y in RECENT_YEARS},
        recent=recent.sharpe if recent else None,
        full=full.sharpe if full else None,
    )


# -- verdicts ------------------------------------------------------------------


@dataclass(slots=True)
class NeighbourResult:
    kind: str
    name: str
    #: queued | done | failed
    state: str
    sharpe: float | None = None
    test_sharpe: float | None = None
    holds: bool | None = None


@dataclass(slots=True)
class Verdict:
    status: Status
    held: int = 0
    measured: int = 0
    neighbours: list[NeighbourResult] = field(default_factory=list)


def holds(
    origin_sharpe: float | None,
    sharpe: float | None,
    test_sharpe: float | None,
    keep: float = HOLD,
) -> bool | None:
    """Whether a neighbour keeps ``keep`` of the finalist's signal; ``None`` when it cannot be
    told."""
    if origin_sharpe is None or sharpe is None or origin_sharpe <= 0:
        return None
    return sharpe >= keep * origin_sharpe and (test_sharpe is None or test_sharpe > 0)


def verdict(results: list[NeighbourResult]) -> Verdict:
    """Robust once enough neighbours are back and most hold; fragile once they cannot."""
    if not results:
        return Verdict("untested")
    measured = [r for r in results if r.holds is not None]
    held = sum(1 for r in measured if r.holds)
    waiting = sum(1 for r in results if r.state == "queued")
    found = Verdict("running", held, len(measured), results)
    # Decided early when the outcome can no longer change.
    most = len(measured) + waiting
    if most >= MIN_MEASURED and held + waiting < PLATEAU * most:
        found.status = "fragile"
    elif waiting:
        found.status = "running"
    elif len(measured) < MIN_MEASURED:
        found.status = "inconclusive"
    else:
        found.status = "robust" if held >= PLATEAU * len(measured) else "fragile"
    return found


# -- the agent -----------------------------------------------------------------


class Guard:
    def __init__(self, state: Any, bus: AgentBus) -> None:
        self.state = state
        self.bus = bus

    async def add_task(self, alpha_ids: list[str], cores: int) -> AddedTask | None:
        """Write every finalist's neighbours as one task in Tasks, not started."""
        async with self.state.db.session() as session:
            tested = set(
                (
                    await session.scalars(
                        select(GuardCheck.origin_alpha_id).where(
                            GuardCheck.origin_alpha_id.in_(alpha_ids)
                        )
                    )
                ).all()
            )
        wanted = [a for a in alpha_ids if a not in tested]
        if not wanted:
            return None
        rows = await self.state.alphas.by_ids(wanted)
        schema = await self.state.metadata.cached_settings_schema() or {}
        planned = [
            (alpha_id, kind, request)
            for alpha_id in wanted
            if alpha_id in rows
            for kind, request in neighbours_of(rows[alpha_id], schema)
        ]
        if not planned:
            return None
        markets = Counter((r.settings.region, r.settings.delay) for _, _, r in planned)
        (region, delay), _ = markets.most_common(1)[0]
        finalists = len({a for a, _, _ in planned})
        added = await add_study(
            self.state,
            now=utcnow(),
            sampler=GUARD_SAMPLER,
            params=GuardParams(
                region=region,
                delay=delay,
                cores=cores,
                alphas=finalists,
                checks=sorted({k for _, k, _ in planned}),
            ),
            simulations=len(planned),
            batch_size=(cores + 1) * MULTI_SIMULATION_SIZE,
            template_source=f"{len(planned)} neighbours of {finalists} finalists",
            template_name="Overfitting Guard",
            seeds=settings_sampler.seed_trials([r for _, _, r in planned]),
        )
        async with self.state.db.session() as session:
            session.add_all(
                GuardCheck(
                    origin_alpha_id=alpha_id,
                    kind=kind,
                    study_id=added.id,
                    trial_number=number,
                )
                for number, (alpha_id, kind, _) in enumerate(planned)
            )
        self.bus.say("critic", f"Wrote {len(planned)} neighbour checks; run them from Tasks.")
        return added

    async def verdicts(self, origins: dict[str, dict[str, Any]]) -> dict[str, Verdict]:
        """Each finalist's neighbour verdict, read from its neighbours' Alphas."""
        async with self.state.db.session() as session:
            checks = list(
                await session.scalars(
                    select(GuardCheck).where(GuardCheck.origin_alpha_id.in_(list(origins)))
                )
            )
            trials = {
                (t.study_id, t.number): t
                for t in await session.scalars(
                    select(Trial).where(Trial.study_id.in_({c.study_id for c in checks}))
                )
            }
        alpha_ids = [t.alpha_id for t in trials.values() if t.alpha_id]
        rows = await self.state.alphas.by_ids(alpha_ids)
        grouped: dict[str, list[NeighbourResult]] = {}
        for check in checks:
            trial = trials.get((check.study_id, check.trial_number))
            kind = str(check.kind)
            result = NeighbourResult(kind, NAMES.get(kind) or kind, "queued")
            row = rows.get(trial.alpha_id) if trial and trial.alpha_id else None
            if row is not None and row.get("sharpe") is not None:
                result.state = "done"
                result.sharpe, result.test_sharpe = row.get("sharpe"), row.get("test_sharpe")
                origin = origins[check.origin_alpha_id]
                result.holds = holds(
                    origin.get("sharpe"), result.sharpe, result.test_sharpe, KEEPS.get(kind, HOLD)
                )
            elif (
                trial is None
                or trial.state == TrialState.FAIL
                or (trial.finished_at is not None and not trial.alpha_id)
            ):
                # Cancelled, refused by BRAIN, or ended without an Alpha: it tells nothing.
                result.state = "failed"
            grouped.setdefault(check.origin_alpha_id, []).append(result)
        return {a: verdict(grouped.get(a, [])) for a in origins}

    async def siblings(self, rows: dict[str, dict[str, Any]]) -> dict[str, int]:
        """How many simulations in the vault share each finalist's idea and region."""
        wanted = {
            (str(r["region"]), skeleton(r["expression"]))
            for r in rows.values()
            if r.get("expression")
        }
        if not wanted:
            return {}
        regions = sorted({region for region, _ in wanted})
        placeholders = ", ".join("?" for _ in regions)
        found = await self.state.alphas.catalog.query(
            f"""
            SELECT region, s, count(*) AS n FROM (
                SELECT region, regexp_replace(
                    regexp_replace(expression, '\\s+', '', 'g'), '[0-9]+(\\.[0-9]+)?', '#', 'g'
                ) AS s
                FROM alpha WHERE expression IS NOT NULL AND region IN ({placeholders})
            ) GROUP BY region, s
            """,  # noqa: S608
            regions,
        )
        counts = {(str(r["region"]), str(r["s"])): int(r["n"]) for r in found}
        return {
            alpha_id: counts.get((str(r["region"]), skeleton(r["expression"])), 1)
            for alpha_id, r in rows.items()
            if r.get("expression")
        }
