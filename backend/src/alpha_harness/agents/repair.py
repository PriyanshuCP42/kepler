"""The Repair Engineer: turns near misses into passing Alphas, and learns which fix works.

A near miss failed by a little: a BRAIN check just past its limit, test years that kept
somewhat less of the Sharpe than they should, a margin a little thin, or a PnL too close to
an Alpha already submitted. Finding a new passing idea costs hundreds of simulations; a near
miss often needs one change. So for each near miss this reads what failed, picks the fixes
most likely to work, and writes them as one task, run from Tasks like any lab's.

**Fixes** (:data:`MOVES`) change one thing each: a setting (decay, universe, neutralization,
truncation) or the expression, wrapped in a smoother, a ``rank`` or a ``winsorize``. For an
Alpha that repeats your pool it borrows the Correlation Breaker's re-shapes, which residualize
or orthogonalize the signal against the exposures pools share.

**Learning**: every (failure, fix) pair keeps a count of tries and of repairs the Critic then
passed. Fixes are chosen by Thompson sampling over those counts, so the ones that work get
chosen more and an untried one still gets its turn. A repair is settled once the Critic has
judged its result: passing is a win, anything else a try.
"""

import asyncio
import contextlib
import json
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select

from ..brain.platform import IGNORED_CHECKS, INSTRUMENT_TYPE, MULTI_SIMULATION_SIZE, TEST_PERIOD
from ..brain.schemas import FULL_MODE, SimulationRequest, SimulationSettings
from ..brain.settings_schema import valid_values, validate_settings
from ..db.models import CriticVerdict, RepairAttempt, RepairStat, Trial, TrialState, utcnow
from ..labs.fastexpr import ParseError
from ..labs.launch import AddedTask, add_study
from ..labs.params import REPAIR_SAMPLER, RepairParams
from ..tools import correlation_breaker, settings_sampler
from ..vault.yields import REFUSING, checks_of
from .themes import takeable

if TYPE_CHECKING:
    from .bus import AgentBus

log = structlog.get_logger(__name__)

#: Fixes tried per near miss in one task.
PER_ALPHA = 3
#: How often finished repairs are settled.
SETTLE_SECONDS = 120.0
NEUTRALIZATIONS = ("MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY")
#: Failures every fix is judged against; BRAIN's check names plus the Critic's own gates.
DECAY, COST, POOL = "DECAY", "COST", "POOL"

Settings = dict[str, Any]
#: (expression, settings, options) -> (expression, settings), or None when it does not apply.
Transform = Callable[[str, Settings, "Options"], tuple[str, Settings] | None]


@dataclass(frozen=True, slots=True)
class Options:
    """What the market allows, read from BRAIN's settings schema."""

    universes: tuple[str, ...]
    neutralizations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Move:
    id: str
    name: str
    fixes: frozenset[str]
    apply: Transform


def _wrap(tail: str) -> Transform:
    """An expression fix: the original bound to ``alpha``, then ``tail`` around it."""

    def apply(expression: str, settings: Settings, _: Options) -> tuple[str, Settings] | None:
        try:
            compressed = correlation_breaker.compress(expression)
        except ParseError:
            return None
        return compressed.program(tail), settings

    return apply


def setting_move(change: Callable[[Settings, Options], Settings | None]) -> Transform:
    def apply(expression: str, settings: Settings, options: Options) -> tuple[str, Settings] | None:
        changed = change(dict(settings), options)
        return None if changed is None or changed == settings else (expression, changed)

    return apply


def _size(universe: str) -> int | None:
    digits = re.findall(r"\d+", universe)
    return int(digits[0]) if digits and universe.upper().startswith("TOP") else None


def universe_step(step: int) -> Callable[[Settings, Options], Settings | None]:
    def change(settings: Settings, options: Options) -> Settings | None:
        current = _size(str(settings.get("universe") or ""))
        sized = sorted((s, u) for u in options.universes if (s := _size(u)) is not None)
        if current is None or not sized:
            return None
        pick = (
            next((u for s, u in sized if s > current), None)
            if step > 0
            else next((u for s, u in reversed(sized) if s < current), None)
        )
        return None if pick is None else settings | {"universe": pick}

    return change


def neutralization_step(step: int) -> Callable[[Settings, Options], Settings | None]:
    def change(settings: Settings, options: Options) -> Settings | None:
        ladder = [n for n in NEUTRALIZATIONS if n in options.neutralizations]
        current = str(settings.get("neutralization") or "")
        if current not in ladder:
            return None
        index = ladder.index(current) + step
        return settings | {"neutralization": ladder[index]} if 0 <= index < len(ladder) else None

    return change


def _neutralize_to(value: str) -> Callable[[Settings, Options], Settings | None]:
    def change(settings: Settings, options: Options) -> Settings | None:
        if value not in options.neutralizations:
            return None
        return settings | {"neutralization": value}

    return change


def _decay(factor: float) -> Callable[[Settings, Options], Settings | None]:
    def change(settings: Settings, _: Options) -> Settings | None:
        current = int(settings.get("decay") or 0)
        new = min(20, max(4, current * 2)) if factor > 1 else current // 2
        return settings | {"decay": new}

    return change


def _truncation(settings: Settings, _: Options) -> Settings | None:
    current = float(settings.get("truncation") or 0.08)
    return settings | {"truncation": round(max(0.01, min(current, 0.08) / 2), 3)}


def _breaker(recipe_id: str) -> Transform:
    recipe = next(r for r in correlation_breaker.RECIPES if r.id == recipe_id)

    def apply(expression: str, settings: Settings, _: Options) -> tuple[str, Settings] | None:
        try:
            compressed = correlation_breaker.compress(expression)
        except ParseError:
            return None
        return compressed.program(recipe.tail, recipe.setup), settings

    return apply


_TURNOVER = frozenset({"HIGH_TURNOVER", COST})
_ROBUST = frozenset({"LOW_SUB_UNIVERSE_SHARPE", "LOW_ROBUST_UNIVERSE_SHARPE"})
_WEAK = frozenset({"LOW_SHARPE", "LOW_FITNESS", "LOW_2Y_SHARPE", "IS_LADDER_SHARPE", DECAY})
_CROWDED = frozenset({POOL, "SELF_CORRELATION"})

MOVES: tuple[Move, ...] = (
    Move(
        "slower_decay",
        "Raise the decay setting",
        _TURNOVER | {"LOW_FITNESS"},
        setting_move(_decay(2)),
    ),
    Move("smooth_mean", "Average over a week", _TURNOVER, _wrap("ts_mean(alpha, 5)")),
    Move(
        "smooth_linear",
        "Decay linearly over two weeks",
        _TURNOVER | {DECAY},
        _wrap("ts_decay_linear(alpha, 10)"),
    ),
    Move(
        "faster_decay",
        "Lower the decay setting",
        frozenset({"LOW_TURNOVER"}),
        setting_move(_decay(0.5)),
    ),
    Move(
        "liquid_tilt",
        "Lean towards liquid stocks",
        _ROBUST,
        _wrap("rank(alpha) * rank(adv20)"),
    ),
    Move(
        "rank", "Rank across the universe", _ROBUST | {"CONCENTRATED_WEIGHT"}, _wrap("rank(alpha)")
    ),
    Move(
        "winsorize",
        "Clip outliers",
        frozenset({"CONCENTRATED_WEIGHT", DECAY, "LOW_2Y_SHARPE"}),
        _wrap("winsorize(alpha, std=4)"),
    ),
    Move(
        "tighter_truncation",
        "Halve truncation",
        frozenset({"CONCENTRATED_WEIGHT"}),
        setting_move(_truncation),
    ),
    Move("wider_universe", "Use the next larger universe", _ROBUST, setting_move(universe_step(1))),
    Move(
        "narrower_universe",
        "Use the next smaller universe",
        frozenset({"LOW_FITNESS", "LOW_SHARPE"}),
        setting_move(universe_step(-1)),
    ),
    Move(
        "finer_neutralization",
        "Neutralize within finer groups",
        _WEAK | _CROWDED,
        setting_move(neutralization_step(1)),
    ),
    Move(
        "coarser_neutralization",
        "Neutralize within broader groups",
        frozenset({"LOW_SHARPE", "LOW_FITNESS"}),
        setting_move(neutralization_step(-1)),
    ),
    Move(
        "statistical_neutralization",
        "Neutralize statistically",
        _CROWDED,
        setting_move(_neutralize_to("STATISTICAL")),
    ),
    Move(
        "crowding_neutralization",
        "Neutralize against crowded factors",
        _CROWDED,
        setting_move(_neutralize_to("CROWDING")),
    ),
    Move(
        "momentum_neutralization",
        "Neutralize reversion and momentum",
        _CROWDED | {DECAY},
        setting_move(_neutralize_to("REVERSION_AND_MOMENTUM")),
    ),
    Move("resid_returns", "Keep what returns do not explain", _CROWDED, _breaker("resid_returns")),
    Move("neut_momentum", "Orthogonalize against momentum", _CROWDED, _breaker("neut_momentum")),
    Move("neut_size", "Orthogonalize against size", _CROWDED, _breaker("neut_size")),
    Move(
        "fast_innovation",
        "Trade the change, not the level",
        _CROWDED | {DECAY},
        _breaker("fast_innovation"),
    ),
)
BY_ID = {move.id: move for move in MOVES}


def failures(row: dict[str, Any], gates: dict[str, Any]) -> list[str]:
    """What a near miss failed: BRAIN's refusing checks, then the Critic's own gates."""
    found = [
        str(c.get("name")).upper()
        for c in checks_of(row.get("checks"))
        if str(c.get("name", "")).upper() not in IGNORED_CHECKS
        and str(c.get("result", "")).upper() in REFUSING
    ]
    for gate, name in (("decay", DECAY), ("cost", COST), ("pool", POOL)):
        if (gates.get(gate) or {}).get("ok") is False:
            found.append(name)
    return list(dict.fromkeys(found))


def choose(
    failed: list[str],
    stats: dict[tuple[str, str], tuple[int, int]],
    count: int = PER_ALPHA,
    rng: random.Random | None = None,
) -> list[tuple[Move, str]]:
    """Up to ``count`` fixes for these failures, each with the failure it was chosen for.

    Thompson sampling: each candidate draws from Beta(1 + wins, 1 + misses) for the failure it
    fixes best, and the highest draws win.
    """
    rng = rng or random.Random()
    drawn: list[tuple[float, Move, str]] = []
    for move in MOVES:
        best: tuple[float, str] | None = None
        for failure in failed:
            if failure not in move.fixes:
                continue
            tries, wins = stats.get((failure, move.id), (0, 0))
            sample = rng.betavariate(1 + wins, 1 + tries - wins)
            if best is None or sample > best[0]:
                best = (sample, failure)
        if best is not None:
            drawn.append((best[0], move, best[1]))
    drawn.sort(key=lambda d: d[0], reverse=True)
    return [(move, failure) for _, move, failure in drawn[:count]]


def base_settings(row: dict[str, Any]) -> Settings:
    """A stored Alpha's settings as a simulation asks for them, in Full mode with a test period."""
    return {
        # BRAIN's schema offers universes and neutralizations per instrument type.
        "instrumentType": INSTRUMENT_TYPE,
        "region": row["region"],
        "universe": row["universe"],
        "delay": int(row["delay"]),
        "neutralization": row["neutralization"],
        "decay": int(row["decay"] or 0),
        "truncation": float(row["truncation"] or 0.08),
        "testPeriod": TEST_PERIOD,
        "simulationMode": FULL_MODE,
    }


def options_for(schema: dict[str, Any], base: Settings) -> Options:
    return Options(
        tuple(str(u) for u in valid_values(schema, "universe", base)),
        tuple(str(n) for n in valid_values(schema, "neutralization", base)),
    )


def checked_request(
    expression: str, settings: Settings, schema: dict[str, Any]
) -> SimulationRequest | None:
    """The simulation, or ``None`` when BRAIN's settings schema would refuse it."""
    try:
        request = SimulationRequest(
            settings=SimulationSettings.model_validate(settings), regular=expression
        )
    except ValueError:
        return None
    if schema and validate_settings(schema, request.to_wire()["settings"]):
        return None
    return request


@dataclass(frozen=True, slots=True)
class Repair:
    origin: str
    failure: str
    move: str
    request: SimulationRequest


class RepairEngineer:
    def __init__(self, state: Any, bus: AgentBus) -> None:
        self.state = state
        self.bus = bus
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="repair-engineer")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        while True:
            try:
                await self.settle()
                await self._report()
            except Exception:
                log.warning("repair.settle_failed", exc_info=True)
            await asyncio.sleep(SETTLE_SECONDS)

    # -- planning ------------------------------------------------------------

    async def waiting(self) -> list[str]:
        """Near misses not tried yet, taken in turn from each market, best first within each.

        Best-first alone filled a whole round from one market, which spreads nothing across
        the pyramids; taking turns gives every market with near misses its share.
        """
        async with self.state.db.session() as session:
            tried = select(RepairAttempt.origin_alpha_id)
            rows = (
                (
                    await session.execute(
                        select(CriticVerdict.alpha_id, CriticVerdict.region, CriticVerdict.delay)
                        .where(
                            CriticVerdict.verdict == "near_miss",
                            CriticVerdict.alpha_id.not_in(tried),
                        )
                        .order_by(CriticVerdict.score.desc())
                    )
                )
                .tuples()
                .all()
            )
        return interleave([(alpha_id, (region, delay)) for alpha_id, region, delay in rows])

    async def plan(
        self, limit: int, per_alpha: int = PER_ALPHA, ids: list[str] | None = None
    ) -> list[Repair]:
        """Fixes for the best ``limit`` untried near misses, checked against BRAIN's settings.

        ``ids`` names the alphas to fix instead: the Submission Desk's candidates that BRAIN's
        submission check refused, whatever the Critic made of them.
        """
        ids = (ids if ids is not None else await self.waiting())[:limit]
        if not ids:
            return []
        async with self.state.db.session() as session:
            verdicts = {
                v.alpha_id: v
                for v in await session.scalars(
                    select(CriticVerdict).where(CriticVerdict.alpha_id.in_(ids))
                )
            }
            stats = {
                (s.failure, s.move): (s.tries, s.wins)
                for s in await session.scalars(select(RepairStat))
            }
        placeholders = ", ".join("?" for _ in ids)
        rows = await self.state.alphas.catalog.query(
            "SELECT alpha_id, expression, region, delay, universe, neutralization, decay, "  # noqa: S608
            f"truncation, checks FROM alpha WHERE alpha_id IN ({placeholders})",
            ids,
        )
        schema = await self.state.metadata.cached_settings_schema() or {}
        repairs: list[Repair] = []
        seen: set[str] = set()
        for row in sorted(rows, key=lambda r: ids.index(str(r["alpha_id"]))):
            expression = row.get("expression")
            verdict = verdicts.get(str(row["alpha_id"]))
            if not expression or verdict is None:
                continue
            failed = failures(row, verdict.gates or {})
            if not failed:
                continue
            base = base_settings(row)
            options = options_for(schema, base)
            for move, failure in choose(failed, stats, per_alpha * 2):
                built = move.apply(expression, base, options)
                request = None if built is None else checked_request(*built, schema)
                if request is None:
                    continue
                # Two fixes can write the same simulation; it is sent, and counted, once.
                key = json.dumps(request.to_wire(), sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                repairs.append(Repair(str(row["alpha_id"]), failure, move.id, request))
                if sum(1 for r in repairs if r.origin == str(row["alpha_id"])) >= per_alpha:
                    break
        return repairs

    async def add_task(
        self, limit: int, per_alpha: int, cores: int, ids: list[str] | None = None
    ) -> AddedTask | None:
        """Write the repairs as one task in Tasks, not started. ``None`` when there are none."""
        repairs = await self.plan(limit, per_alpha, ids)
        if not repairs:
            return None
        regions = Counter((r.request.settings.region, r.request.settings.delay) for r in repairs)
        (region, delay), _ = regions.most_common(1)[0]
        moves = sorted({r.move for r in repairs})
        origins = len({r.origin for r in repairs})
        added = await add_study(
            self.state,
            now=utcnow(),
            sampler=REPAIR_SAMPLER,
            params=RepairParams(
                region=region,
                delay=delay,
                cores=cores,
                alphas=origins,
                moves=moves,
            ),
            simulations=len(repairs),
            batch_size=(cores + 1) * MULTI_SIMULATION_SIZE,
            template_source=f"{len(repairs)} repairs of {origins} near misses",
            template_name="Repair Engineer",
            seeds=settings_sampler.seed_trials([r.request for r in repairs]),
        )
        async with self.state.db.session() as session:
            session.add_all(
                RepairAttempt(
                    origin_alpha_id=r.origin,
                    failure=r.failure,
                    move=r.move,
                    study_id=added.id,
                    trial_number=number,
                )
                for number, r in enumerate(repairs)
            )
        self.bus.say("repair", f"Wrote {len(repairs)} repairs; run the task from Tasks.")
        return added

    # -- learning ------------------------------------------------------------

    async def settle(self) -> int:
        """Judge finished repairs by the Critic's verdict and update what has been learned."""
        settled = await self._correct()
        async with self.state.db.session() as session:
            open_attempts = (
                await session.scalars(select(RepairAttempt).where(RepairAttempt.outcome.is_(None)))
            ).all()
            if not open_attempts:
                return 0
            trials = {
                (t.study_id, t.number): t
                for t in await session.scalars(
                    select(Trial).where(Trial.study_id.in_({a.study_id for a in open_attempts}))
                )
            }
            alpha_ids = [t.alpha_id for t in trials.values() if t.alpha_id]
            judged = list(
                await session.scalars(
                    select(CriticVerdict).where(CriticVerdict.alpha_id.in_(alpha_ids))
                )
            )
            verdicts = {v.alpha_id: v.verdict for v in judged}
            placed = {v.alpha_id for v in judged if v.family_checked_at is not None}
            blocked = await self._untakeable([a for a, v in verdicts.items() if v == "pass"])
            for attempt in open_attempts:
                trial = trials.get((attempt.study_id, attempt.trial_number))
                outcome = _outcome(trial, verdicts, placed, blocked)
                if outcome is None:
                    continue
                attempt.outcome, attempt.settled_at = outcome, utcnow()
                attempt.alpha_id = trial.alpha_id if trial else None
                settled += 1
                if outcome == "cancelled":
                    continue
                stat = await session.get(RepairStat, (attempt.failure, attempt.move))
                if stat is None:
                    stat = RepairStat(failure=attempt.failure, move=attempt.move, tries=0, wins=0)
                    session.add(stat)
                stat.tries += 1
                stat.wins += outcome == "fixed"
                if outcome == "fixed":
                    self.bus.count("repair", "fixed")
                    self.bus.handoff("repair", "critic", "fixed", 1, "repaired and passing")
        return settled

    async def _untakeable(self, alpha_ids: list[str]) -> set[str]:
        """Those of these alphas BRAIN cannot take: under the regular limits, and not a Power
        Pool alpha in a market this month's themes cover (see :mod:`.themes`)."""
        if not alpha_ids:
            return set()
        themes = await self.state.themes.current()
        rows = await self.state.alphas.by_ids(alpha_ids)
        return {a for a, row in rows.items() if not takeable(row, themes)}

    async def _correct(self) -> int:
        """Turn back fixes whose Alpha no longer passes, e.g. found to repeat your pool, or
        that BRAIN cannot take: one still under the regular limits outside a theme market."""
        async with self.state.db.session() as session:
            fixed = list(
                await session.scalars(select(RepairAttempt).where(RepairAttempt.outcome == "fixed"))
            )
            ids = [a.alpha_id for a in fixed if a.alpha_id]
            passing = set(
                await session.scalars(
                    select(CriticVerdict.alpha_id).where(
                        CriticVerdict.alpha_id.in_(ids), CriticVerdict.verdict == "pass"
                    )
                )
            )
            blocked = await self._untakeable(list(passing))
            wrong = [a for a in fixed if a.alpha_id not in passing or a.alpha_id in blocked]
            for attempt in wrong:
                attempt.outcome = "not_fixed"
                stat = await session.get(RepairStat, (attempt.failure, attempt.move))
                if stat is not None and stat.wins > 0:
                    stat.wins -= 1
            return len(wrong)

    async def summary(self) -> dict[str, Any]:
        async with self.state.db.session() as session:
            attempts = (await session.scalars(select(RepairAttempt))).all()
            stats = (await session.scalars(select(RepairStat))).all()
        outcomes = Counter(a.outcome or "pending" for a in attempts)
        return {
            "waiting": len(await self.waiting()),
            "pending": outcomes.get("pending", 0),
            "fixed": outcomes.get("fixed", 0),
            "notFixed": outcomes.get("not_fixed", 0),
            "errors": outcomes.get("error", 0),
            "moves": [
                {
                    "failure": s.failure,
                    "move": s.move,
                    "name": BY_ID[s.move].name if s.move in BY_ID else s.move,
                    "tries": s.tries,
                    "wins": s.wins,
                }
                for s in sorted(stats, key=lambda s: (-s.wins, -s.tries))
            ],
        }

    async def _report(self) -> None:
        summary = await self.summary()
        self.bus.set_stat("repair", "near misses waiting", summary["waiting"])
        self.bus.set_stat("repair", "repairs running", summary["pending"])
        self.bus.set_stat("repair", "fixed", summary["fixed"])
        if not summary["pending"]:
            self.bus.say(
                "repair", f"{summary['waiting']} near misses waiting. Plan repairs in Autopilot."
            )


def interleave(rows: list[tuple[str, Any]]) -> list[str]:
    """Ids taken in turn from each group, keeping each group's own order."""
    groups: dict[Any, list[str]] = {}
    for alpha_id, group in rows:
        groups.setdefault(group, []).append(alpha_id)
    out: list[str] = []
    queues = list(groups.values())
    while queues:
        out.extend(queue.pop(0) for queue in queues)
        queues = [queue for queue in queues if queue]
    return out


def _outcome(
    trial: Trial | None,
    verdicts: dict[str, str],
    placed: frozenset[str] | set[str] = frozenset(),
    blocked: frozenset[str] | set[str] = frozenset(),
) -> str | None:
    """A finished repair's outcome, or ``None`` while it is still out or not judged.

    A pass is only a fix once the Family Clerk has compared it with your pool (``placed``):
    one that repeats the pool passes the Critic first and is turned back afterwards.
    """
    if trial is None:
        return "cancelled"
    if trial.state in (TrialState.QUEUED, TrialState.RUNNING) or (
        trial.state == TrialState.PRUNED and trial.alpha_id is None and trial.finished_at is None
    ):
        return None
    if trial.alpha_id:
        verdict = verdicts.get(trial.alpha_id)
        if verdict is None:
            return None
        if verdict == "pass":
            if trial.alpha_id in blocked:
                return "not_fixed"  # passes the Critic, yet BRAIN cannot take it
            return "fixed" if trial.alpha_id in placed else None
        return "not_fixed"
    if trial.state == TrialState.FAIL:
        return "error"
    return "cancelled" if trial.state == TrialState.PRUNED else None
