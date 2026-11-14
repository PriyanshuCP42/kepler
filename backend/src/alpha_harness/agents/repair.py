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
#: Near misses read per one repaired, so crowded ones can be passed over and the round still filled.
CROWD_LOOKAHEAD = 4
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
    #: Pseudo-counts (tries, wins) added to what Kepler has measured, from outside evidence.
    prior: tuple[int, int] = (0, 0)


#: WorldQuant's own measurements of smoothing operators, from the BRAIN webinar slide
#: "Operators and Universes that can potentially improve performance" (November 2026): the
#: average change each brought to turnover, margin, Fitness and IS Sharpe.
#:
#: ==========================  ========  ======  =======  =========
#: operator                    turnover  margin  Fitness  IS Sharpe
#: ==========================  ========  ======  =======  =========
#: ts_target_tvr_decay         -17%      +43%    +12%     -5%
#: jump_decay                  -13%      +27%    +10%     -3%
#: group_backfill              -10%      +21%    +9%      -1%
#: hump                        -14%      +25%    +10%     -1%
#: ts_mean                     -15%      +27%    +4%      -4%
#: ts_decay_linear             -9%       +19%    +5%      -2%
#: ts_target_tvr_hump          -16%      +36%    +14%     -3%
#: ts_target_tvr_delta_limit   -8%       +15%    +11%     +2%
#: ==========================  ========  ======  =======  =========
#:
#: Heavy smoothing can cost a little IS Sharpe, so the strength (d, hump, target_tvr) is kept
#: moderate. jump_decay and ts_target_tvr_delta_limit open at higher Genius levels and are
#: left out until the account has them.
SLIDE_FITNESS = {
    "ts_target_tvr_decay": 0.12,
    "group_backfill": 0.09,
    "hump": 0.10,
    "ts_mean": 0.04,
    "ts_decay_linear": 0.05,
    "ts_target_tvr_hump": 0.14,
}
#: Pseudo-tries a slide-backed fix starts with: enough to try it first, few enough that what
#: Kepler measures on this account outweighs it within a couple of dozen repairs.
SLIDE_TRIES = 6


def slide_prior(operator: str) -> tuple[int, int]:
    """Pseudo (tries, wins) for a fix, its win rate 0.5 plus 2.5x the slide's Fitness gain."""
    rate = 0.5 + 2.5 * SLIDE_FITNESS[operator]
    return SLIDE_TRIES, int(SLIDE_TRIES * min(rate, 0.9) + 0.5)


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


#: BRAIN's own Production Correlation check. Ignored elsewhere as a failure of the idea, it
#: is repairable when only a little over the 0.7 ceiling: re-neutralizing a crowded alpha
#: inside sector and statistical clusters has brought one back under it.
PROD = "PROD_CORRELATION"
#: Measured Production Correlation up to which a repair is still worth its simulations. Of
#: the alphas measured over it, none came back under 0.7 whatever was done to them.
REPAIRABLE_PROD = 0.8
INVESTABILITY = "LOW_INVESTABILITY_CONSTRAINED_SHARPE"
#: Sharpe inside one part of a regional market: Japan within ASI, a continent within GLB.
_SUB_MARKET = frozenset(
    {"LOW_ASI_JPN_SHARPE", "LOW_GLB_AMER_SHARPE", "LOW_GLB_EMEA_SHARPE", "LOW_GLB_APAC_SHARPE"}
)

_TURNOVER = frozenset({"HIGH_TURNOVER", COST})
_ROBUST = frozenset(
    {
        "LOW_SUB_UNIVERSE_SHARPE",
        "LOW_ROBUST_UNIVERSE_SHARPE",
        "LOW_ROBUST_UNIVERSE_SHARPE.WITH_RATIO",
        "LOW_ROBUST_UNIVERSE_RETURNS",
    }
)
_WEAK = frozenset({"LOW_SHARPE", "LOW_FITNESS", "LOW_2Y_SHARPE", "IS_LADDER_SHARPE", DECAY})
_CROWDED = frozenset({POOL, "SELF_CORRELATION", PROD})

#: Each market's statistical cluster grouping, the second half of the sector-and-cluster
#: neutralization that repaired a crowded alpha. A market missing here skips that move.
CLUSTERS = {
    "USA": "pv13_h_min2_3000_sector",
    "GLB": "sta1_allc20",
    "EUR": "sta1_allc20",
    "ASI": "sta1_allxjp_513_c20",
    "IND": "sta1_allxjp_513_c20",
    "KOR": "sta1_allxjp_513_c20",
    "HKG": "sta1_allxjp_513_c20",
    "GBR": "sta1_top1200c20",
    "DEU": "sta1_top1200c20",
}


def _cluster_sector(expression: str, settings: Settings, _: Options) -> tuple[str, Settings] | None:
    field = CLUSTERS.get(str(settings.get("region") or ""))
    if field is None:
        return None
    return _wrap(f"group_neutralize(alpha, densify(group_cartesian_product(sector, {field})))")(
        expression, settings, _
    )


def _max_trade(settings: Settings, _: Options) -> Settings | None:
    return None if settings.get("maxTrade") == "ON" else settings | {"maxTrade": "ON"}


MOVES: tuple[Move, ...] = (
    Move(
        "slower_decay",
        "Raise the decay setting",
        _TURNOVER | {"LOW_FITNESS"},
        setting_move(_decay(2)),
    ),
    Move(
        "smooth_mean",
        "Average over a week",
        _TURNOVER,
        _wrap("ts_mean(alpha, 5)"),
        slide_prior("ts_mean"),
    ),
    Move(
        "smooth_linear",
        "Decay linearly over two weeks",
        _TURNOVER | {DECAY},
        _wrap("ts_decay_linear(alpha, 10)"),
        slide_prior("ts_decay_linear"),
    ),
    Move(
        "target_tvr_hump",
        "Hold turnover near 20% by skipping small trades",
        _TURNOVER | {"LOW_FITNESS"},
        _wrap("ts_target_tvr_hump(alpha, lambda_min=0, lambda_max=1, target_tvr=0.2)"),
        slide_prior("ts_target_tvr_hump"),
    ),
    Move(
        "target_tvr_decay",
        "Hold turnover near 20% by decaying",
        _TURNOVER | {"LOW_FITNESS"},
        _wrap("ts_target_tvr_decay(alpha, lambda_min=0, lambda_max=1, target_tvr=0.2)"),
        slide_prior("ts_target_tvr_decay"),
    ),
    Move(
        "hump",
        "Hold positions through small changes",
        _TURNOVER | {"LOW_FITNESS"},
        # hump's limit is hump x the book's gross (group_sum(abs(alpha), market)), so 0.01
        # all but freezes a 3,000-stock alpha. Measured: 0.0005 kept Sharpe (1.40 to 1.39),
        # 0.002 cost a fifth of it (1.40 to 1.10).
        _wrap("hump(alpha, hump=0.001)"),
        slide_prior("hump"),
    ),
    Move(
        "group_backfill",
        "Fill gaps from industry peers",
        _TURNOVER | {"LOW_FITNESS"},
        _wrap("group_backfill(alpha, industry, 21)"),
        slide_prior("group_backfill"),
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
    # The moves below each turned a near miss into a submission on this account in November
    # 2026; their priors say so, and what Kepler measures from here takes over.
    Move(
        "cluster_sector",
        "Neutralize inside sector and statistical clusters",
        _CROWDED,
        _cluster_sector,
        (4, 2),
    ),
    Move(
        "size_tilt",
        "Lean towards larger companies",
        _ROBUST,
        _wrap("alpha * (0.5 + rank(cap))"),
        (4, 2),
    ),
    Move(
        "liquid_volume_tilt",
        "Lean gently towards liquid stocks",
        _ROBUST,
        _wrap("alpha * (0.5 + rank(adv20))"),
    ),
    Move(
        "volatility_tilt",
        "Lean towards the more volatile stocks of each industry",
        frozenset({COST, "LOW_FITNESS", "LOW_RETURNS"}),
        _wrap("alpha * group_rank(ts_std_dev(returns, 20), industry)"),
        (4, 2),
    ),
    Move(
        "convex",
        "Weight the extremes",
        _CROWDED | {COST, "LOW_FITNESS"},
        _wrap("signed_power(rank(alpha) - 0.5, 2)"),
        (4, 2),
    ),
    Move("max_trade", "Turn Max Trade on", frozenset({INVESTABILITY}), setting_move(_max_trade)),
    Move(
        "country_books",
        "Balance the book inside every country",
        _SUB_MARKET,
        _wrap("group_neutralize(alpha, country)"),
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
    for check in checks_of(row.get("checks")):
        value = check.get("value")
        if (
            str(check.get("name", "")).upper() == PROD
            and str(check.get("result", "")).upper() == "FAIL"
            and isinstance(value, int | float)
            and value <= REPAIRABLE_PROD
        ):
            found.append(PROD)
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
    fixes best, and the highest draws win. A fix's :attr:`Move.prior` is added to its counts.
    """
    rng = rng or random.Random()
    drawn: list[tuple[float, Move, str]] = []
    for move in MOVES:
        best: tuple[float, str] | None = None
        for failure in failed:
            if failure not in move.fixes:
                continue
            tries, wins = stats.get((failure, move.id), (0, 0))
            tries, wins = tries + move.prior[0], wins + move.prior[1]
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
    } | ({"maxTrade": "ON"} if str(row.get("max_trade") or "").upper() == "ON" else {})


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
        ids_given = ids
        # Read past ``limit``: crowded near misses are passed over below.
        ids = (ids if ids is not None else await self.waiting())[: limit * CROWD_LOOKAHEAD]
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
            f"truncation, max_trade, checks FROM alpha WHERE alpha_id IN ({placeholders})",
            ids,
        )
        schema = await self.state.metadata.cached_settings_schema() or {}
        # A fix keeps the idea, so it keeps the idea's Production Correlation: repairs of an
        # alpha others already have came back over 0.7 nearly every time.
        crowding = getattr(self.state, "crowding", None)
        if crowding is not None:
            known = await crowding.measured([str(r["alpha_id"]) for r in rows])
            # A little over the ceiling is repairable; only a measured crowd past
            # REPAIRABLE_PROD, or a guess from its datasets, rules an alpha out.
            rows = [
                r
                for r in rows
                if (
                    (measured := known.get(str(r["alpha_id"]))) is not None
                    and measured <= REPAIRABLE_PROD
                )
                or (measured is None and not await crowding.too_crowded(r.get("expression"), None))
            ]
        rows = sorted(rows, key=lambda r: ids.index(str(r["alpha_id"])))[:limit]
        repairs: list[Repair] = []
        seen: set[str] = set()
        for row in sorted(rows, key=lambda r: ids.index(str(r["alpha_id"]))):
            expression = row.get("expression")
            verdict = verdicts.get(str(row["alpha_id"]))
            # Named alphas are BRAIN's own refusals and need no Critic verdict to be read.
            if not expression or (verdict is None and ids_given is None):
                continue
            failed = failures(row, (verdict.gates if verdict else None) or {})
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
