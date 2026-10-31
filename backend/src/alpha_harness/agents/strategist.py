"""The Strategist: which pyramids get the next simulations.

A **niche** is a pyramid: one dataset category in one region and delay, the cell BRAIN pays a
multiplier for and counts as complete once three Alphas are submitted in it in a quarter.
Effort piles up where a search last worked (a third of this vault is one universe) while the
cells the Genius bars need go untried. So the Strategist keeps a **niche map** of every
pyramid, built from the vault: simulations spent there, Alphas the Critic passed, and new
families found. Then it scores each one:

- **Value**: the pyramid's multiplier, times how much it is still needed this quarter (a cell
  one Alpha from complete is worth more than one already complete).
- **Yield**: new families per 100 simulations, learned per niche. It is *sampled* from a
  Gamma posterior rather than averaged (Thompson sampling), so a niche with little history
  still gets chosen sometimes, and one with a long dry record rarely does.
- **Stall**: a niche that spent :data:`STALL_SIMULATIONS` in the last :data:`RECENT_DAYS` days
  without one Critic pass is cooled, so effort cannot pile into it again.

A plan takes the top niches, keeps one place for a niche barely tried, splits the simulations
between them by score, and writes one Search Lab task for each, not started.
"""

import asyncio
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from fastapi import HTTPException
from sqlalchemy import select

from ..brain.filters import platform_midnight
from ..brain.platform import PYRAMID_MIN_ALPHAS
from ..brain.settings_schema import valid_values
from ..catalog.pyramids import pyramid_grid
from ..db.models import CriticVerdict
from ..labs.fastexpr import ParseError, data_fields, parse

log = structlog.get_logger(__name__)

RECENT_DAYS = 14
STALL_SIMULATIONS = 300
STALL_FACTOR = 0.2
#: Below this many simulations a niche counts as barely tried, and one plan place is kept
#: for such a niche.
UNTRIED = 50
#: Prior for families per 100 simulations: about one family per 300 simulations, the vault's
#: own long-run rate, held loosely.
PRIOR_SHAPE = 0.5
PRIOR_RATE = 1.5
#: The history is rebuilt at most this often; the grid refreshes on its own schedule.
HISTORY_SECONDS = 1800
#: A niche in a market this month's Power Pool theme covers is boosted by this much per unit
#: of the theme's multiplier: GLB at 1 by half again, ASI at 2 twice over.
THEME_BOOST = 0.5
#: Your own hypotheses, as a boost to a niche's score: (region, category) -> factor.
HYPOTHESES: dict[tuple[str, str], float] = {("USA", "risk"): 1.3, ("GLB", "risk"): 1.3}
NEUTRALIZATIONS = ("SUBINDUSTRY", "INDUSTRY", "SECTOR", "MARKET")
#: Datasets a Strategist task searches per niche.
DATASETS_PER_TASK = 3
#: Offered to every task, kept only where BRAIN's operators include them.
VECTOR_OPERATORS = ("vec_avg",)


def need(submitted: int) -> float:
    """How much a pyramid still needs Alphas this quarter, from those already submitted."""
    if submitted >= PYRAMID_MIN_ALPHAS:
        return 0.35
    return 1.0 + 0.15 * submitted


@dataclass(slots=True)
class Niche:
    region: str
    delay: int
    category: str
    category_name: str
    multiplier: float | None = None
    submitted: int = 0
    synced: bool = False
    simulations: int = 0
    passed: int = 0
    families: int = 0
    recent_simulations: int = 0
    recent_passed: int = 0
    status: str = "ready"
    value: float = 0.0
    expected: float = 0.0
    score: float = 0.0
    share: float = 0.0
    boost: float = 1.0
    reasons: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.region, self.delay, self.category)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["categoryName"] = out.pop("category_name")
        out["recentSimulations"] = out.pop("recent_simulations")
        out["recentPassed"] = out.pop("recent_passed")
        return out


def assess(
    niches: list[Niche],
    rng: random.Random | None = None,
    themes: dict[tuple[str, int], tuple[str, float]] | None = None,
) -> list[Niche]:
    """Score every niche and return them best first. Pure apart from the random draws.

    ``themes`` maps a market (region, delay) to this month's Power Pool theme there and its
    multiplier: only in such a market can an alpha under the regular limits be submitted, so
    its niches are boosted by :data:`THEME_BOOST` per unit of multiplier.
    """
    rng = rng or random.Random()
    for niche in niches:
        niche.reasons = []
        niche.value = (niche.multiplier or 1.0) * need(niche.submitted)
        niche.boost = HYPOTHESES.get((niche.region, niche.category), 1.0)
        theme = (themes or {}).get((niche.region, niche.delay))
        if theme is not None:
            niche.boost *= 1 + THEME_BOOST * theme[1]
            niche.reasons.append(f"This month's {theme[0]} Power Pool theme covers it.")
        shape = PRIOR_SHAPE + niche.families
        rate = PRIOR_RATE + niche.simulations / 100
        niche.expected = shape / rate
        if not niche.synced:
            niche.status, niche.score = "not_downloaded", 0.0
            niche.reasons.append("Its data is not downloaded; sync it in Data Explorer.")
            continue
        sampled = rng.gammavariate(shape, 1 / rate)
        niche.score = sampled * niche.value * niche.boost
        if niche.recent_simulations >= STALL_SIMULATIONS and niche.recent_passed == 0:
            niche.status = "stalled"
            niche.score *= STALL_FACTOR
            niche.reasons.append(
                f"Stalled: {niche.recent_simulations:,} simulations in {RECENT_DAYS} days "
                "and no Critic pass."
            )
        elif niche.simulations < UNTRIED:
            niche.status = "untried"
            niche.reasons.append("Barely tried yet.")
        else:
            niche.status = "ready"
        if niche.submitted < PYRAMID_MIN_ALPHAS:
            left = PYRAMID_MIN_ALPHAS - niche.submitted
            niche.reasons.append(
                f"{left} more submitted Alpha{'s' if left > 1 else ''} completes it."
            )
        if HYPOTHESES.get((niche.region, niche.category), 1.0) > 1:
            niche.reasons.append("One of your hypotheses.")
    return sorted(niches, key=lambda n: n.score, reverse=True)


def _crowded(table: dict[str, Any], dataset: Any) -> bool:
    found = table.get(str(dataset))
    return bool(found and found.crowded)


def _clean(table: dict[str, Any], dataset: Any) -> bool:
    found = table.get(str(dataset))
    return bool(found and found.clean)


def choose(ranked: list[Niche], count: int) -> list[Niche]:
    """The plan: the best ``count - 1`` niches plus the most valuable barely-tried one, with
    each one's share of the simulations by score."""
    eligible = [n for n in ranked if n.score > 0]
    picked = eligible[: max(1, count - 1)]
    untried = [n for n in eligible if n.status == "untried" and n not in picked]
    if untried and count > 1:
        picked.append(max(untried, key=lambda n: n.value * n.boost))
    else:
        picked = eligible[:count]
    total = sum(n.score for n in picked) or 1.0
    for niche in picked:
        niche.share = niche.score / total
    return picked


class Strategist:
    def __init__(self, state: Any, bus: Any) -> None:
        self.state = state
        self.bus = bus
        self._history: dict[tuple[str, int, str], dict[str, int]] = {}
        self._history_at = float("-inf")
        self._lock = asyncio.Lock()

    async def niche_map(self) -> list[Niche]:
        """Every pyramid with its value and record, unscored."""
        try:
            grid = await pyramid_grid(self.state.endpoints, self.state.catalog)
        except Exception:
            log.warning("strategist.grid_failed", exc_info=True)
            grid = {"cells": [], "categories": []}
        names = {c["id"]: c["name"] for c in grid.get("categories", [])}
        history = await self._load_history()
        niches: dict[tuple[str, int, str], Niche] = {}
        for cell in grid.get("cells", []):
            key = (cell["region"], int(cell["delay"]), str(cell["categoryId"]).lower())
            niches[key] = Niche(
                region=key[0],
                delay=key[1],
                category=key[2],
                category_name=names.get(cell["categoryId"], cell["categoryId"]),
                multiplier=cell.get("multiplier"),
                submitted=int(cell.get("alphaCount") or 0),
                synced=bool(cell.get("synced")),
            )
        if not niches:
            # Without BRAIN's grid (signed out), fall back to what the vault and catalog know.
            for key in history:
                niches[key] = Niche(key[0], key[1], key[2], key[2], synced=True)
        for key, niche in niches.items():
            record = history.get(key, {})
            niche.simulations = record.get("simulations", 0)
            niche.passed = record.get("passed", 0)
            niche.families = record.get("families", 0)
            niche.recent_simulations = record.get("recent_simulations", 0)
            niche.recent_passed = record.get("recent_passed", 0)
        return list(niches.values())

    async def assessed(self, seed: int | None = None) -> list[Niche]:
        """The niches scored and ranked. The draws are seeded by the platform's day, so the
        plan shown is the plan written; it moves when the day or the history does."""
        if seed is None:
            seed = platform_midnight().date().toordinal()
        themes = {
            (t.region, t.delay): (t.name, t.multiplier)
            for t in await self._themes()
            if t.region is not None and t.delay is not None
        }
        return assess(await self.niche_map(), random.Random(seed), themes)

    async def _load_history(self) -> dict[tuple[str, int, str], dict[str, int]]:
        """Per niche: simulations, Critic passes and new families, all time and recent."""
        async with self._lock:
            if time.monotonic() - self._history_at < HISTORY_SECONDS:
                return self._history
            fields = {
                str(r["field_id"]): str(r["c"]).lower()
                for r in await self.state.catalog.query(
                    "SELECT field_id, any_value(category_id) AS c FROM data_field "
                    "WHERE category_id IS NOT NULL GROUP BY field_id"
                )
            }
            alphas = await self.state.catalog.query(
                "SELECT alpha_id, expression, region, delay, date_created FROM alpha "
                "WHERE expression IS NOT NULL AND region IS NOT NULL"
            )
            async with self.state.db.session() as session:
                verdicts = {
                    alpha_id: (verdict, family)
                    for alpha_id, verdict, family in (
                        await session.execute(
                            select(
                                CriticVerdict.alpha_id,
                                CriticVerdict.verdict,
                                CriticVerdict.family_id,
                            )
                        )
                    ).tuples()
                }
            self._history = await asyncio.to_thread(_tally, alphas, fields, verdicts)
            self._history_at = time.monotonic()
            return self._history

    async def plan(self, count: int = 5, simulations: int = 1500) -> list[dict[str, Any]]:
        ranked = await self.assessed()
        picked = choose(ranked, count)
        return [
            n.to_dict() | {"simulations": max(100, round(simulations * n.share))} for n in picked
        ]

    async def add_tasks(
        self, count: int = 5, simulations: int = 1500, cores: int = 4
    ) -> dict[str, Any]:
        """One Search Lab task per planned niche, not started. Returns what was added and
        why any niche was skipped."""
        # Imported here: the API layer builds on the agents.
        from ..api.search_lab import add_task
        from ..labs.launch import SearchRequest

        planned = await self.plan(count, simulations)
        schema = await self.state.metadata.cached_settings_schema() or {}
        added: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for niche in planned:
            label = f"{niche['region']} D{niche['delay']} {niche['categoryName']}"
            market = await self.market(niche, schema)
            if market is None:
                skipped.append({"niche": label, "reason": "No dataset of it is downloaded."})
                continue
            body = SearchRequest(
                region=niche["region"],
                delay=niche["delay"],
                universe=market["universe"],
                dataset_ids=market["datasets"],
                # Some datasets hold only vector fields (short interest, job postings); without
                # a vector operator the Search Lab finds nothing to search in them.
                vector_operators=list(VECTOR_OPERATORS),
                neutralizations=market["neutralizations"],
                cores=cores,
                simulations=niche["simulations"],
            )
            try:
                task = await add_task(body, self.state)
            except HTTPException as exc:
                # The Search Lab refuses a market it cannot search; say why and plan the rest.
                detail = exc.detail
                message = detail.get("message") if isinstance(detail, dict) else str(detail)
                skipped.append({"niche": label, "reason": str(message)[:200]})
                continue
            added.append(
                {"id": task.id, "niche": label, "simulations": niche["simulations"]}
                | {"datasets": market["datasets"], "universe": market["universe"]}
            )
        if added:
            self.bus.handoff("strategist", "writers", "plan", len(added), "tasks planned")
            self.bus.say(
                "strategist", f"Planned {len(added)} Search Lab tasks; run them from Tasks."
            )
        return {"added": added, "skipped": skipped}

    async def _themes(self) -> list[Any]:
        source = getattr(self.state, "themes", None)
        return await source.current() if source is not None else []

    async def market(self, niche: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any] | None:
        """The universe, the least crowded datasets and the neutralizations for one niche.

        In a market whose Power Pool theme names a universe (ASI D1 MINVOL10M), that universe,
        so what it finds can be submitted as pure Power Pool."""
        rows = await self.state.catalog.query(
            "SELECT dataset_id, universe, coalesce(alpha_count, 0) AS alphas, "
            "coalesce(coverage, 0) AS coverage, coalesce(field_count, 0) AS fields "
            "FROM data_set WHERE region = ? AND delay = ? AND lower(category_id) = ?",
            [niche["region"], niche["delay"], niche["category"]],
        )
        if not rows:
            return None
        universes = Counter(str(r["universe"]) for r in rows)
        universe = max(universes, key=lambda u: (universes[u], _size(u)))
        for theme in await self._themes():
            if (
                theme.region == niche["region"]
                and theme.delay == niche["delay"]
                and theme.universe in universes
            ):
                universe = theme.universe
        in_universe = [r for r in rows if str(r["universe"]) == universe and r["fields"] > 0]
        # Datasets whose measured alphas came back over the Production Correlation target are
        # left out while others remain; those that came back low go first, then the least
        # used: an alpha on data few others use correlates less with everyone's.
        source = getattr(self.state, "crowding", None)
        table = await source.table() if source is not None else {}
        clear = [r for r in in_universe if not _crowded(table, r["dataset_id"])]
        in_universe = clear or in_universe
        in_universe.sort(
            key=lambda r: (not _clean(table, r["dataset_id"]), r["alphas"], -r["coverage"])
        )
        datasets = list(dict.fromkeys(str(r["dataset_id"]) for r in in_universe))[
            :DATASETS_PER_TASK
        ]
        settings = {
            "instrumentType": "EQUITY",
            "region": niche["region"],
            "delay": niche["delay"],
            "universe": universe,
        }
        allowed = (
            set(map(str, valid_values(schema, "neutralization", settings))) if schema else set()
        )
        neutralizations = [n for n in NEUTRALIZATIONS if not allowed or n in allowed][:2]
        if not datasets or not neutralizations:
            return None
        return {"universe": universe, "datasets": datasets, "neutralizations": neutralizations}


def _size(universe: str) -> int:
    digits = "".join(ch for ch in universe if ch.isdigit())
    return int(digits) if digits else 0


def _tally(
    alphas: list[dict[str, Any]],
    fields: dict[str, str],
    verdicts: dict[str, tuple[str, str | None]],
) -> dict[tuple[str, int, str], dict[str, int]]:
    """Count each Alpha against the category of the first data field it reads."""
    since = datetime.now(UTC) - timedelta(days=RECENT_DAYS)
    out: dict[tuple[str, int, str], dict[str, int]] = defaultdict(Counter)
    for row in alphas:
        try:
            names = data_fields(parse(row["expression"]))
        except ParseError:
            continue
        category = next((fields[n] for n in names if n in fields), None)
        if category is None:
            continue
        key = (str(row["region"]), int(row["delay"] or 0), category)
        verdict, family = verdicts.get(str(row["alpha_id"]), (None, None))
        passed = verdict == "pass"
        created = row.get("date_created")
        if isinstance(created, datetime) and created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        recent = isinstance(created, datetime) and created >= since
        record = out[key]
        record["simulations"] += 1
        record["passed"] += passed
        record["families"] += passed and family == row["alpha_id"]
        record["recent_simulations"] += recent
        record["recent_passed"] += recent and passed
    return {key: dict(value) for key, value in out.items()}
