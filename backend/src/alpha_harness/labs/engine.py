"""The template engine: writes searches fitted to each dataset, and learns which ones work.

For a region it asks the Dataset Doctor which datasets hold usable, uncrowded fields of each
shape, leaves out datasets you have already submitted there (a new idea must be new data),
then for each promising dataset chooses a recipe for its shape and writes a Template Lab task
for it, not started. The Daily Autopilot starts it.

**Learning.** Every engine task is named ``Engine · recipe · shape · region · dataset``. Its
finished trials are judged by the Submit Gate's local tests: Sharpe and Fitness limits, margin,
returns per drawdown, and test years keeping their Sharpe. Each (recipe, shape, region) keeps a
count of trials and passes. Recipes are chosen by Thompson sampling over those counts, so one
that works is chosen more and an untried one still gets its turn. One that has spent
:data:`RETIRE_TRIES` trials in a region without a pass is retired there.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select

from ..agents import gate
from ..catalog.queries import FieldFilter, Tuple4
from ..db.models import Study, Trial
from . import doctor, recipes

if TYPE_CHECKING:
    from .launch import AddedTask

log = structlog.get_logger(__name__)

PREFIX = "Engine"
SEPARATOR = " · "
#: Trials after which a recipe with no pass in a region is retired there.
RETIRE_TRIES = 150
#: Datasets the Doctor reads per plan, most valuable first.
DATASETS_READ = 40
#: Simulations each engine task spends, and the cores it holds.
TASK_SIMULATIONS = 60
TASK_CORES = 2


def task_name(recipe: str, shape: str, region: str, dataset: str) -> str:
    return SEPARATOR.join((PREFIX, recipe, shape, region, dataset))


def parse_name(name: str | None) -> tuple[str, str, str, str] | None:
    """``(recipe, shape, region, dataset)`` of an engine task's name, or ``None``."""
    parts = (name or "").split(SEPARATOR)
    if len(parts) != 5 or parts[0] != PREFIX:
        return None
    return parts[1], parts[2], parts[3], parts[4]


@dataclass(slots=True)
class Record:
    """How one recipe has done for one shape in one region."""

    tries: int = 0
    passes: int = 0

    @property
    def retired(self) -> bool:
        return self.tries >= RETIRE_TRIES and self.passes == 0

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.passes + 1, self.tries - self.passes + 1)


def passes(stats: dict[str, Any], limits: dict[str, Any] | None = None) -> bool:
    """Whether a finished trial passes the Submit Gate's local tests: everything but what only
    BRAIN's check and the pool effect can say."""
    limits = limits or {}

    def num(key: str) -> float | None:
        value = stats.get(key)
        return float(value) if isinstance(value, int | float) else None

    facts = gate.Facts(
        alpha_id="",
        sharpe=num("sharpe"),
        fitness=num("fitness"),
        sharpe_limit=float(limits.get("LOW_SHARPE") or 1.58),
        fitness_limit=float(limits.get("LOW_FITNESS") or 1.0),
        margin=num("margin"),
        returns=num("returns"),
        drawdown=num("drawdown"),
        train_sharpe=num("train_sharpe"),
        test_sharpe=num("test_sharpe"),
    )
    return gate.judge(facts).verdict in (gate.Verdict.READY, gate.Verdict.CHECK)


def tally(
    rows: list[tuple[str | None, dict[str, Any] | None]],
) -> dict[tuple[str, str, str], Record]:
    """Records from ``(task name, trial result)`` pairs; rows of other tasks are ignored."""
    out: dict[tuple[str, str, str], Record] = {}
    for name, result in rows:
        parsed = parse_name(name)
        if parsed is None or not result:
            continue
        recipe, shape, region, _ = parsed
        stats = result.get("stats") or {}
        if stats.get("sharpe") is None:
            continue
        record = out.setdefault((recipe, shape, region), Record())
        record.tries += 1
        record.passes += passes(stats, result.get("limits"))
    return out


async def records(state: Any) -> dict[tuple[str, str, str], Record]:
    async with state.db.session() as session:
        rows = (
            await session.execute(
                select(Study.template_name, Trial.result)
                .join(Trial, Trial.study_id == Study.id)
                .where(Study.template_name.like(f"{PREFIX}{SEPARATOR}%"))
                .where(Trial.result.is_not(None))
            )
        ).all()
    return tally([(name, result) for name, result in rows])


async def busy_datasets(state: Any, region: str) -> set[str]:
    """Datasets an engine task not yet finished is already searching in ``region``."""
    async with state.db.session() as session:
        names = await session.scalars(
            select(Study.template_name).where(
                Study.template_name.like(f"{PREFIX}{SEPARATOR}%"),
                Study.status.in_(("IDLE", "QUEUED", "RUNNING", "PAUSED")),
            )
        )
        return {p[3] for n in names if (p := parse_name(n)) and p[2] == region}


@dataclass(slots=True)
class Choice:
    candidate: doctor.Candidate
    recipe: recipes.Recipe
    score: float


def choose(
    found: list[doctor.Candidate],
    learned: dict[tuple[str, str, str], Record],
    region: str,
    count: int,
    rng: random.Random,
) -> list[Choice]:
    """Up to ``count`` searches, one per dataset: each candidate's best sampled recipe, scored
    by the dataset's priority times the recipe's sampled pass rate."""
    if not found:
        return []
    top = max(c.priority for c in found) or 1.0
    picked: list[Choice] = []
    for c in found:
        options = [
            r
            for r in recipes.for_shape(c.shape)
            if not learned.get((r.key, str(c.shape), region), Record()).retired
        ]
        if not options:
            continue
        draws = [
            (learned.get((r.key, str(c.shape), region), Record()).sample(rng), r) for r in options
        ]
        rate, recipe = max(draws, key=lambda d: d[0])
        picked.append(Choice(candidate=c, recipe=recipe, score=c.priority / top * rate))
    picked.sort(key=lambda ch: -ch.score)
    seen: set[str] = set()
    out: list[Choice] = []
    for ch in picked:
        if ch.candidate.dataset_id in seen:
            continue
        seen.add(ch.candidate.dataset_id)
        out.append(ch)
        if len(out) >= count:
            break
    return out


async def diagnose(
    state: Any,
    region: str,
    avoid: frozenset[str] = frozenset(),
    category: str | None = None,
) -> list[doctor.Candidate]:
    """The Dataset Doctor's candidates in ``region``'s main universe, of one dataset
    ``category`` only when given."""
    market = recipes.market_of(region)
    if market is None:
        return []
    scope = Tuple4(region=region, delay=1, universe=market.universes[0])
    datasets = await state.queries.datasets(scope)
    datasets = [d for d in datasets if d.get("dataset_id") not in doctor.EXCLUDED | avoid]
    if category is not None:
        datasets = [d for d in datasets if str(d.get("category_id") or "").lower() == category]
    if not datasets:
        return []
    datasets.sort(
        key=lambda d: -float(d.get("value_score") or 0) * float(d.get("pyramid_multiplier") or 1)
    )
    chosen = datasets[:DATASETS_READ]
    page = await state.queries.fields(
        scope,
        FieldFilter(
            dataset_ids=[str(d["dataset_id"]) for d in chosen],
            coverage_min=doctor.MIN_COVERAGE,
            alpha_count_max=doctor.CROWDED_FIELD,
            limit=5000,
        ),
        counted=False,
    )
    return doctor.candidates(chosen, list(page.get("results") or []), avoid=avoid)


def task_for(choice: Choice, region: str) -> Any:
    """The Template Lab task a choice becomes."""
    from ..api.template_lab import FieldsVariable, TemplateTask, ValuesVariable

    market = recipes.market_of(region)
    if market is None:
        raise ValueError(f"No market rules for {region}")
    r, c = choice.recipe, choice.candidate
    variables: dict[str, Any] = {
        "f": FieldsVariable(
            kind="fields",
            dataset_ids=[c.dataset_id],
            filter=FieldFilter(field_ids=c.field_ids, field_types=["MATRIX", "VECTOR"]),
            vector_operators=["vec_avg"],
        )
    }
    variables |= {k: ValuesVariable(kind="values", values=v) for k, v in r.values.items()}
    return TemplateTask(
        region=region,
        delay=1,
        template=r.template,
        variables=variables,
        universes=list(market.universes),
        neutralizations=list(market.neutralizations),
        investability=list(market.investability),
        decay=r.decay,
        truncation=market.truncation,
        cores=TASK_CORES,
        simulations=TASK_SIMULATIONS,
        template_name=task_name(r.key, str(c.shape), region, c.dataset_id),
    )


async def submitted_datasets(state: Any, region: str) -> frozenset[str]:
    """Datasets your submitted Alphas in ``region`` already read: a new search wants new data."""
    from ..agents.picks import datasets_of

    members = [m for m in await state.alphas.submitted_members() if m.get("region") == region]
    found = await datasets_of(
        state, {str(m["alpha_id"]): str(m.get("expression") or "") for m in members}
    )
    return frozenset(d for ds in found.values() for d in ds)


async def plan(state: Any, region: str, count: int = 3, seed: int | None = None) -> list[AddedTask]:
    """Write up to ``count`` engine tasks for ``region``, not started."""
    from ..api.template_lab import add_task

    avoid = await submitted_datasets(state, region) | await busy_datasets(state, region)
    found = await diagnose(state, region, frozenset(avoid))
    chosen = choose(found, await records(state), region, count, random.Random(seed))
    added: list[AddedTask] = []
    for ch in chosen:
        try:
            added.append(await add_task(task_for(ch, region), state))
        except Exception:
            log.warning("engine.task_refused", dataset=ch.candidate.dataset_id, exc_info=True)
    log.info("engine.planned", region=region, added=len(added))
    return added


async def plan_niche(
    state: Any, region: str, delay: int, category: str, simulations: int, cores: int
) -> AddedTask | None:
    """One engine task for a pyramid the Strategist chose, or ``None`` when the engine has no
    recipe there (a delay or region it does not search, or no usable dataset of the category).
    The Strategist then falls back to a Search Lab task."""
    from ..api.template_lab import add_task

    if delay != 1 or recipes.market_of(region) is None:
        return None
    avoid = await submitted_datasets(state, region) | await busy_datasets(state, region)
    found = await diagnose(state, region, frozenset(avoid), category.lower())
    chosen = choose(found, await records(state), region, 1, random.Random())
    if not chosen:
        return None
    task = task_for(chosen[0], region).model_copy(
        update={"simulations": simulations, "cores": cores}
    )
    return await add_task(task, state)


__all__ = [
    "PREFIX",
    "RETIRE_TRIES",
    "Choice",
    "Record",
    "busy_datasets",
    "choose",
    "diagnose",
    "parse_name",
    "passes",
    "plan",
    "plan_niche",
    "records",
    "tally",
    "task_for",
    "task_name",
]
