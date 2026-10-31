"""The Hypothesis Writer: new ideas, each with a reason, aimed where the Strategist says pays.

The labs mostly recombine: templates over fields, mutations of what worked. New families need
new ideas. So for each pyramid the Strategist plans, this picks a dataset few others use
(their Alphas correlate least with production), shows an LLM its fields and BRAIN's operators,
and asks for hypotheses: a reason rooted in how investors behave, then three short expressions
that test it.

Every expression is checked before it can cost a simulation: it must parse, use only real
operators and fields, stay short (Power Pool's limits), and be new. An expression that shares
most of its fields with an alpha you hold (:mod:`.overlap`) or repeats one already in the vault
with other numbers is dropped. You see every hypothesis and can switch any off; the rest are
simulated as one task, and each hypothesis keeps a record of how its expressions did.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import Counter
from typing import TYPE_CHECKING, Any, Literal

import structlog
from sqlalchemy import select

from ..brain.platform import MULTI_SIMULATION_SIZE
from ..db.models import CriticVerdict, Hypothesis, Trial, utcnow
from ..labs.fastexpr import operator_table
from ..labs.launch import AddedTask, add_study
from ..labs.params import HYPOTHESIS_SAMPLER, HypothesisParams
from ..labs.power_pool import Context, Rejected, check, context_for, field_line, operators_text
from ..llm.keys import LLMError
from ..llm.prompts import HYPOTHESIS_WRITER
from ..llm.text import FENCE
from ..tools import settings_sampler
from .guard import skeleton
from .overlap import Overlap
from .repair import base_settings, checked_request

if TYPE_CHECKING:
    from .bus import AgentBus

log = structlog.get_logger(__name__)

#: Pyramids written for per round, and hypotheses asked for in each.
NICHES = 3
PER_NICHE = 4
#: Field lines shown to the model, most complete first.
FIELDS_SHOWN = 120
#: An expression sharing this much of its fields with an alpha you hold is not new.
MAX_OVERLAP = 0.75
#: Decay each expression is simulated at, once per neutralization the pyramid allows.
DECAY = 4

Status = Literal["idle", "writing", "failed"]

SCHEMA = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "rationale": {"type": "string"},
                    "expressions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "rationale", "expressions"],
            },
        }
    },
    "required": ["hypotheses"],
}


class NoAnswerError(RuntimeError):
    """Every model failed for one pyramid."""


def parse_hypotheses(text: str) -> list[dict[str, Any]]:
    """The hypotheses in an answer, or none; a fenced block is tried after the whole text."""
    for candidate in (text, *(m.group(1) for m in FENCE.finditer(text))):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError, TypeError:
            continue
        items = payload.get("hypotheses") if isinstance(payload, dict) else payload
        if isinstance(items, list):
            return [
                h
                for h in items
                if isinstance(h, dict) and h.get("title") and isinstance(h.get("expressions"), list)
            ]
    return []


def user_prompt(ctx: Context, operators: list[dict[str, Any]], market: str, count: int) -> str:
    fields = "\n".join(field_line(f) for f in ctx.fields[:FIELDS_SHOWN])
    return "\n\n".join(
        [
            f"MARKET\n{market}",
            "OPERATORS\n" + operators_text(operators),
            f"DATASET\n{ctx.id} · {ctx.name} · {ctx.category}\n{ctx.description}",
            f"FIELDS OF {ctx.id} · most complete first\n{fields}",
            "PRICE AND VOLUME FIELDS · count as data fields\n"
            + "\n".join(field_line(f) for f in ctx.basics),
            "GROUPING FIELDS · not counted\n" + ", ".join(ctx.groups),
            f"Write {count} hypotheses about {ctx.id}, each with 3 expressions.",
        ]
    )


def screen(
    expressions: list[Any],
    ctx: Context,
    table: dict[str, Any],
    region: str,
    known: set[str],
    overlap: Overlap,
) -> tuple[list[str], list[str]]:
    """The expressions worth a simulation, and why each other one was dropped."""
    kept: list[str] = []
    dropped: list[str] = []
    for text in expressions:
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            expression, _ = check(text, ctx, table)
        except Rejected as exc:
            dropped.append(f"{text[:60]}: {exc}")
            continue
        shape = skeleton(expression)
        if shape in known:
            dropped.append(f"{expression[:60]}: already simulated with other numbers.")
            continue
        if (shared := overlap.worst(expression, region)) >= MAX_OVERLAP:
            dropped.append(f"{expression[:60]}: shares {shared:.0%} of its fields with your pool.")
            continue
        known.add(shape)
        kept.append(expression)
    return kept, dropped


class HypothesisWriter:
    def __init__(self, state: Any, bus: AgentBus) -> None:
        self.state = state
        self.bus = bus
        self.status: Status = "idle"
        self.note: str | None = None
        self._task: asyncio.Task[int] | None = None

    # -- writing -------------------------------------------------------------

    def write_in_background(self, niches: int = NICHES, per_niche: int = PER_NICHE) -> bool:
        if self._task is not None and not self._task.done():
            return False
        self.status, self.note = "writing", "Choosing pyramids and datasets."
        self._task = asyncio.create_task(self.write(niches, per_niche), name="hypothesis-writer")
        return True

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def write(self, niches: int = NICHES, per_niche: int = PER_NICHE) -> int:
        """Ask the LLM for hypotheses on the Strategist's top pyramids; returns how many kept."""
        try:
            kept = await self._write(niches, per_niche)
        except Exception as exc:
            log.warning("hypothesis.write_failed", exc_info=True)
            self.status, self.note = "failed", str(exc)[:300]
            self.bus.block("hypothesis", self.note)
            return 0
        self.status = "idle"
        self.bus.block("hypothesis", None)
        return kept

    async def _write(self, niches: int, per_niche: int) -> int:
        models = await self.state.llm.runnable()
        if not models:
            raise RuntimeError("No model can answer. Set one up under LLM Integration › Models.")
        operators = await self.state.metadata.cached_operators() or []
        if not operators:
            raise RuntimeError("Your BRAIN operators could not be read. Sign in, then try again.")
        table = operator_table(operators)
        schema = await self.state.metadata.cached_settings_schema() or {}
        overlap = Overlap(self.state.db, self.state.alphas)
        await overlap.refresh()
        async with self.state.db.session() as session:
            used = Counter((await session.scalars(select(Hypothesis.dataset_id))).all())

        kept = 0
        notes: list[str] = []
        for niche in await self.state.strategist.plan(niches):
            label = f"{niche['region']} D{niche['delay']} {niche['categoryName']}"
            market = await self.state.strategist.market(niche, schema)
            if market is None:
                notes.append(f"{label}: no dataset downloaded.")
                continue
            # The least crowded dataset this writer has used least.
            dataset = min(market["datasets"], key=lambda d: used[d])
            used[dataset] += 1
            ctx = await context_for(
                self.state.catalog, niche["region"], niche["delay"], [market["universe"]], dataset
            )
            if ctx is None or not ctx.fields:
                notes.append(f"{label}: {dataset} has no fields downloaded.")
                continue
            self.note = f"Asking for hypotheses on {dataset} ({label})."
            self.bus.say("hypothesis", self.note)
            where = f"{niche['region']} · Delay {niche['delay']} · {market['universe']}"
            try:
                answer, model = await self._ask(
                    models, user_prompt(ctx, operators, where, per_niche)
                )
            except NoAnswerError as exc:
                # One pyramid the model could not answer for leaves the others to write.
                notes.append(f"{label}: {exc}")
                continue
            known = await self._known(niche["region"])
            rows: list[Hypothesis] = []
            dropped = 0
            for item in parse_hypotheses(answer)[:per_niche]:
                good, bad = screen(item["expressions"], ctx, table, niche["region"], known, overlap)
                dropped += len(bad)
                if not good:
                    continue
                rows.append(
                    Hypothesis(
                        region=niche["region"],
                        delay=int(niche["delay"]),
                        universe=market["universe"],
                        category=str(niche["categoryName"]),
                        dataset_id=dataset,
                        title=str(item["title"])[:200],
                        rationale=str(item.get("rationale") or "")[:2000],
                        expressions=good,
                        neutralizations=market["neutralizations"],
                        model=model,
                    )
                )
            async with self.state.db.session() as session:
                session.add_all(rows)
            kept += len(rows)
            notes.append(
                f"{label}: {len(rows)} hypotheses on {dataset}, {dropped} expressions dropped."
            )
        self.note = " ".join(notes) or "The Strategist planned no pyramids."
        if kept:
            self.bus.handoff("strategist", "hypothesis", "plan", kept, "hypotheses written")
            self.bus.say("hypothesis", f"Wrote {kept} hypotheses; choose which to simulate.")
        return kept

    async def _ask(self, models: list[dict[str, Any]], user: str) -> tuple[str, str]:
        """The first model that answers, most requests left first."""
        failure: Exception | None = None
        for model in models:
            try:
                answer = await self.state.llm.generate(
                    system=HYPOTHESIS_WRITER,
                    user=user,
                    model_ref=str(model["ref"]),
                    response_schema=SCHEMA,
                    temperature=0.9,
                )
            except LLMError as exc:
                failure = exc
                continue
            return answer.text, str(model["id"])
        raise NoAnswerError(f"No model answered: {failure}")

    async def _known(self, region: str) -> set[str]:
        """Every expression already in the vault for ``region``, numbers blanked."""
        rows = await self.state.alphas.catalog.query(
            "SELECT DISTINCT expression FROM alpha WHERE region = ? AND expression IS NOT NULL",
            [region],
        )
        return {skeleton(str(r["expression"])) for r in rows}

    # -- testing -------------------------------------------------------------

    async def add_task(self, cores: int) -> tuple[AddedTask, int] | None:
        """Every switched-on hypothesis not yet simulated, as one task in Tasks, not started."""
        async with self.state.db.session() as session:
            waiting = list(
                await session.scalars(
                    select(Hypothesis)
                    .where(Hypothesis.enabled.is_(True), Hypothesis.study_id.is_(None))
                    .order_by(Hypothesis.id)
                )
            )
        if not waiting:
            return None
        schema = await self.state.metadata.cached_settings_schema() or {}
        planned: list[tuple[int, Any]] = []
        for hypothesis in waiting:
            row = {
                "region": hypothesis.region,
                "universe": hypothesis.universe,
                "delay": hypothesis.delay,
                "decay": DECAY,
                "truncation": None,
            }
            for neutralization in hypothesis.neutralizations or ["SUBINDUSTRY"]:
                base = base_settings(row | {"neutralization": neutralization})
                for expression in hypothesis.expressions:
                    request = checked_request(str(expression), base, schema)
                    if request is not None:
                        planned.append((hypothesis.id, request))
        if not planned:
            return None
        markets = Counter((r.settings.region, r.settings.delay) for _, r in planned)
        (region, delay), _ = markets.most_common(1)[0]
        added = await add_study(
            self.state,
            now=utcnow(),
            sampler=HYPOTHESIS_SAMPLER,
            params=HypothesisParams(
                region=region, delay=delay, cores=cores, hypotheses=len(waiting)
            ),
            simulations=len(planned),
            batch_size=(cores + 1) * MULTI_SIMULATION_SIZE,
            template_source=f"{len(planned)} simulations testing {len(waiting)} hypotheses",
            template_name="Hypothesis Writer",
            seeds=settings_sampler.seed_trials([r for _, r in planned]),
        )
        ranges: dict[int, list[int]] = {}
        for number, (hypothesis_id, _) in enumerate(planned):
            ranges.setdefault(hypothesis_id, []).append(number)
        async with self.state.db.session() as session:
            for hypothesis in waiting:
                row = await session.get(Hypothesis, hypothesis.id)
                numbers = ranges.get(hypothesis.id)
                if row is None or not numbers:
                    continue
                row.study_id = added.id
                row.trial_from, row.trial_to = min(numbers), max(numbers) + 1
        self.bus.handoff("hypothesis", "writers", "plan", len(planned), "hypotheses to test")
        return added, len(waiting)

    async def results(self, limit: int = 60) -> list[dict[str, Any]]:
        """The latest hypotheses, each with how its simulations did so far."""
        async with self.state.db.session() as session:
            rows = list(
                await session.scalars(
                    select(Hypothesis).order_by(Hypothesis.id.desc()).limit(limit)
                )
            )
            studies = {r.study_id for r in rows if r.study_id}
            trials = list(await session.scalars(select(Trial).where(Trial.study_id.in_(studies))))
            ids = [t.alpha_id for t in trials if t.alpha_id]
            verdicts = {
                v.alpha_id: v.verdict
                for v in await session.scalars(
                    select(CriticVerdict).where(CriticVerdict.alpha_id.in_(ids))
                )
            }
        stored = await self.state.alphas.by_ids(ids)
        out: list[dict[str, Any]] = []
        for row in rows:
            mine = [
                t
                for t in trials
                if t.study_id == row.study_id
                and row.trial_from is not None
                and row.trial_to is not None
                and row.trial_from <= t.number < row.trial_to
            ]
            done = [t for t in mine if t.alpha_id]
            sharpes = [
                float(s)
                for t in done
                if (s := (stored.get(str(t.alpha_id)) or {}).get("sharpe")) is not None
            ]
            out.append(
                {
                    "id": row.id,
                    "createdAt": row.created_at.isoformat(),
                    "region": row.region,
                    "delay": row.delay,
                    "universe": row.universe,
                    "category": row.category,
                    "datasetId": row.dataset_id,
                    "title": row.title,
                    "rationale": row.rationale,
                    "expressions": list(row.expressions or []),
                    "model": row.model,
                    "enabled": row.enabled,
                    "simulations": (row.trial_to - row.trial_from)
                    if row.trial_from is not None and row.trial_to is not None
                    else 0,
                    "finished": len(done),
                    "passed": sum(1 for t in done if verdicts.get(str(t.alpha_id)) == "pass"),
                    "bestSharpe": max(sharpes) if sharpes else None,
                }
            )
        return out

    async def set_enabled(self, hypothesis_id: int, enabled: bool) -> bool:
        async with self.state.db.session() as session:
            row = await session.get(Hypothesis, hypothesis_id)
            if row is None:
                return False
            row.enabled = enabled
            return True
