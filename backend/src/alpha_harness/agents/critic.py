"""The Critic: judges every finished alpha on whether it should ever be submitted.

BRAIN's checks say whether an alpha *can* be submitted. The Critic asks whether it *should*:
an alpha that passes every check on its in-sample years but falls apart on the years it was
not fitted on will pull Combined Alpha Performance down once it is in the pool. So on top of
BRAIN's checks it applies, from the numbers the vault already stores:

- **Decay**: the test years' Sharpe must keep at least :data:`MIN_DECAY` of the train years',
  or, for an alpha whose train Sharpe is far above the bar, still clear BRAIN's Sharpe limit by
  :data:`STRONG_TEST` on its own while keeping :data:`STRONG_DECAY` of it. A ratio alone
  punishes an alpha for being strong in-sample: test 2.6 against train 5.0 holds up better
  than test 1.2 against train 1.9.
- **Cost**: a margin under :data:`MIN_MARGIN_BPS` leaves too little after trading costs.
- **Size**: at most :data:`MAX_OPERATORS` operators, the Power Pool limit; fewer score better.

Correlation against the submitted pool needs daily PnL, so the Family Clerk checks it later
(:mod:`.families`) and may turn a pass into a near miss.

A **near miss** failed narrowly (every failing BRAIN check within :data:`NEAR` of its limit,
or decay or cost just short). Those go to the repair desk rather than the bin.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import structlog
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert

from ..brain.schemas import QUICK_MODE
from ..db.models import CriticVerdict, utcnow
from ..vault.yields import IGNORED_CHECKS, REFUSING, checks_of, verdict

if TYPE_CHECKING:
    from ..db.sqlite import Database
    from ..tasks import TaskRegistry
    from ..vault.store import AlphaVault
    from .bus import AgentBus

log = structlog.get_logger(__name__)

Verdict = Literal["pass", "near_miss", "fail", "unjudged"]

#: Test-years Sharpe kept, as a share of train-years Sharpe, for an alpha to count as holding up.
MIN_DECAY = 0.6
#: Below this share it is overfitting, not a near miss.
NEAR_DECAY = 0.3
#: A test Sharpe this many times BRAIN's Sharpe limit holds up whatever the train years say...
STRONG_TEST = 1.3
#: ...as long as it keeps at least this share of them.
STRONG_DECAY = 0.45
#: Trading cost assumed per unit traded, in basis points, as on the Portfolio page.
COST_BPS = 2.0
#: Margin that leaves a third of the Sharpe after :data:`COST_BPS`.
MIN_MARGIN_BPS = 3.0
MAX_OPERATORS = 8
#: A failing check within this share of its limit is a near miss.
NEAR = 0.25
#: Failing checks a near miss may have.
NEAR_FAILURES = 2

#: Score weights. Starting guesses, to be refit against BRAIN's performance bars. Simplicity
#: weighs more than it did: Genius breaks ties on fewer distinct operators and fields per Alpha,
#: and BRAIN's own advice is to strip what does not drive performance.
WEIGHTS = {"robust": 0.40, "decay": 0.25, "fitness": 0.2, "simplicity": 0.15}
#: Operators at which simplicity is full, and the count past it at which it is spent:
#: Master consultants average about 3.6 distinct operators per Alpha.
SIMPLE_OPERATORS = 2
SIMPLICITY_SPAN = 6

#: The columns :func:`judge` reads, so a page of the vault reads nothing more.
COLUMNS = (
    "alpha_id, region, delay, universe, sharpe, fitness, margin, train_sharpe, test_sharpe, "
    "operator_count, checks, simulation_mode"
)


@dataclass(slots=True)
class Judgement:
    verdict: Verdict
    score: float
    gates: dict[str, dict[str, Any]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except TypeError, ValueError:
        return None
    return number if math.isfinite(number) else None


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _gate(ok: bool | None, value: Any, note: str) -> dict[str, Any]:
    return {"ok": ok, "value": value, "note": note}


def _sharpe_limit(checks: list[dict[str, Any]]) -> float | None:
    """BRAIN's Sharpe limit for this alpha's market, as its own check reports it."""
    for check in checks:
        if str(check.get("name", "")).upper() == "LOW_SHARPE":
            limit = _num(check.get("limit"))
            if limit is not None and limit > 0:
                return limit
    return None


def judge(row: dict[str, Any]) -> Judgement:
    """The Critic's verdict on one vault row. Pure: reads nothing else."""
    checks = checks_of(row.get("checks"))
    mode = row.get("simulation_mode")
    gates: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    near = True  # stays true while every failure so far is narrow

    brain = verdict(checks, mode)
    if brain is None:
        return Judgement(
            "unjudged", 0.0, {"checks": _gate(None, None, "BRAIN reported no checks yet.")}
        )
    if mode == QUICK_MODE:
        return Judgement(
            "fail",
            0.0,
            {"checks": _gate(False, None, "Quick mode: BRAIN will not take it.")},
            ["Simulated in quick mode, which BRAIN never accepts for submission."],
        )
    failing = [
        c
        for c in checks
        if str(c.get("name", "")).upper() not in IGNORED_CHECKS
        and str(c.get("result", "")).upper() in REFUSING
    ]
    if failing:
        names = ", ".join(str(c.get("name")) for c in failing)
        gates["checks"] = _gate(False, len(failing), f"BRAIN refused: {names}.")
        reasons.append(f"Fails BRAIN checks: {names}.")
        near = near and len(failing) <= NEAR_FAILURES and all(_narrow(c) for c in failing)
    else:
        pending = brain == "pending"
        gates["checks"] = _gate(
            True, 0, "Passes; BRAIN still has checks to run." if pending else "Passes every check."
        )

    sharpe, train, test = (
        _num(row.get("sharpe")),
        _num(row.get("train_sharpe")),
        _num(row.get("test_sharpe")),
    )
    ratio: float | None = None
    if train is not None and test is not None and train > 0:
        ratio = test / train
        bar = _sharpe_limit(checks)
        if ratio >= MIN_DECAY:
            gates["decay"] = _gate(
                True, round(ratio, 2), f"Test years keep {ratio:.0%} of train Sharpe."
            )
        elif bar is not None and ratio >= STRONG_DECAY and test >= STRONG_TEST * bar:
            gates["decay"] = _gate(
                True,
                round(ratio, 2),
                f"Test Sharpe {test:.2f} clears BRAIN's {bar:g} by {test / bar:.1f}x, "
                f"keeping {ratio:.0%} of a strong train Sharpe.",
            )
        else:
            gates["decay"] = _gate(
                False, round(ratio, 2), f"Test years keep only {ratio:.0%} of train Sharpe."
            )
            reasons.append(
                f"Decays out of sample: test Sharpe {test:.2f} against train {train:.2f}."
            )
            near = near and ratio >= NEAR_DECAY
    else:
        gates["decay"] = _gate(None, None, "Simulated without held-out test years.")

    margin = _num(row.get("margin"))
    after_cost: float | None = None
    if margin is not None and sharpe is not None:
        margin_bps = margin * 1e4
        after_cost = sharpe * (1 - COST_BPS / margin_bps) if margin_bps > 0 else -abs(sharpe)
        if margin_bps >= MIN_MARGIN_BPS:
            gates["cost"] = _gate(
                True, round(margin_bps, 2), f"Margin {margin_bps:.1f} bps covers costs."
            )
        else:
            gates["cost"] = _gate(
                False, round(margin_bps, 2), f"Margin {margin_bps:.1f} bps; costs take most of it."
            )
            reasons.append(f"Margin of {margin_bps:.1f} bps is under {MIN_MARGIN_BPS:g} bps.")
            near = near and margin_bps >= COST_BPS
    else:
        gates["cost"] = _gate(None, None, "No margin reported.")

    operators = row.get("operator_count")
    ops = int(operators) if isinstance(operators, int | float) else None
    if ops is not None and ops > MAX_OPERATORS:
        gates["size"] = _gate(False, ops, f"{ops} operators; at most {MAX_OPERATORS}.")
        reasons.append(f"Uses {ops} operators, more than {MAX_OPERATORS}.")
    else:
        gates["size"] = _gate(
            True if ops is not None else None,
            ops,
            "Small enough." if ops else "Operator count unknown.",
        )

    gates["pool"] = _gate(
        None, None, "Waiting for the Family Clerk to compare its PnL with your pool."
    )

    robust_inputs = [v for v in (sharpe, train, test, after_cost) if v is not None]
    robust = min(robust_inputs) if robust_inputs else 0.0
    fitness = _num(row.get("fitness")) or 0.0
    parts = {
        "robust": _clip(robust / 2.5),
        "decay": _clip(ratio) if ratio is not None else 0.5,
        "fitness": _clip(fitness / 2.0),
        "simplicity": _clip(1 - ((ops or 5) - SIMPLE_OPERATORS) / SIMPLICITY_SPAN),
    }
    score = round(100 * sum(WEIGHTS[k] * v for k, v in parts.items()), 1)

    if not reasons:
        return Judgement("pass", score, gates, reasons)
    return Judgement("near_miss" if near else "fail", score, gates, reasons)


#: Failing checks that say nothing about returns, so the search may be steered by them
#: without reading the test years. A failed one costs this share of the search score.
STRUCTURAL_CHECKS = frozenset({"HIGH_TURNOVER", "LOW_TURNOVER", "CONCENTRATED_WEIGHT"})
STRUCTURAL_PENALTY = 0.3
#: What an expression over :data:`MAX_OPERATORS` keeps: Power Pool refuses it outright.
OVERSIZE_KEEPS = 0.5


def search_score(row: dict[str, Any], overlap: float = 0.0) -> float | None:
    """What a lab maximises: the Critic's score from the train years alone, 0 to 100.

    The Critic's own score reads the test years, which is what makes its decay gate a real
    out-of-sample check. A search steered by it would fit those years too and leave the gate
    measuring nothing, so this drops the decay part, scores Sharpe and Fitness on the train
    years, and keeps only the checks that are not about returns. Without a test period there
    are no held-out years to protect, and the whole period is scored.

    ``overlap`` is the share of data fields shared with an alpha you already hold
    (:mod:`.overlap`); a repeat of the pool scores less, before its correlation is known.

    ``None`` when the row has no Sharpe yet.
    """
    sharpe = _num(row.get("train_sharpe"))
    if sharpe is None:
        sharpe = _num(row.get("sharpe"))
    if sharpe is None:
        return None
    fitness = _num(row.get("train_fitness"))
    if fitness is None:
        fitness = _num(row.get("fitness")) or 0.0

    robust = sharpe
    margin = _num(row.get("margin"))
    if margin is not None:
        margin_bps = margin * 1e4
        after_cost = sharpe * (1 - COST_BPS / margin_bps) if margin_bps > 0 else -abs(sharpe)
        robust = min(robust, after_cost)

    operators = row.get("operator_count")
    ops = int(operators) if isinstance(operators, int | float) else None
    parts = {
        "robust": _clip(robust / 2.5),
        "fitness": _clip(fitness / 2.0),
        "simplicity": _clip(1 - ((ops or 5) - SIMPLE_OPERATORS) / SIMPLICITY_SPAN),
    }
    weight = sum(WEIGHTS[k] for k in parts)
    score = 100 * sum(WEIGHTS[k] * v for k, v in parts.items()) / weight

    checks = row.get("checks")
    listed = checks if isinstance(checks, list) else checks_of(checks)
    for check in listed:
        name = str(check.get("name", "")).upper()
        if name in STRUCTURAL_CHECKS and str(check.get("result", "")).upper() in REFUSING:
            score *= 1 - STRUCTURAL_PENALTY
    if ops is not None and ops > MAX_OPERATORS:
        score *= OVERSIZE_KEEPS
    return round(score * overlap_factor(overlap), 2)


def overlap_factor(overlap: float) -> float:
    """What a score keeps at this overlap: all of it at none, half at the same field set.

    Squared, so sharing one field of several costs little and repeating the set costs most.
    """
    return 1.0 - 0.5 * _clip(overlap) ** 2


def _narrow(check: dict[str, Any]) -> bool:
    """Whether a failing check missed its limit by no more than :data:`NEAR` of it."""
    value, limit = _num(check.get("value")), _num(check.get("limit"))
    if value is None or limit is None or limit == 0:
        return False
    return abs(value - limit) <= NEAR * abs(limit)


class Critic:
    """Scores alphas as they land, and the whole vault on request."""

    #: Rows read and written per round of a full re-score.
    PAGE = 2000

    def __init__(self, db: Database, alphas: AlphaVault, bus: AgentBus) -> None:
        self.db = db
        self.alphas = alphas
        self.bus = bus

    async def landed(self, alpha_id: str) -> Judgement | None:
        """Judge one alpha the vault has just stored."""
        self.bus.handoff("brain", "critic", "result", label="finished simulation")
        rows = await self.alphas.catalog.query(
            f"SELECT {COLUMNS} FROM alpha WHERE alpha_id = ?",  # noqa: S608
            [alpha_id],
        )
        if not rows:
            return None
        result = judge(rows[0])
        await self._save([(rows[0], result)], landed=True)
        self._report([result])
        return result

    async def rescore(self, tasks: TaskRegistry) -> int:
        """Judge every alpha in the vault. Returns how many were scored."""
        task = await tasks.start("critic", "Critic scoring the vault")
        done = 0
        try:
            total_rows = await self.alphas.catalog.query("SELECT count(*) AS n FROM alpha")
            total = int(total_rows[0]["n"]) if total_rows else 0
            offset = 0
            while True:
                rows = await self.alphas.catalog.query(
                    f"SELECT {COLUMNS} FROM alpha ORDER BY alpha_id LIMIT ? OFFSET ?",  # noqa: S608
                    [self.PAGE, offset],
                )
                if not rows:
                    break
                judged = [(row, judge(row)) for row in rows]
                await self._save(judged)
                self._report([j for _, j in judged], quiet=True)
                done += len(rows)
                offset += self.PAGE
                await tasks.update(
                    task, progress=done / total if total else None, detail=f"{done:,} of {total:,}"
                )
            counts = await self.counts()
            self.bus.say(
                "critic",
                f"Scored the vault: {counts.get('pass', 0):,} pass, "
                f"{counts.get('near_miss', 0):,} near misses.",
            )
            await tasks.finish(task)
        except Exception as exc:
            await tasks.finish(task, state="failed", error=str(exc))
            raise
        return done

    async def counts(self) -> dict[str, int]:
        async with self.db.session() as session:
            rows = await session.execute(
                select(CriticVerdict.verdict, func.count()).group_by(CriticVerdict.verdict)
            )
            return {str(v): int(n) for v, n in rows.all()}

    async def _save(
        self, judged: list[tuple[dict[str, Any], Judgement]], *, landed: bool = False
    ) -> None:
        now = utcnow()
        values = [
            {
                "alpha_id": str(row["alpha_id"]),
                "verdict": j.verdict,
                "score": j.score,
                "gates": j.gates,
                "reasons": j.reasons,
                "region": row.get("region"),
                "delay": row.get("delay"),
                "universe": row.get("universe"),
                "scored_at": now,
                "landed_at": now if landed else None,
            }
            for row, j in judged
        ]
        if not values:
            return
        statement = insert(CriticVerdict)
        # A re-score replaces the judgement but keeps what the Family Clerk found: the pool
        # gate it filled in is reapplied by the clerk's next pass over this alpha.
        statement = statement.on_conflict_do_update(
            index_elements=[CriticVerdict.alpha_id],
            set_={
                "verdict": statement.excluded.verdict,
                "score": statement.excluded.score,
                "gates": statement.excluded.gates,
                "reasons": statement.excluded.reasons,
                "region": statement.excluded.region,
                "delay": statement.excluded.delay,
                "universe": statement.excluded.universe,
                "scored_at": statement.excluded.scored_at,
                "family_checked_at": None,
                # The first landing is kept: a re-score never makes an old alpha today's.
                "landed_at": func.coalesce(CriticVerdict.landed_at, statement.excluded.landed_at),
            },
        )
        async with self.db.session() as session:
            for start in range(0, len(values), 400):
                await session.execute(statement, values[start : start + 400])

    def _report(self, results: list[Judgement], *, quiet: bool = False) -> None:
        by: dict[str, int] = {}
        for result in results:
            by[result.verdict] = by.get(result.verdict, 0) + 1
            self.bus.count(
                "critic",
                {"pass": "passed", "near_miss": "near misses"}.get(result.verdict, "rejected"),
            )
        if quiet:
            return
        if by.get("pass"):
            self.bus.handoff("critic", "families", "passed", by["pass"], "passed the Critic")
        if by.get("near_miss"):
            self.bus.handoff("critic", "repair", "near_miss", by["near_miss"], "near miss to fix")


def reasons_text(raw: Any) -> list[str]:
    """``reasons`` as stored, read back safely."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
    return [str(r) for r in raw] if isinstance(raw, list) else []
