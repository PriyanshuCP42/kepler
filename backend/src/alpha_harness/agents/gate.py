"""The Submit Gate: one verdict on whether an Alpha should be submitted, and why.

BRAIN's checks say whether an Alpha *can* be submitted. The gate asks whether it *should*:
whether it earns more than its trading costs, gives back too much on its bad stretches,
lowers your combined or Power Pool performance, repeats an idea you have just submitted, or
leans on a reversion term BRAIN warns it may stop accepting. Every test that decided a pick
by hand is here, so Kepler reaches the same verdict without anyone at the keyboard.

Four verdicts:

- ``READY``: passes every test, BRAIN's own check included. Submit it.
- ``CHECK``: passes every local test, but BRAIN's check or the pool effect is not known yet.
  Run BRAIN's check, then judge again.
- ``REPAIR``: fails only narrowly (Fitness, margin, returns per drawdown, Sharpe, or how much of
  the train Sharpe the test years keep). The Repair Engineer's work, not the bin.
- ``REJECT``: fails something no small change fixes: BRAIN refuses it, it is crowded (or its
  PnL twin was), it lowers a pool, it repeats a recent idea, or it has a reversion component.

Pure: everything it reads is in :class:`Facts`, so every rule is tested without BRAIN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

#: Margin an Alpha must clear: the 5 bps trading cost After-Cost Mode charges.
MIN_MARGIN = 0.0005
#: Annual returns an Alpha must earn per unit of its worst drawdown.
MIN_RETURN_TO_DRAWDOWN = 1.5
#: Production and self correlation BRAIN refuses at.
CORRELATION_CEILING = 0.7
#: Power Pool correlation an Alpha must stay under to count as Power Pool.
POWER_POOL_CORRELATION_CEILING = 0.5
#: How much of its train Sharpe the test years must keep.
MIN_TEST_KEEP = 0.7
#: How close to a limit a failure must be to count as a near miss, as a share of the limit.
NEAR = 0.15
#: Sharpe a pure Power Pool Alpha must reach: the floor in Power Pool mode, where the regular
#: Sharpe and Fitness limits are only warnings.
POWER_POOL_MIN_SHARPE = 1.0
#: Power Pool's size limits.
MAX_OPERATORS = 8
MAX_FIELDS = 3

#: BRAIN checks the gate reads itself rather than as a plain refusal.
_JUDGED_HERE = frozenset(
    {
        "PROD_CORRELATION",
        "SELF_CORRELATION",
        "POWER_POOL_CORRELATION",
        "REVERSION_COMPONENT",
        "LOW_FITNESS",
        "LOW_SHARPE",
    }
)


class Verdict(StrEnum):
    READY = "READY"
    CHECK = "CHECK"
    REPAIR = "REPAIR"
    REJECT = "REJECT"


@dataclass(slots=True)
class Facts:
    """What the gate needs to know about one Alpha. ``None`` means not known yet."""

    alpha_id: str
    sharpe: float | None
    fitness: float | None
    sharpe_limit: float | None
    fitness_limit: float | None
    margin: float | None
    returns: float | None
    drawdown: float | None
    train_sharpe: float | None = None
    test_sharpe: float | None = None
    prod_correlation: float | None = None
    self_correlation: float | None = None
    power_pool_correlation: float | None = None
    #: BRAIN's own check has run on it (its correlations included where it gives them).
    checked: bool = False
    #: Checks BRAIN failed it on, by name.
    brain_fails: list[str] = field(default_factory=list)
    #: BRAIN warned the expression has a reversion component.
    reversion: bool = False
    #: Change in combined and in Power Pool Sharpe if it joins.
    whole_delta: float | None = None
    power_delta: float | None = None
    #: The recent submission it repeats (same region and dataset), if any.
    lookalike_of: str | None = None
    operators: int | None = None
    fields: int | None = None
    #: Matches the month's Power Pool theme in its market (``None`` where there is none).
    theme_match: bool | None = None
    #: A checked Alpha its daily PnL moves with, and the production correlation BRAIN
    #: measured for it: what this one's check will most likely say (see ``agents.twins``).
    twin_of: str | None = None
    twin_prod: float | None = None


@dataclass(slots=True)
class Judgement:
    alpha_id: str
    verdict: Verdict
    #: Every reason it is not READY, worst first; empty when READY.
    reasons: list[str]
    #: Within Power Pool's size and correlation limits.
    power_pool: bool
    #: Clears the regular Sharpe and Fitness limits.
    regular: bool
    return_to_drawdown: float | None
    #: How strong a READY pick is, for ordering: higher first.
    score: float


def return_to_drawdown(returns: float | None, drawdown: float | None) -> float | None:
    if returns is None or drawdown is None or drawdown <= 0:
        return None
    return returns / drawdown


def _short(value: float | None, limit: float | None) -> str | None:
    """``"near"`` when ``value`` misses ``limit`` by at most :data:`NEAR`, ``"far"`` when by
    more, ``None`` when it clears it or either is unknown."""
    if value is None or limit is None or value >= limit:
        return None
    return "near" if value >= limit * (1 - NEAR) else "far"


def judge(f: Facts) -> Judgement:
    """The gate's verdict on one Alpha, with every reason it falls short."""
    hard: list[str] = []
    near: list[str] = []
    unknown: list[str] = []

    if f.reversion:
        hard.append("REVERSION_COMPONENT")
    refused = [n for n in f.brain_fails if n not in _JUDGED_HERE]
    hard.extend(f"BRAIN_{n}" for n in refused)
    if f.prod_correlation is not None and f.prod_correlation >= CORRELATION_CEILING:
        hard.append("PROD_CORRELATION")
    elif (
        f.prod_correlation is None
        and f.twin_prod is not None
        and f.twin_prod >= CORRELATION_CEILING
    ):
        # Its twin failed; 94% of twins land on the same side. Not worth a check.
        hard.append("PROD_TWIN")
    if f.self_correlation is not None and f.self_correlation >= CORRELATION_CEILING:
        hard.append("SELF_CORRELATION")
    if f.lookalike_of:
        hard.append("LOOKALIKE")
    if f.whole_delta is not None and f.whole_delta < 0:
        hard.append("LOWERS_COMBINED")
    if f.power_delta is not None and f.power_delta < 0:
        hard.append("LOWERS_POWER_POOL")

    regular = (
        f.sharpe is not None
        and f.fitness is not None
        and f.sharpe_limit is not None
        and f.fitness_limit is not None
        and f.sharpe >= f.sharpe_limit
        and f.fitness >= f.fitness_limit
    )
    power_pool = (
        f.operators is not None
        and f.operators <= MAX_OPERATORS
        and f.fields is not None
        and f.fields <= MAX_FIELDS
        and (
            f.power_pool_correlation is None
            or f.power_pool_correlation < POWER_POOL_CORRELATION_CEILING
        )
    )

    themed = power_pool and bool(f.theme_match)
    checks: list[tuple[str, float | None, float | None]] = [("MARGIN", f.margin, MIN_MARGIN)]
    if themed and not regular:
        # A pure Power Pool Alpha its theme covers: Power Pool's own Sharpe floor, no Fitness.
        checks.insert(0, ("SHARPE", f.sharpe, POWER_POOL_MIN_SHARPE))
    else:
        checks[:0] = [("SHARPE", f.sharpe, f.sharpe_limit), ("FITNESS", f.fitness, f.fitness_limit)]
    for name, value, limit in checks:
        miss = _short(value, limit)
        if miss == "far":
            hard.append(f"LOW_{name}")
        elif miss == "near":
            near.append(f"LOW_{name}")
    ratio = return_to_drawdown(f.returns, f.drawdown)
    miss = _short(ratio, MIN_RETURN_TO_DRAWDOWN)
    if miss == "far":
        hard.append("RETURN_TO_DRAWDOWN")
    elif miss == "near":
        near.append("RETURN_TO_DRAWDOWN")
    if f.train_sharpe and f.test_sharpe is not None and f.train_sharpe > 0:
        keep = f.test_sharpe / f.train_sharpe
        if keep < MIN_TEST_KEEP * (1 - NEAR):
            hard.append("TEST_DECAY")
        elif keep < MIN_TEST_KEEP:
            near.append("TEST_DECAY")
    # Neither a regular Alpha nor a Power Pool one a theme covers: BRAIN refuses it, unless
    # the shortfall is a near miss already listed.
    if not regular and not themed and not any(r in near for r in ("LOW_SHARPE", "LOW_FITNESS")):
        hard.append("NOT_SUBMITTABLE")

    if not f.checked:
        unknown.append("BRAIN_CHECK_NOT_RUN")
    if f.whole_delta is None or f.power_delta is None:
        unknown.append("POOL_EFFECT_NOT_MEASURED")

    if hard:
        verdict = Verdict.REJECT
    elif near:
        verdict = Verdict.REPAIR
    elif unknown:
        verdict = Verdict.CHECK
    else:
        verdict = Verdict.READY

    score = (f.whole_delta or 0.0) + (f.power_delta or 0.0)
    score += 0.01 * ((ratio or 0.0) + (f.fitness or 0.0))
    prod = f.prod_correlation if f.prod_correlation is not None else f.twin_prod
    score -= 0.05 * (prod if prod is not None else CORRELATION_CEILING)
    return Judgement(
        alpha_id=f.alpha_id,
        verdict=verdict,
        reasons=[*hard, *near, *unknown],
        power_pool=power_pool,
        regular=regular,
        return_to_drawdown=ratio,
        score=score,
    )


__all__ = [
    "CORRELATION_CEILING",
    "MIN_MARGIN",
    "MIN_RETURN_TO_DRAWDOWN",
    "Facts",
    "Judgement",
    "Verdict",
    "judge",
    "return_to_drawdown",
]
