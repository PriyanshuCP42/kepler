"""Quality search: constrained multi-objective NSGA-II, as BRAIN's IQC 2026 study ran it.

A single score asks the search to trade one virtue for another blindly. This asks it for
three at once and keeps the front where none can be bettered without giving up another:

- **K-ratio**: the t-statistic of the cumulative PnL's slope. Two Alphas with one Sharpe can
  climb steadily or in a few jumps; this rewards the steady one.
- **Calmar**: return over the worst drawdown, return per unit of pain. A flat Alpha that
  does nothing scores nothing.
- **Fitness**: BRAIN's own quality score.

Each over the train years only, so the held-out test years stay an out-of-sample check, as
they are for the Critic's search score.

**Constraints**, each feasible at 0 or less. An infeasible trial stays in the gene pool, so
the sampler learns which regions to avoid rather than forgetting them:

- train Sharpe at least :data:`MIN_SHARPE`;
- none of BRAIN's :data:`HARD_CHECKS` failed, and no check in ERROR;
- margin at least :data:`metrics.DEFAULT_COST_BPS`, what the tables charge to trade;
- correlation with your submitted pool at most :data:`families.POOL_LIMIT`, measured live
  against the pool as it grows: a crowded idea is infeasible while it is searched, not found
  out after.

Daily PnL costs one read from BRAIN, so it is read only for a trial that clears the first
three; any other scores nothing on the first two objectives, which constrained domination
ranks below every feasible trial anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np

from ..brain.platform import TEST_PERIOD_YEARS
from ..vault import metrics

#: The objectives, maximised in this order.
OBJECTIVES = ("train_k_ratio", "train_calmar", "train_fitness")
#: The constraints, in the order they are stored.
CONSTRAINTS = ("sharpe", "checks", "margin", "pool")
#: The least train Sharpe a feasible trial has.
MIN_SHARPE = 1.0
#: Checks whose failure makes a trial infeasible however good its curve: BRAIN refuses it.
HARD_CHECKS = frozenset({"LOW_TURNOVER", "HIGH_TURNOVER", "LOW_SUB_UNIVERSE_SHARPE"})
#: Correlation with a submitted alpha at which a candidate repeats your pool (as the Family
#: Clerk holds it).
POOL_LIMIT = 0.6
#: The elite pool's size: a fifth of the fields searched, between these.
POPULATION_SHARE = 0.2
MIN_POPULATION = 100
MAX_POPULATION = 500
#: A small task still gets this many generations, however few fields it reads.
MIN_GENERATIONS = 3
#: NSGA-II's breeding: crossover between two parents most of the time, half the genes
#: re-drawn, each crossed gene taken from either parent.
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.5
SWAPPING_PROB = 0.5
#: Where a trial's generation is kept, so a restart rebuilds the search where it was.
GENERATION_KEY = "NSGAIISampler:generation"


def is_quality(keys: list[str] | tuple[str, ...]) -> bool:
    return tuple(keys) == OBJECTIVES


def population(fields: int, max_trials: int) -> int:
    """The elite pool's size for a task over ``fields`` field choices and ``max_trials``."""
    share = round(POPULATION_SHARE * fields)
    size = min(MAX_POPULATION, max(MIN_POPULATION, share))
    return max(2, min(size, max_trials // MIN_GENERATIONS))


def hard_failures(checks: list[dict[str, Any]]) -> int:
    """Hard checks failed, plus any check in ERROR."""
    count = 0
    for check in checks:
        name = str(check.get("name", "")).upper()
        result = str(check.get("result", "")).upper()
        if result == "ERROR" or (result == "FAIL" and name in HARD_CHECKS):
            count += 1
    return count


@dataclass(frozen=True, slots=True)
class Scored:
    values: list[float]
    constraints: list[float]
    #: What it read, for the result row.
    detail: dict[str, float | None]

    @property
    def feasible(self) -> bool:
        return all(c <= 0 for c in self.constraints)


def cheap_constraints(
    train_sharpe: float | None, margin: float | None, checks: list[dict[str, Any]]
) -> list[float]:
    """The three constraints read off the result itself, before any PnL is fetched."""
    sharpe = MIN_SHARPE - (train_sharpe if train_sharpe is not None else 0.0)
    margin_bps = (margin or 0.0) * 1e4
    return [
        round(sharpe, 4),
        float(hard_failures(checks)),
        round(metrics.DEFAULT_COST_BPS - margin_bps, 4),
    ]


def split_of(days: dict[date, float], test_start: date | str | None) -> date:
    """Where the train years end: the stored test start, else BRAIN's test period back from
    the last day."""
    if isinstance(test_start, date):
        return test_start
    if isinstance(test_start, str) and test_start:
        return date.fromisoformat(test_start[:10])
    last = max(days)
    return last - timedelta(days=round(365.25 * TEST_PERIOD_YEARS))


def train_curve(
    days: dict[date, float], test_start: date | str | None
) -> tuple[float | None, float | None]:
    """K-ratio and Calmar over the train years. Pure."""
    if not days:
        return None, None
    split = split_of(days, test_start)
    train = sorted(d for d in days if d < split)
    if len(train) < 60:
        return None, None
    pnl = np.array([days[d] for d in train], dtype=float)
    turnover = np.zeros_like(pnl)
    k = metrics.k_ratio(pnl, turnover)
    found = metrics.stats(pnl, turnover)
    calmar = None
    if found is not None and found.drawdown:
        calmar = found.returns / found.drawdown
    return k, calmar


def score(
    train_sharpe: float | None,
    train_fitness: float | None,
    margin: float | None,
    checks: list[dict[str, Any]],
    curve: tuple[float | None, float | None] = (None, None),
    pool_correlation: float | None = None,
) -> Scored:
    """A trial's objectives and constraints. Pure.

    Without a curve (not fetched, or too short) the first two objectives score 0; without a
    pool correlation the pool constraint is met, as nothing says it is crowded.
    """
    k, calmar = curve
    constraints = [
        *cheap_constraints(train_sharpe, margin, checks),
        round(pool_correlation - POOL_LIMIT, 4) if pool_correlation is not None else 0.0,
    ]
    return Scored(
        values=[
            float(k) if k is not None else 0.0,
            float(calmar) if calmar is not None else 0.0,
            float(train_fitness) if train_fitness is not None else 0.0,
        ],
        constraints=constraints,
        detail={
            "kRatio": None if k is None else round(k, 3),
            "calmar": None if calmar is None else round(calmar, 3),
            "poolCorrelation": None if pool_correlation is None else round(pool_correlation, 3),
        },
    )
