"""PnL twins: what BRAIN's production-correlation check will say, before it is asked.

Production correlation is measured against every Alpha on the platform, so it cannot be
worked out here. But it can be *looked up*: an Alpha whose daily PnL moves with one BRAIN has
already checked lands where that one did. Measured on the checked Alphas in the vault, a pair
whose daily PnL correlated at 0.9 or more fell on the same side of the 0.7 ceiling 94% of the
time, their production correlations about 0.01 apart.

Nothing else tried predicted it. How crowded the fields are, the datasets and operators
shared with checked Alphas, Sharpe and turnover: each was scarcely better than the region's
average. So the gate uses only a twin: it rejects an Alpha whose twin failed, which saves
the check, and asks BRAIN about everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..vault.yields import checks_of

#: Daily PnL correlation at which two Alphas count as twins.
TWIN_CORRELATION = 0.9
#: Shared trading days a correlation needs before it is believed.
MIN_SHARED_DAYS = 500


@dataclass(frozen=True, slots=True)
class Twin:
    #: The checked Alpha this one moves with.
    alpha_id: str
    #: Their daily PnL correlation.
    correlation: float
    #: What BRAIN measured for the twin.
    prod_correlation: float


def prod_value(row: dict[str, Any]) -> float | None:
    """The production correlation BRAIN measured for a stored Alpha, or ``None``."""
    kept = row.get("checks")
    checks = kept if isinstance(kept, list) else checks_of(kept)
    for check in checks:
        if check.get("name") == "PROD_CORRELATION" and isinstance(check.get("value"), int | float):
            return float(check["value"])
    value = row.get("prod_correlation")
    return float(value) if isinstance(value, int | float) else None


def correlations(grid: np.ndarray, column: int, others: list[int]) -> list[float | None]:
    """Daily PnL correlation of one column with each of ``others``, on the days both traded.

    ``None`` where they share fewer than :data:`MIN_SHARED_DAYS` days or one never moves.
    """
    mine = grid[:, column]
    out: list[float | None] = []
    for k in others:
        theirs = grid[:, k]
        both = ~np.isnan(mine) & ~np.isnan(theirs)
        if int(both.sum()) < MIN_SHARED_DAYS:
            out.append(None)
            continue
        a, b = mine[both], theirs[both]
        if a.std() == 0 or b.std() == 0:
            out.append(None)
            continue
        out.append(float(np.corrcoef(a, b)[0, 1]))
    return out


def nearest(
    ids: list[str],
    grid: np.ndarray,
    candidate: str,
    checked: dict[str, float],
) -> Twin | None:
    """The checked Alpha ``candidate`` moves most with, if they are twins.

    ``ids`` names ``grid``'s columns; ``checked`` gives each checked Alpha's production
    correlation. A candidate is never its own twin.
    """
    if candidate not in ids:
        return None
    index = {a: i for i, a in enumerate(ids)}
    others = [a for a in checked if a in index and a != candidate]
    if not others:
        return None
    found = correlations(grid, index[candidate], [index[a] for a in others])
    best = max(
        ((rho, a) for rho, a in zip(found, others, strict=True) if rho is not None),
        default=None,
    )
    if best is None or best[0] < TWIN_CORRELATION:
        return None
    return Twin(alpha_id=best[1], correlation=best[0], prod_correlation=checked[best[1]])


async def find(state: Any, regions: dict[str, str]) -> dict[str, Twin]:
    """Each candidate's checked twin in its own region, by candidate id.

    ``regions`` maps each candidate to its region. Candidates without a stored PnL, or with
    no twin, are absent.
    """
    if not regions:
        return {}
    checked_rows = await state.alphas.prod_checked(sorted(set(regions.values())))
    checked: dict[str, dict[str, float]] = {}
    for row in checked_rows:
        value = prod_value(row)
        if value is not None:
            checked.setdefault(str(row["region"]), {})[str(row["alpha_id"])] = value
    wanted = [*regions, *(a for by in checked.values() for a in by)]
    ids, _, grid = await state.alphas.pnl_grid(list(dict.fromkeys(wanted)))
    out: dict[str, Twin] = {}
    for candidate, region in regions.items():
        twin = nearest(ids, grid, candidate, checked.get(region, {}))
        if twin is not None:
            out[candidate] = twin
    return out


__all__ = [
    "MIN_SHARED_DAYS",
    "TWIN_CORRELATION",
    "Twin",
    "correlations",
    "find",
    "nearest",
    "prod_value",
]
