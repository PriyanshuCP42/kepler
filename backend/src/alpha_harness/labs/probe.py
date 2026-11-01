"""Probe first: a template task spends a little before it spends the rest.

Most new templates miss; on the day this was written, ten ideas spent about 1,400
simulations to learn that none came within half of BRAIN's Sharpe limit. So once a large
template task has :data:`PROBE_TRIALS` simulations back, its best Sharpe is held to
:data:`PROBE_SHARE` of BRAIN's own limit for that market. A task that reaches it carries on;
one that does not is stopped, its unsent simulations dropped. Running it again skips the
probe: the decision to spend anyway is the user's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..vault.yields import checks_of

#: Simulations back before the probe decides.
PROBE_TRIALS = 30
#: Share of BRAIN's Sharpe limit the probe's best must reach.
PROBE_SHARE = 0.8
#: Only tasks at least this many times the probe are probed; a small task is its own probe.
PROBE_FACTOR = 2
#: Where the probe's outcome is kept on the study, so it is decided once.
KEY = "probe"
#: BRAIN's regular Sharpe limit at delay 1, when no result reports one.
FALLBACK_LIMIT = 1.58


@dataclass(frozen=True, slots=True)
class Probe:
    passed: bool
    best: float
    bar: float
    simulations: int

    @property
    def message(self) -> str:
        return (
            f"Stopped by the probe: {self.simulations} simulations reached a best Sharpe of "
            f"{self.best:.2f}, under "
            f"{self.bar:.2f} ({PROBE_SHARE:.0%} of BRAIN's Sharpe limit). Unsent simulations "
            "were dropped; run the task again to spend them anyway."
        )


def sharpe_limit(rows: list[dict[str, Any]]) -> float:
    """BRAIN's Sharpe limit for these results' market, as their own checks report it."""
    for row in rows:
        for check in checks_of(row.get("checks")):
            limit = check.get("limit")
            if str(check.get("name", "")).upper() == "LOW_SHARPE" and isinstance(
                limit, int | float
            ):
                return float(limit)
    return FALLBACK_LIMIT


def judge(rows: list[dict[str, Any]]) -> Probe | None:
    """The probe over these stored results, or ``None`` while fewer than
    :data:`PROBE_TRIALS` of them carry a Sharpe: BRAIN can finish simulations before the vault
    has stored them, and a probe of fewer would cut an idea on less than it promises. Pure."""
    sharpes = [float(s) for r in rows if isinstance(s := r.get("sharpe"), int | float)]
    if len(sharpes) < PROBE_TRIALS:
        return None
    bar = PROBE_SHARE * sharpe_limit(rows)
    best = max(sharpes)
    return Probe(best >= bar, best, bar, len(sharpes))


def due(params: dict[str, Any], max_trials: int, finished: int) -> bool:
    """Whether a task should be probed now: large enough, not yet decided, enough back."""
    return (
        KEY not in params and max_trials >= PROBE_FACTOR * PROBE_TRIALS and finished >= PROBE_TRIALS
    )
