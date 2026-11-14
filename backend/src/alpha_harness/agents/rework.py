"""Rework: send BRAIN's narrow refusals of strong Alphas to the Repair Engineer, unasked.

Broad searches rarely pass BRAIN's submission check on their own; what passed on this account
was nearly always a strong Alpha BRAIN refused by a little, reworked once: a production
correlation just over the ceiling re-neutralized inside sector and statistical clusters, a
signal too concentrated in small stocks tilted towards larger ones. So each round the Daily
Autopilot reads recent Alphas BRAIN has checked, keeps the strong ones every failure of which
a repair move addresses, and hands them to the Repair Engineer.

An Alpha is worth reworking when it already clears the Sharpe bar, misses Fitness by no more
than :data:`FITNESS_SLACK`, fails at most :data:`MAX_FAILURES` checks, each of them one a move
fixes, carries no reversion warning, repeats no recent submission, and has not been reworked
before.

:func:`worth_reworking` holds the rule and is pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import structlog
from sqlalchemy import select

from ..db.models import RepairAttempt, utcnow
from ..vault.yields import checks_of
from . import repair

log = structlog.get_logger(__name__)

#: How far back an Alpha is read: older near misses had their chance.
LOOKBACK = timedelta(days=4)
#: Checks failed beyond which one Alpha needs a new idea, not a repair.
MAX_FAILURES = 2
#: Fitness may sit this far under its limit: the moves raise it as they fix the rest.
FITNESS_SLACK = 0.85
#: Alphas sent per rework, and the fixes tried on each.
PER_ROUND = 4
PER_ALPHA = 4
#: Every failure some repair move addresses.
FIXABLE = frozenset().union(*(m.fixes for m in repair.MOVES))


@dataclass(frozen=True, slots=True)
class Candidate:
    alpha_id: str
    region: str
    sharpe: float
    failures: tuple[str, ...]


def _value(checks: list[dict[str, Any]], name: str, key: str) -> Any:
    return next((c.get(key) for c in checks if c.get("name") == name), None)


def worth_reworking(row: dict[str, Any]) -> tuple[str, ...] | None:
    """What a repair would fix in this stored Alpha, or ``None`` when it is not worth one."""
    checks = checks_of(row.get("checks"))
    if not checks:
        return None
    if any(
        c.get("name") == "REVERSION_COMPONENT" and str(c.get("result")).upper() != "PASS"
        for c in checks
    ):
        return None
    sharpe, fitness = row.get("sharpe"), row.get("fitness")
    sharpe_limit = _value(checks, "LOW_SHARPE", "limit")
    fitness_limit = _value(checks, "LOW_FITNESS", "limit")
    if not all(isinstance(x, int | float) for x in (sharpe, fitness, sharpe_limit, fitness_limit)):
        return None
    if sharpe < sharpe_limit or fitness < fitness_limit * FITNESS_SLACK:
        return None
    found = [f for f in repair.failures(row, {}) if f != "LOW_SHARPE"]
    # Production correlation over the repairable limit is dropped by ``failures``; it still
    # rules the Alpha out, since no move brings a crowd that far back.
    prod = _value(checks, repair.PROD, "value")
    if isinstance(prod, int | float) and prod > repair.REPAIRABLE_PROD:
        return None
    if not found or len(found) > MAX_FAILURES or any(f not in FIXABLE for f in found):
        return None
    return tuple(found)


async def candidates(state: Any, limit: int = PER_ROUND) -> list[Candidate]:
    """The strongest recent Alphas worth a rework, not reworked yet, no lookalikes."""
    from .picks import LOOKALIKE_DAYS, datasets_of

    since = (utcnow() - LOOKBACK).replace(tzinfo=None)
    rows = await state.alphas.catalog.query(
        "SELECT alpha_id, region, delay, expression, sharpe, fitness, checks FROM alpha "
        "WHERE status = 'UNSUBMITTED' AND date_created >= ? AND checks IS NOT NULL "
        "ORDER BY sharpe DESC LIMIT 400",
        [since],
    )
    async with state.db.session() as session:
        tried = set(await session.scalars(select(RepairAttempt.origin_alpha_id).distinct()))
    picked = [
        (row, found)
        for row in rows
        if str(row["alpha_id"]) not in tried and (found := worth_reworking(row)) is not None
    ]
    if not picked:
        return []

    members = await state.alphas.submitted_members()
    cutoff = (utcnow() - timedelta(days=LOOKALIKE_DAYS)).date().isoformat()
    recent = [m for m in members if str(m.get("date_submitted") or "")[:10] >= cutoff]
    texts = {str(r["alpha_id"]): str(r.get("expression") or "") for r, _ in picked}
    texts |= {f"sub:{m['alpha_id']}": str(m.get("expression") or "") for m in recent}
    owned = await datasets_of(state, texts)
    taken: dict[str, set[str]] = {}
    for m in recent:
        taken.setdefault(str(m.get("region")), set()).update(owned.get(f"sub:{m['alpha_id']}", ()))

    out: list[Candidate] = []
    for row, found in picked:
        alpha_id, region = str(row["alpha_id"]), str(row.get("region"))
        if owned.get(alpha_id, frozenset()) & taken.get(region, set()):
            continue  # a lookalike: reworking it cannot make it submittable
        out.append(Candidate(alpha_id, region, float(row["sharpe"]), found))
        if len(out) >= limit:
            break
    return out


__all__ = [
    "FIXABLE",
    "MAX_FAILURES",
    "PER_ALPHA",
    "PER_ROUND",
    "Candidate",
    "candidates",
    "worth_reworking",
]
