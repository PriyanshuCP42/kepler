"""This month's Power Pool themes, and whether BRAIN can take an alpha at all.

An alpha that clears BRAIN's regular limits (Sharpe, Fitness, weight concentration and the
like) is submitted as a regular alpha. One that misses them is taken only as a **pure Power
Pool** alpha, and BRAIN takes those only with a Power Pool description of 100 characters or
more *and* in a market one of the month's Power Pool themes covers: otherwise its check fails
"Pure Power Pool submission does not match any Power Pool Theme". The threshold misses
themselves then stay WARNING.

BRAIN publishes the themes in a help article, not its API, but lists them in every alpha's
``MATCHES_THEMES`` check, simulated or submission-checked: ``{"id", "name", "multiplier"}``,
named like ``GLB/D1 Power Pool Nov`26`` or ``ASI/D1 MINVOL10M Theme``. They are read from the
newest alphas, so the month's change is picked up by itself.

What a theme asks beyond its market (November 2026's GLB theme: Americas Sharpe above 1) is
in the article alone, so it is written down here, in :data:`CONDITIONS`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from ..vault.yields import checks_of

#: Seconds the themes are kept before they are read again.
TTL = 1800.0
#: Newest alphas read for the themes.
NEWEST = 300
#: Refused or warned checks that only a pure Power Pool alpha may leave unmet.
REGULAR_LIMITS = frozenset(
    {
        "LOW_SHARPE",
        "LOW_FITNESS",
        "LOW_2Y_SHARPE",
        "IS_LADDER_SHARPE",
        "CONCENTRATED_WEIGHT",
        "LOW_SUB_UNIVERSE_SHARPE",
        "LOW_TURNOVER",
        "HIGH_TURNOVER",
    }
)
#: What a theme asks beyond its market, from BRAIN's help article "Current month Power Pool
#: Themes": the checks an alpha must clear to be in it, keyed by the theme's name as
#: :func:`parse` gives it. BRAIN refuses a pure Power Pool alpha that misses one with "does not
#: match any Power Pool Theme". A theme not listed asks for its market alone, so last month's
#: entries are simply never matched again.
CONDITIONS: dict[str, tuple[str, ...]] = {
    "GLB/D1 Power Pool Nov'26": ("LOW_GLB_AMER_SHARPE",),
}
#: When a theme runs, by its name: BRAIN's checks name a theme but not its dates. From
#: WorldQuant's webinar slide "GLB D1 Power Pool Theme" (November 2026): 26 October to
#: 22 November, a week before the month ends. Its leaderboard snapshots the tagged,
#: theme-matching alphas at the end.
WINDOWS: dict[str, tuple[str, str]] = {
    "GLB/D1 Power Pool Nov'26": ("2026-10-26", "2026-11-22"),
}
#: Tagged theme-matching Power Pool alphas a consultant needs at the snapshot to win awards.
MIN_FOR_AWARDS = 5
_NAME = re.compile(r"^\s*([A-Z]{3})\s*/\s*D(\d)\b\s*(.*)$")
#: Words of a theme's name that say nothing about the market it covers.
_PLAIN = frozenset({"POWER", "POOL", "THEME"})


@dataclass(frozen=True, slots=True)
class Theme:
    id: str
    name: str
    multiplier: float
    region: str | None
    delay: int | None
    #: What else the name narrows it to, e.g. ``MINVOL10M``, which is an ASI universe.
    qualifier: str | None
    #: The checks it asks an alpha to clear beyond its market: :data:`CONDITIONS`.
    requires: tuple[str, ...] = ()

    def covers(
        self,
        region: Any,
        delay: Any,
        universe: Any = None,
        checks: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Whether an alpha in this market is in the theme. A qualifier that names a universe
        (one word of capitals and digits) must match the alpha's universe when it is known,
        and the alpha must not miss any check the theme :attr:`requires`. One its checks do not
        report (a quick simulation's few) is not held against it."""
        if self.region is None or str(region) != self.region:
            return False
        if delay is None or int(delay) != self.delay:
            return False
        if self.universe and universe and str(universe).upper() != self.universe:
            return False
        return not missed(checks or [], self.requires)

    @property
    def universe(self) -> str | None:
        """The universe the qualifier names, when it reads like one (``MINVOL10M``)."""
        q = self.qualifier
        return q if q and re.fullmatch(r"[A-Z][A-Z0-9_]*\d[A-Z0-9_]*", q) else None


def parse(raw: dict[str, Any]) -> Theme | None:
    name = str(raw.get("name") or "")
    if not name:
        return None
    found = _NAME.match(name)
    region, delay, rest = (
        (found.group(1), int(found.group(2)), found.group(3)) if found else (None, None, name)
    )
    extra = [
        w
        for w in re.findall(r"[A-Z][A-Z0-9_]+", rest)
        if w not in _PLAIN and not re.fullmatch(r"[A-Z][a-z]{2}", w)
    ]
    multiplier = raw.get("multiplier")
    return Theme(
        id=str(raw.get("id") or name),
        name=name.replace("`", "'"),
        multiplier=float(multiplier) if isinstance(multiplier, int | float) else 1.0,
        region=region,
        delay=delay,
        qualifier=" ".join(extra) or None,
        requires=CONDITIONS.get(name.replace("`", "'"), ()),
    )


def from_checks(checks: list[dict[str, Any]]) -> list[Theme]:
    found: list[Theme] = []
    for check in checks:
        if str(check.get("name", "")).upper() != "MATCHES_THEMES":
            continue
        found.extend(
            theme
            for raw in check.get("themes") or []
            if isinstance(raw, dict) and (theme := parse(raw)) is not None
        )
    return found


def missed(checks: list[dict[str, Any]], names: frozenset[str] | tuple[str, ...]) -> list[str]:
    """Which of ``names`` this alpha misses, warned or refused, by value against limit."""
    out: list[str] = []
    for check in checks:
        name = str(check.get("name", "")).upper()
        result = str(check.get("result", "")).upper()
        if name not in names or result not in ("FAIL", "WARNING", "ERROR"):
            continue
        value, limit = check.get("value"), check.get("limit")
        if not isinstance(value, int | float) or not isinstance(limit, int | float):
            continue
        if (value < limit) if name.startswith("LOW_") else (value > limit):
            out.append(name)
    return out


def needs_power_pool(checks: list[dict[str, Any]]) -> list[str]:
    """The regular limits this alpha misses, warned or refused: what only Power Pool forgives."""
    return missed(checks, REGULAR_LIMITS)


def is_power_pool(checks: list[dict[str, Any]]) -> bool:
    return any(str(c.get("name", "")).upper() == "POWER_POOL_CORRELATION" for c in checks)


def takeable(row: dict[str, Any], themes: list[Theme]) -> bool:
    """Whether BRAIN can take this alpha on its checks: clear of every regular limit, or a
    Power Pool alpha in a market one of the month's themes covers. ``True`` while the themes
    are unknown, rather than holding everything back for want of a check to read them from."""
    checks = (
        checks_of(row.get("checks")) if not isinstance(row.get("checks"), list) else row["checks"]
    )
    if not needs_power_pool(checks):
        return True
    if not is_power_pool(checks):
        return False
    if not themes:
        return True
    return any(
        t.covers(row.get("region"), row.get("delay"), row.get("universe"), checks) for t in themes
    )


class Themes:
    """The month's themes, read from the newest alphas' checks and kept for :data:`TTL`."""

    def __init__(self, state: Any) -> None:
        self.state = state
        self._themes: list[Theme] = []
        self._read = float("-inf")

    async def current(self) -> list[Theme]:
        if time.monotonic() - self._read < TTL:
            return self._themes
        rows = await self.state.alphas.catalog.query(
            "SELECT checks FROM alpha WHERE checks LIKE '%MATCHES_THEMES%' "
            "ORDER BY date_created DESC NULLS LAST LIMIT ?",
            [NEWEST],
        )
        seen: dict[str, Theme] = {}
        for row in rows:
            for theme in from_checks(checks_of(row.get("checks"))):
                seen.setdefault(theme.id, theme)
        if not seen:
            # Fall back to the submission checks kept from BRAIN.
            from sqlalchemy import select

            from ..db.models import BrainCache

            async with self.state.db.session() as session:
                kept = list(
                    await session.scalars(
                        select(BrainCache.body)
                        .where(BrainCache.key.like("check:%"))
                        .order_by(BrainCache.fetched_at.desc())
                        .limit(20)
                    )
                )
            for body in kept:
                data = body if isinstance(body, dict) else json.loads(body or "{}")
                for theme in from_checks(((data.get("is") or {}).get("checks")) or []):
                    seen.setdefault(theme.id, theme)
        self._themes = sorted(seen.values(), key=lambda t: -t.multiplier)
        self._read = time.monotonic()
        return self._themes
