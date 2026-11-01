"""Path to a Genius level: where you stand against its bars and against those already there.

A level takes three bars cleared by quarter end (signals, pyramids, and the best of four
combined performances) and then a place among everyone who cleared them, decided by a
tie-breaker: fewer distinct operators and fields per Alpha, more of them in all, and the longest
simulation streak (BRAIN's Genius documentation). BRAIN does not publish the bars where Kepler
can read them, so :data:`.genius.BARS` holds them; what consultants at the level have this
quarter is read from BRAIN's Genius leaderboard, the yardstick for the tie-breaker.

Alphas submitted in a quarter's last month may not reach that quarter's combined performance,
which takes weeks to compute: performance has to be won before then.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import median
from typing import Any

from . import genius

#: Seconds the leaderboard and your summary are kept before BRAIN is asked again.
TTL = 6 * 3600
#: Leaderboard rows read: the board is ranked, and the levels sit near its top.
BOARD_ROWS = 3000
_PAGE = 500

#: The four combined performances, of which the best counts.
PERFORMANCES = {
    "combinedAlphaPerformance": "All your Alphas",
    "combinedSelectedAlphaPerformance": "Alphas WorldQuant selected",
    "combinedPowerPoolAlphaPerformance": "Power Pool Alphas",
    "combinedOsmosisPerformance": "Osmosis",
}
#: Tie-breaker figures, and whether lower is better.
TIE_BREAKERS = {
    "operatorAvg": ("Distinct operators per Alpha", True),
    "fieldAvg": ("Distinct fields per Alpha", True),
    "operatorCount": ("Distinct operators in all", False),
    "fieldCount": ("Distinct fields in all", False),
    "maxSimulationStreak": ("Longest simulation streak (days)", False),
}
LEVEL_NAMES = {"Expert": "EXPERT", "Master": "MASTER", "Grandmaster": "GRANDMASTER"}


@dataclass(slots=True)
class Row:
    key: str
    label: str
    you: float | None
    bar: float | None
    peers: float | None
    #: Lower is better (tie-breaker averages).
    lower_better: bool = False
    note: str = ""

    @property
    def met(self) -> bool | None:
        target = self.bar if self.bar is not None else self.peers
        if self.you is None or target is None:
            return None
        return self.you <= target if self.lower_better else self.you >= target


@dataclass(slots=True)
class Plan:
    level: str
    quarter_end: date
    days_left: int
    #: The last day an Alpha is likely to count towards this quarter's combined performance.
    performance_cutoff: date
    bars: list[Row] = field(default_factory=list)
    tie_breakers: list[Row] = field(default_factory=list)
    performances: dict[str, float | None] = field(default_factory=dict)
    peers_counted: int = 0
    pace: list[str] = field(default_factory=list)


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def peer_medians(board: list[dict[str, Any]], level: str) -> tuple[dict[str, float], int]:
    """Median of each figure among leaderboard consultants currently at ``level``."""
    at = [r for r in board if r.get("geniusLevel") == LEVEL_NAMES.get(level, level.upper())]
    keys = ["alphaCount", "pyramidCount", *PERFORMANCES, *TIE_BREAKERS]
    out: dict[str, float] = {}
    for key in keys:
        values = [v for r in at if (v := _num(r.get(key))) is not None]
        if values:
            out[key] = float(median(values))
    if at:
        out["best"] = float(
            median(
                max((v for k in PERFORMANCES if (v := _num(r.get(k))) is not None), default=0.0)
                for r in at
            )
        )
    return out, len(at)


def quarter_end(today: date) -> date:
    month = ((today.month - 1) // 3 + 1) * 3
    return date(today.year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)


def plan(
    current: dict[str, Any], board: list[dict[str, Any]], today: date, level: str = "Master"
) -> Plan:
    """Where you stand against ``level``. Pure."""
    index = genius.LEVELS.index(level)
    bars = {k: v[index] for k, v in genius.BARS.items()}
    peers, counted = peer_medians(board, level)
    end = quarter_end(today)
    # The quarter's last month may not reach its combined performance.
    cutoff = date(end.year, end.month, 1) - timedelta(days=1)
    performances = {label: _num(current.get(k)) for k, label in PERFORMANCES.items()}
    best = max((v for v in performances.values() if v is not None), default=None)
    days_left = max((end - today).days + 1, 1)

    signals, pyramids = _num(current.get("alphaCount")), _num(current.get("pyramidCount"))
    found = Plan(level, end, days_left, cutoff, performances=performances, peers_counted=counted)
    found.bars = [
        Row("signals", "Signals this quarter", signals, bars["signals"], peers.get("alphaCount")),
        Row(
            "pyramids",
            "Pyramids this quarter",
            pyramids,
            bars["pyramids"],
            peers.get("pyramidCount"),
        ),
        Row(
            "performance",
            "Best combined performance",
            best,
            bars["combined"],
            peers.get("best"),
            note="The best of all Alphas, selected, Power Pool and Osmosis counts.",
        ),
    ]
    found.tie_breakers = [
        Row(key, label, _num(current.get(key)), None, peers.get(key), lower_better=lower)
        for key, (label, lower) in TIE_BREAKERS.items()
    ]
    if signals is not None and signals < bars["signals"]:
        need = bars["signals"] - signals
        found.pace.append(
            f"{need:.0f} more signals in {days_left} days: {need / days_left:.1f} a day, against "
            "BRAIN's limit of 4 Alphas and 1 SuperAlpha a day."
        )
    if pyramids is not None and pyramids < bars["pyramids"]:
        need = bars["pyramids"] - pyramids
        found.pace.append(
            f"{need:.0f} more pyramids, 3 Alphas each in a new region, delay and data category: "
            f"at least {need * 3 / 2:.0f} Alphas, as one Alpha can count towards two."
        )
    if best is None or best < bars["combined"]:
        found.pace.append(
            f"Combined performance must reach {bars['combined']:.1f} on one of the four. Alphas "
            f"submitted after {cutoff:%d %b} may not count this quarter; Osmosis needs a valid "
            "allocation for 6 weeks."
        )
    return found


class MasterTracker:
    """Your Genius summary and the leaderboard, read from BRAIN at most every :data:`TTL`."""

    def __init__(self, state: Any) -> None:
        self.state = state
        self._read = float("-inf")
        self._current: dict[str, Any] = {}
        self._board: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def figures(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        async with self._lock:
            if time.monotonic() - self._read < TTL:
                return self._current, self._board
            summary = await self.state.endpoints.consultant_summary()
            self._current = ((summary.get("performance") or {}).get("current")) or {}
            board: list[dict[str, Any]] = []
            for offset in range(0, BOARD_ROWS, _PAGE):
                page = await self.state.endpoints.genius_board(_PAGE, offset)
                board += page
                if len(page) < _PAGE:
                    break
            self._board = board
            self._read = time.monotonic()
            return self._current, self._board
