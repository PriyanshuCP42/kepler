"""Competitions: what BRAIN is running, and where you stand in each.

BRAIN lists every competition with your status in it, but your rank only comes with one
competition at a time, so only the ongoing ones are read in full.
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter
from pydantic import Field

from ..agents import themes as theme_rules
from ..brain.platform import POWER_POOL_CLASS
from ..schemas import Out
from ..vault.store import json_list
from ..vault.yields import checks_of
from .deps import State

router = APIRouter(prefix="/api/competitions", tags=["competitions"])
log = structlog.get_logger(__name__)


class CompetitionStanding(Out):
    rank: int | None
    alphas: int | None


class Entry(Out):
    """One of your submitted Alphas in a competition or theme."""

    alpha_id: str
    region: str | None
    delay: int | None
    universe: str | None
    sharpe: float | None
    fitness: float | None
    date_submitted: str | None
    #: For a theme: whether it meets every condition the theme asks; the checks it misses.
    matches: bool | None = None
    missing: list[str] = Field(default_factory=list)


class Competition(Out):
    id: str
    name: str
    #: BRAIN's word for your place in it, e.g. ``ACCEPTED`` or ``EXCLUDED``.
    status: str
    enrolled: bool
    ongoing: bool
    scoring: str | None
    start_date: str | None
    end_date: str | None
    sign_up_end_date: str | None
    faq: str | None
    #: Only for an ongoing competition you are enrolled in.
    standing: CompetitionStanding | None
    #: Your submitted Alphas tagged with its id (an Alpha Quality Project cycle's tag).
    entries: list[Entry] = Field(default_factory=list)


class ThemeTab(Out):
    """A Power Pool theme: its market, what it asks, and your tagged Power Pool Alphas in it."""

    id: str
    name: str
    multiplier: float
    region: str | None
    delay: int | None
    qualifier: str | None
    requires: list[str]
    start_date: str | None
    end_date: str | None
    min_for_awards: int
    #: Power Pool Alphas tagged PowerPoolSelected in the theme's market, matching or not.
    entries: list[Entry]


class Competitions(Out):
    competitions: list[Competition]
    themes: list[ThemeTab] = Field(default_factory=list)


def _entry(row: dict[str, Any], **extra: Any) -> Entry:
    return Entry(
        alpha_id=str(row["alpha_id"]),
        region=row.get("region"),
        delay=row.get("delay"),
        universe=row.get("universe"),
        sharpe=row.get("sharpe"),
        fitness=row.get("fitness"),
        date_submitted=str(row["date_submitted"]) if row.get("date_submitted") else None,
        **extra,
    )


async def _themes(state: State, members: list[dict[str, Any]]) -> list[ThemeTab]:
    source = getattr(state, "themes", None)
    found = await source.current() if source is not None else []
    if not found:
        return []
    tagged = [
        m
        for m in members
        if POWER_POOL_CLASS in json_list(m.get("classifications"))
        and "PowerPoolSelected" in json_list(m.get("tags"))
    ]
    stored = await state.alphas.by_ids([str(m["alpha_id"]) for m in tagged])
    tabs: list[ThemeTab] = []
    for theme in found:
        entries: list[Entry] = []
        for m in tagged:
            if m.get("region") != theme.region or m.get("delay") != theme.delay:
                continue
            checks = checks_of((stored.get(str(m["alpha_id"])) or {}).get("checks"))
            entries.append(
                _entry(
                    m,
                    matches=theme.covers(
                        m.get("region"), m.get("delay"), m.get("universe"), checks
                    ),
                    missing=theme_rules.missed(checks, theme.requires),
                )
            )
        window = theme_rules.WINDOWS.get(theme.name)
        tabs.append(
            ThemeTab(
                id=theme.id,
                name=theme.name,
                multiplier=theme.multiplier,
                region=theme.region,
                delay=theme.delay,
                qualifier=theme.qualifier,
                requires=list(theme.requires),
                start_date=window[0] if window else None,
                end_date=window[1] if window else None,
                min_for_awards=theme_rules.MIN_FOR_AWARDS,
                entries=sorted(entries, key=lambda e: (not e.matches, -(e.sharpe or 0))),
            )
        )
    return tabs


def _when(value: Any) -> datetime | None:
    """A BRAIN timestamp as an aware datetime. BRAIN sends an offset; one without is read as
    UTC rather than left naive, which could not be compared with ``now``."""
    if not value:
        return None
    try:
        at = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return at if at.tzinfo is not None else at.replace(tzinfo=UTC)


def _is_ongoing(c: dict[str, Any], now: datetime) -> bool:
    start, end = _when(c.get("startDate")), _when(c.get("endDate"))
    return start is not None and end is not None and start <= now <= end


@router.get("")
async def competitions(state: State) -> Competitions:
    listed = await state.endpoints.competitions()
    now = datetime.now(UTC)
    ongoing = [c for c in listed if _is_ongoing(c, now)]
    # One competition BRAIN will not show (restricted, or a passing error) costs only its rank,
    # not the page.
    details = await asyncio.gather(
        *(state.endpoints.competition(str(c["id"])) for c in ongoing), return_exceptions=True
    )
    boards: dict[str, Any] = {}
    for c, d in zip(ongoing, details, strict=True):
        if isinstance(d, BaseException):
            log.warning("competitions.detail_failed", competition=c.get("id"), error=str(d))
        else:
            boards[str(d.get("id"))] = d.get("leaderboard")

    members = await state.alphas.submitted_members()

    def tagged(cid: str) -> list[Entry]:
        wanted = cid.upper()
        return [
            _entry(m) for m in members if wanted in {t.upper() for t in json_list(m.get("tags"))}
        ]

    def row(c: dict[str, Any]) -> Competition:
        cid = str(c["id"])
        board = boards.get(cid)
        return Competition(
            id=cid,
            name=str(c.get("name") or cid),
            status=str(c.get("status") or ""),
            enrolled=c.get("signUpDate") is not None,
            ongoing=_is_ongoing(c, now),
            scoring=c.get("scoring"),
            start_date=c.get("startDate"),
            end_date=c.get("endDate"),
            sign_up_end_date=c.get("signUpEndDate"),
            faq=c.get("faq"),
            standing=CompetitionStanding(rank=board.get("rank"), alphas=board.get("alphas"))
            if isinstance(board, dict)
            else None,
            entries=tagged(cid) if _is_ongoing(c, now) else [],
        )

    rows = [row(c) for c in listed]

    # Ongoing first, soonest to end; then the rest, most recent first. By the moment, not the
    # text: BRAIN's offsets change with daylight saving, and text sorts ignore them.
    def ends(r: Competition) -> datetime:
        return _when(r.end_date) or datetime.min.replace(tzinfo=UTC)

    live = sorted((r for r in rows if r.ongoing), key=ends)
    past = sorted((r for r in rows if not r.ongoing), key=ends, reverse=True)
    return Competitions(competitions=live + past, themes=await _themes(state, members))
