"""Today's Picks: the submittable Alphas measured and judged by the Submit Gate.

For each candidate this works out what the gate needs and the vault does not hold as one
figure: its size as Power Pool counts it, whether it matches the month's Power Pool theme,
the change in your combined and Power Pool Sharpe if it joins, whether it repeats a recent
submission's dataset in the same market, and what BRAIN's checks already said. The challenge
tab measures its candidates the same way.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import structlog

from ..brain.platform import POWER_POOL_CLASS
from ..db.models import utcnow
from ..labs.fastexpr import ParseError, data_fields
from ..labs.template import program
from ..vault.store import json_list
from ..vault.yields import checks_of
from . import gate

if TYPE_CHECKING:
    from ..api.lab_tasks import TaskAlpha

log = structlog.get_logger(__name__)

WHOLE = "all"
POWER = "Power Pool"
#: How far back a submission counts as recent for the lookalike test.
LOOKALIKE_DAYS = 30
#: Datasets so many Alphas read only for a price, a size or a universe that sharing one says
#: nothing about sharing an idea.
HELPER_DATASETS = frozenset({"pv1", "univ1", "univ2", "pv149"})


@dataclass(slots=True)
class Measured:
    row: TaskAlpha
    stored: dict[str, Any]
    checks: list[dict[str, Any]]
    operators: int | None
    fields: int | None
    power_pool_correlation: float | None
    theme_match: bool | None
    whole_delta: float | None
    power_delta: float | None
    whole_before: float | None
    power_before: float | None
    lookalike_of: str | None
    datasets: frozenset[str]
    facts: gate.Facts
    judgement: gate.Judgement


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _sharpe(figures: dict[str, Any] | None) -> float | None:
    return _num((figures or {}).get("sharpe"))


async def datasets_of(state: Any, expressions: dict[str, str]) -> dict[str, frozenset[str]]:
    """Each expression's datasets, helper datasets left out."""
    fields: dict[str, list[str]] = {}
    for key, text in expressions.items():
        try:
            fields[key] = data_fields(program(text))
        except ParseError:
            fields[key] = []
    names = sorted({n for found in fields.values() for n in found})
    owners = await state.queries.sole_datasets(names) if names else {}
    return {
        key: frozenset(owners[n] for n in found if n in owners) - HELPER_DATASETS
        for key, found in fields.items()
    }


async def measure(state: Any, picked: list[TaskAlpha]) -> list[Measured]:
    """Measure each candidate for the Submit Gate and judge it."""
    from ..api.template_lab import sizes
    from ..tools import portfolio
    from . import guard

    if not picked:
        return []
    ids = [str(a.alpha_id) for a in picked]
    stored = await state.alphas.by_ids(ids)
    members = await state.alphas.submitted_members()

    sized: dict[str, dict[str, Any]] = {}
    by_market: dict[tuple[str, int], list[str]] = defaultdict(list)
    for a in picked:
        settings = a.settings or {}
        by_market[(str(settings.get("region")), int(settings.get("delay") or 0))].append(
            str(a.alpha_id)
        )
    for (region, delay), market_ids in by_market.items():
        texts = [str((stored.get(i) or {}).get("expression") or "") for i in market_ids]
        sized.update(zip(market_ids, await sizes(state, texts, region, delay), strict=True))

    source = getattr(state, "themes", None)
    month = await source.current() if source is not None else []

    scopes: dict[str, list[str]] = {
        WHOLE: [str(m["alpha_id"]) for m in members],
        POWER: [
            str(m["alpha_id"])
            for m in members
            if POWER_POOL_CLASS in json_list(m.get("classifications"))
        ],
    }
    series = await state.alphas.series([*scopes[WHOLE], *ids])
    meta = await state.alphas.by_ids(scopes[WHOLE]) | stored
    effects = await asyncio.to_thread(
        portfolio.what_if,
        series,
        meta,
        scopes,
        {i: [WHOLE, POWER] for i in ids},
        guard.RECENT_START,
    )

    since = (utcnow() - timedelta(days=LOOKALIKE_DAYS)).date().isoformat()
    recent = [m for m in members if str(m.get("date_submitted") or "")[:10] >= since]
    expressions = {i: str((stored.get(i) or {}).get("expression") or "") for i in ids}
    expressions |= {f"sub:{m['alpha_id']}": str(m.get("expression") or "") for m in recent}
    datasets = await datasets_of(state, expressions)

    out: list[Measured] = []
    for a in picked:
        alpha_id = str(a.alpha_id)
        row = stored.get(alpha_id) or {}
        kept = row.get("checks")
        checks = kept if isinstance(kept, list) else checks_of(kept)
        by_name = {c.get("name"): c for c in checks}
        stat = sized.get(alpha_id) or {}
        operators = (stat.get("operators") or [None])[-1]
        fields = (stat.get("fields") or [None])[-1]
        theme = next(
            (t for t in month if t.region == row.get("region") and t.delay == row.get("delay")),
            None,
        )
        theme_match = (
            theme.covers(row.get("region"), row.get("delay"), row.get("universe"), checks)
            if theme is not None
            else None
        )
        effect = effects.get(alpha_id, {})
        whole_before = _sharpe((effect.get(WHOLE) or {}).get("before"))
        whole_after = _sharpe((effect.get(WHOLE) or {}).get("after"))
        power_before = _sharpe((effect.get(POWER) or {}).get("before"))
        power_after = _sharpe((effect.get(POWER) or {}).get("after"))
        whole_delta = whole_after - (whole_before or 0.0) if whole_after is not None else None
        power_delta = power_after - (power_before or 0.0) if power_after is not None else None
        mine = datasets.get(alpha_id, frozenset())
        lookalike = next(
            (
                str(m["alpha_id"])
                for m in recent
                if m.get("region") == row.get("region")
                and m.get("delay") == row.get("delay")
                and mine & datasets.get(f"sub:{m['alpha_id']}", frozenset())
            ),
            None,
        )
        prod = _num((by_name.get("PROD_CORRELATION") or {}).get("value"))
        if prod is None:
            prod = _num(a.prod_correlation)
        prod_result = (by_name.get("PROD_CORRELATION") or {}).get("result")
        limits = a.limits or {}
        facts = gate.Facts(
            alpha_id=alpha_id,
            sharpe=_num(a.sharpe),
            fitness=_num(a.fitness),
            sharpe_limit=_num(limits.get("LOW_SHARPE")),
            fitness_limit=_num(limits.get("LOW_FITNESS")),
            margin=_num(a.margin),
            returns=_num(a.returns),
            drawdown=_num(a.drawdown),
            train_sharpe=_num(row.get("train_sharpe")),
            test_sharpe=_num(a.test_sharpe),
            prod_correlation=prod,
            self_correlation=_num((by_name.get("SELF_CORRELATION") or {}).get("value")),
            power_pool_correlation=_num((by_name.get("POWER_POOL_CORRELATION") or {}).get("value")),
            checked=prod_result in ("PASS", "FAIL"),
            brain_fails=[
                str(c.get("name")) for c in checks if c.get("result") in ("FAIL", "ERROR")
            ],
            reversion=(by_name.get("REVERSION_COMPONENT") or {}).get("result")
            in ("WARNING", "FAIL"),
            whole_delta=whole_delta,
            power_delta=power_delta,
            lookalike_of=lookalike,
            operators=operators,
            fields=fields,
            theme_match=theme_match,
        )
        out.append(
            Measured(
                row=a,
                stored=row,
                checks=checks,
                operators=operators,
                fields=fields,
                power_pool_correlation=facts.power_pool_correlation,
                theme_match=theme_match,
                whole_delta=whole_delta,
                power_delta=power_delta,
                whole_before=whole_before,
                power_before=power_before,
                lookalike_of=lookalike,
                datasets=mine,
                facts=facts,
                judgement=gate.judge(facts),
            )
        )
    return out


#: How many candidates are measured for Today's Picks: the depth the pool effect allows
#: in one frame without slowing the screen.
DEPTH = 40


def shortlist(rows: list[TaskAlpha]) -> list[TaskAlpha]:
    """The candidates worth measuring: each Alpha once, unsubmitted, strongest first.

    Ones BRAIN has measured come first (their verdict can be READY today), then by Fitness.
    """
    seen: set[str] = set()
    out: list[TaskAlpha] = []
    for a in rows:
        if not a.alpha_id or a.alpha_id in seen or a.submitted:
            continue
        seen.add(a.alpha_id)
        out.append(a)
    out.sort(key=lambda a: (a.prod_correlation is None, -(a.fitness or 0.0)))
    return out[:DEPTH]


__all__ = ["DEPTH", "Measured", "datasets_of", "measure", "shortlist"]
