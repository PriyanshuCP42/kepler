"""Recipes: template structures for each field shape, and the market rules each region needs.

Every recipe is a structure that earned its place on this account, written once as a template
with ``$f`` for the field and small variables for what is searched. None has a reversion term
(BRAIN warns it may stop accepting those), and each stays within Power Pool's limits of 8
operators and 3 fields (``ts_backfill`` and grouping fields are not counted), so a winner can be
a regular Alpha and a Power Pool one at once.

The engine picks among the recipes for a shape by how often each has reached the Submit Gate,
so a recipe that keeps failing in a region stops getting quota there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .doctor import Shape

#: The shared core: today's reading against the stock's own year and three years, with what
#: the stock's one-day change in size already explains taken out.
_OWN_HISTORY = (
    "vector_neut(ts_rank(ts_rank(ts_backfill($f, 60), 120), 750), rank(ts_delta(cap, 1)))"
)


@dataclass(frozen=True, slots=True)
class Recipe:
    key: str
    title: str
    shapes: frozenset[Shape]
    template: str
    #: Values searched, by variable name without the ``$``.
    values: dict[str, str] = field(default_factory=dict)
    #: The Decay setting, when the template does not smooth it itself.
    decay: int = 0
    #: Why it should work, for the task and for the Playbook.
    idea: str = ""


RECIPES: tuple[Recipe, ...] = (
    Recipe(
        key="own-history-neutral",
        title="Own-history extreme, size-move and industry neutral",
        shapes=frozenset({Shape.COUNT, Shape.LEVEL, Shape.PROBABILITY}),
        template=f"$s * ts_decay_linear(group_neutralize({_OWN_HISTORY}, industry), $d)",
        values={"s": "1, -1", "d": "8, 14"},
        idea=(
            "A reading at an extreme of the stock's own history carries news the price has not "
            "absorbed; removing the size move keeps it from being a reversal bet."
        ),
    ),
    Recipe(
        key="own-history-ranked",
        title="Own-history extreme, size-move neutral, ranked within industry",
        shapes=frozenset({Shape.COUNT, Shape.LEVEL}),
        template=f"$s * group_rank({_OWN_HISTORY}, industry)",
        values={"s": "1, -1"},
        decay=8,
        idea="The same extreme, ranked among industry peers so no few names dominate.",
    ),
    Recipe(
        key="estimate-revision",
        title="Estimate revisions, ranked within industry",
        shapes=frozenset({Shape.ESTIMATE}),
        template=(
            "$s * ts_decay_linear(group_rank(ts_delta(ts_backfill($f, 63), $w), industry), $d)"
        ),
        values={"s": "1, -1", "w": "21, 63", "d": "5, 10"},
        idea="Analysts revise in steps; prices follow revisions with a lag.",
    ),
    Recipe(
        key="estimate-own-history",
        title="Estimate against its own year, industry neutral",
        shapes=frozenset({Shape.ESTIMATE, Shape.LEVEL}),
        template=(
            "$s * ts_decay_linear("
            "group_neutralize(ts_rank(ts_backfill($f, 63), 252), industry), $d)"
        ),
        values={"s": "1, -1", "d": "5, 14"},
        idea="Where a slow-moving figure sits in its own year says how much has changed.",
    ),
    Recipe(
        key="surprise-drift",
        title="Surprise ranked within industry",
        shapes=frozenset({Shape.SURPRISE}),
        template="$s * ts_decay_linear(group_rank(ts_backfill($f, 63), industry), $d)",
        values={"s": "1, -1", "d": "5, 14"},
        idea="Prices keep drifting after a surprise, the more so the larger it is.",
    ),
    Recipe(
        key="probability-smoothed",
        title="Smoothed probability, ranked within industry",
        shapes=frozenset({Shape.PROBABILITY}),
        template="$s * ts_decay_linear(group_rank(ts_mean(ts_backfill($f, 21), 5), industry), $d)",
        values={"s": "1, -1", "d": "5, 10"},
        idea="A model's probability is noisy day to day; a week's average carries its view.",
    ),
)

BY_KEY = {r.key: r for r in RECIPES}


def for_shape(shape: Shape) -> list[Recipe]:
    return [r for r in RECIPES if shape in r.shapes]


@dataclass(frozen=True, slots=True)
class Market:
    """Where and how a region is searched."""

    universes: tuple[str, ...]
    neutralizations: tuple[str, ...] = ("STATISTICAL", "SLOW")
    investability: tuple[Literal["none", "max_trade", "max_position"], ...] = ("none",)
    truncation: float = 0.05


#: Each region's rules, learned on this account:
#: - USA: TOP3000 with STATISTICAL kept production correlation lowest.
#: - ASI: MINVOL1M with Max Trade on, since its news Alphas otherwise failed BRAIN's
#:   investability check.
#: - EUR: its theme asks for CROWDING or SLOW_AND_FAST.
MARKETS: dict[str, Market] = {
    "USA": Market(universes=("TOP3000",), truncation=0.08),
    "GLB": Market(universes=("TOP3000",)),
    "ASI": Market(universes=("MINVOL1M",), investability=("max_trade",)),
    "EUR": Market(universes=("TOP2500",), neutralizations=("CROWDING", "SLOW_AND_FAST")),
}


def market_of(region: str) -> Market | None:
    return MARKETS.get(region)


__all__ = ["BY_KEY", "MARKETS", "RECIPES", "Market", "Recipe", "for_shape", "market_of"]
