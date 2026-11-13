"""The Dataset Doctor: what kind of signal each data field is, before quota is spent on it.

A generic template treats every field alike, and on this account such templates passed 0.1-5%
of the time. What a field *is* decides what can be done with it: a count of news items wants
ranking against the stock's own history, an analyst estimate wants its revisions, a surprise is
already a signal, an identifier or a timestamp is none. So each field is given a **shape** from
what the catalog says about it (category, type, description, name) and the template engine
picks recipes for that shape.

Pure: it reads catalog rows and returns shapes and plans, so every rule is tested offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Shape(StrEnum):
    #: A count or volume of text, news, posts or events: rank it against its own history.
    COUNT = "count"
    #: A score, rating, model output or sentiment level.
    LEVEL = "level"
    #: An analyst estimate or consensus that gets revised.
    ESTIMATE = "estimate"
    #: Actual against expected: already a signal.
    SURPRISE = "surprise"
    #: A probability or likelihood between 0 and 1.
    PROBABILITY = "probability"
    #: Not a signal: an identifier, a timestamp, a flag, a name.
    SKIP = "skip"


#: Words in a field's name or description that mark it as something to leave alone.
_SKIP = re.compile(
    r"\b(identifier|isin|cusip|sedol|gvkey|ticker|permno|timestamp|epoch|utc|url|"
    r"currency code|exchange code|flag|announce time)\b|(?<![-_])\bdate\b(?![-_])|"
    r"\btime\b(?![- ]series)|"
    r"\bname of\b|^\w*(_isin|_gvkey|_ticker|_time|_date|timestamp|receivetime|_name)$",
    re.IGNORECASE,
)
_SURPRISE = re.compile(
    r"surprise|actual[ _](?:vs|versus|minus|against)|\bbeats?\b|\bmiss(?:es|ed)?\b",
    re.IGNORECASE,
)
_PROBABILITY = re.compile(r"probability|likelihood|\bprob\b|_prob\b|\bprob_", re.IGNORECASE)
_COUNT = re.compile(
    r"\b(count|number of|mentions?|articles?|posts?|tweets?|sentences?|words?|stories|"
    r"news items?|occurrences?)\b|_count\b|^count_",
    re.IGNORECASE,
)
#: Analyst terms strong enough to mark an estimate in any dataset. "Forecast" and "estimate"
#: alone also describe model and text scores, so they count only in analyst data.
_ESTIMATE = re.compile(
    r"consensus|revision|revised|guidance|target price|\beps\b|\bebitda\b|"
    r"\bfy[12]\b|\bfq[12]\b",
    re.IGNORECASE,
)
_ESTIMATE_WORDS = re.compile(r"estimate|forecast", re.IGNORECASE)
#: Categories whose levels are text or crowd measures, so a count of them behaves like one.
_TEXT_CATEGORIES = frozenset({"news", "sentiment", "socialmedia"})


def classify(row: dict[str, Any]) -> Shape:
    """A catalog field's shape, from its name, description, category and type."""
    name = str(row.get("field_id") or "")
    text = f"{name} {row.get('description') or ''}"
    category = str(row.get("category_id") or "")
    if _SKIP.search(text):
        return Shape.SKIP
    if _SURPRISE.search(text):
        return Shape.SURPRISE
    if _PROBABILITY.search(text):
        return Shape.PROBABILITY
    if _COUNT.search(text) or (category in _TEXT_CATEGORIES and "total" in text.lower()):
        return Shape.COUNT
    if _ESTIMATE.search(text) or (category == "analyst" and _ESTIMATE_WORDS.search(text)):
        return Shape.ESTIMATE
    if category == "analyst":
        return Shape.ESTIMATE
    return Shape.LEVEL


@dataclass(slots=True)
class Candidate:
    """One dataset's fields of one shape, ready for a recipe."""

    dataset_id: str
    category_id: str
    shape: Shape
    field_ids: list[str]
    #: How much a recipe here is worth trying: value, multiplier, coverage, how few use it.
    priority: float
    notes: list[str] = field(default_factory=list)


#: Fields per candidate: enough for a search to choose among, few enough to stay focused.
MAX_FIELDS = 30
#: Coverage below this cannot carry a signal across the universe.
MIN_COVERAGE = 0.6
#: Alphas per field above this count as crowded.
CROWDED_FIELD = 20
#: Datasets never worth a recipe here: the theme excludes one, others are crowded helpers.
EXCLUDED = frozenset({"techindi_model", "pv1", "univ1", "univ2", "pv149"})


def candidates(
    datasets: list[dict[str, Any]],
    fields: list[dict[str, Any]],
    *,
    avoid: frozenset[str] = frozenset(),
) -> list[Candidate]:
    """Each dataset's usable fields grouped by shape, most promising first.

    ``avoid`` names datasets to leave out: ones you already submitted, or ones whose Alphas
    came back crowded.
    """
    by_id = {str(d.get("dataset_id")): d for d in datasets}
    grouped: dict[tuple[str, Shape], list[dict[str, Any]]] = {}
    for f in fields:
        dataset = str(f.get("dataset_id") or "")
        if dataset in EXCLUDED or dataset in avoid or dataset not in by_id:
            continue
        if (f.get("coverage") or 0) < MIN_COVERAGE:
            continue
        if (f.get("alpha_count") or 0) > CROWDED_FIELD:
            continue
        shape = classify(f)
        if shape is Shape.SKIP:
            continue
        grouped.setdefault((dataset, shape), []).append(f)

    out: list[Candidate] = []
    for (dataset, shape), rows in grouped.items():
        d = by_id[dataset]
        rows.sort(key=lambda r: (r.get("alpha_count") or 0, -(r.get("coverage") or 0)))
        chosen = rows[:MAX_FIELDS]
        unused = sum(1 for r in chosen if not r.get("alpha_count"))
        coverage = sum(float(r.get("coverage") or 0) for r in chosen) / len(chosen)
        value = float(d.get("value_score") or 1.0)
        multiplier = float(d.get("pyramid_multiplier") or 1.0)
        priority = value * multiplier * coverage * (1 + unused / len(chosen))
        out.append(
            Candidate(
                dataset_id=dataset,
                category_id=str(d.get("category_id") or ""),
                shape=shape,
                field_ids=[str(r["field_id"]) for r in chosen],
                priority=priority,
                notes=[f"{len(chosen)} fields", f"{unused} never used"],
            )
        )
    out.sort(key=lambda c: -c.priority)
    return out


__all__ = ["Candidate", "Shape", "candidates", "classify"]
