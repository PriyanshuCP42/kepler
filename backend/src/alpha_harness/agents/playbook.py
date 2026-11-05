"""The Playbook: what WorldQuant has told consultants, and what this account has measured.

WorldQuant shares how it judges and combines Alphas in webinars, FAQ articles and competition
rules; a consultant who acts on it stands out from those who never read it. Each **insight** is
one such lesson: a short summary, the points behind it, the figures it rests on, what to do
about it, where it came from, and the topics and Kepler screens it belongs to.

Insights live in ``playbook.json`` in Kepler's data folder, never in the repository: the user
adds them from webinars and research, and they stay on this machine.

**Retrieval** is lexical, BM25 over each insight's words, so it costs nothing to run and needs
no model: a question finds the insights that share its terms, weighted by how rare those terms
are across the playbook. A screen asks for the insights tagged with its place (``genius``,
``desk``, ``labs``...), most recent first.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

FILENAME = "playbook.json"
#: BM25's usual constants: term-frequency saturation and length normalisation.
K1, B = 1.4, 0.75
#: The title says what an insight is about, so its words count this many times.
TITLE_WEIGHT = 3
#: Shorthand a consultant types, and the words the insights use for it.
SYNONYMS = {
    "tvr": "turnover",
    "pp": "power pool",
    "cap": "combined alpha performance",
    "corr": "correlation",
    "prod": "production",
    "oos": "out sample",
    "neut": "neutralization",
    "aqp": "alpha quality project",
    "fte": "full time",
    "job": "full time",
    "hiring": "full time",
}
_WORD = re.compile(r"[a-z0-9_]+")
_STOP = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "will",
        "with",
        "you",
        "your",
    )
)


@dataclass(slots=True)
class Insight:
    id: str
    title: str
    summary: str
    source: str
    date: str = ""
    topics: list[str] = field(default_factory=list)
    #: Kepler screens this insight belongs on.
    places: list[str] = field(default_factory=list)
    points: list[str] = field(default_factory=list)
    #: (label, value) pairs, the numbers it rests on.
    figures: list[list[str]] = field(default_factory=list)
    #: What to do about it.
    actions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Insight:
        known = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        return cls(**known)

    def words(self) -> list[str]:
        body = " ".join(
            [
                self.summary,
                *self.points,
                *(f"{a} {b}" for a, b in self.figures),
                *self.actions,
                *self.topics,
                self.source,
            ]
        )
        return tokens(self.title) * TITLE_WEIGHT + tokens(body)


def _stem(word: str) -> str:
    for suffix in ("ations", "ation", "ings", "ing", "ies", "es", "ed", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokens(text: str) -> list[str]:
    """Lower-cased word stems, stop words dropped."""
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]


def query_tokens(text: str) -> list[str]:
    """A question's stems, shorthand expanded to the words the insights use."""
    words: list[str] = _WORD.findall(text.lower())
    expanded = " ".join(SYNONYMS.get(w) or w for w in words)
    return tokens(expanded)


def rank(insights: list[Insight], query: str) -> list[tuple[Insight, float]]:
    """Insights sharing the query's terms, best BM25 score first; none when nothing matches."""
    terms = query_tokens(query)
    if not terms or not insights:
        return []
    docs = [Counter(i.words()) for i in insights]
    lengths = [sum(d.values()) for d in docs]
    mean = sum(lengths) / len(lengths)
    n = len(docs)
    scored: list[tuple[Insight, float]] = []
    for insight, doc, length in zip(insights, docs, lengths, strict=True):
        score = 0.0
        for term in set(terms):
            tf = doc.get(term, 0)
            if not tf:
                continue
            df = sum(1 for d in docs if term in d)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / mean))
        if score > 0:
            scored.append((insight, score))
    return sorted(scored, key=lambda s: s[1], reverse=True)


class Playbook:
    """The insights in ``playbook.json``, read again whenever the file changes."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / FILENAME
        self._stamp: float | None = None
        self._insights: list[Insight] = []

    def all(self) -> list[Insight]:
        try:
            stamp = self.path.stat().st_mtime
        except FileNotFoundError:
            self._stamp, self._insights = None, []
            return []
        if stamp != self._stamp:
            raw = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            self._insights = [Insight.from_dict(r) for r in raw.get("insights", [])]
            self._stamp = stamp
        return list(self._insights)

    def search(
        self,
        query: str = "",
        *,
        topic: str | None = None,
        place: str | None = None,
        limit: int = 50,
    ) -> list[tuple[Insight, float | None]]:
        """Insights for a question, a topic or a screen. With no question, the newest first."""
        pool = [
            i
            for i in self.all()
            if (topic is None or topic in i.topics) and (place is None or place in i.places)
        ]
        found: list[tuple[Insight, float | None]]
        if query.strip():
            found = [(i, round(s, 3)) for i, s in rank(pool, query)]
        else:
            found = [(i, None) for i in sorted(pool, key=lambda i: i.date, reverse=True)]
        return found[:limit]

    def topics(self) -> list[str]:
        return sorted({t for i in self.all() for t in i.topics})

    def places(self) -> list[str]:
        return sorted({p for i in self.all() for p in i.places})

    def upsert(self, insight: Insight) -> None:
        kept = [i for i in self.all() if i.id != insight.id] + [insight]
        self._write(kept)

    def delete(self, insight_id: str) -> bool:
        current = self.all()
        kept = [i for i in current if i.id != insight_id]
        if len(kept) == len(current):
            return False
        self._write(kept)
        return True

    def _write(self, insights: list[Insight]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"insights": [asdict(i) for i in insights]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)
        self._stamp = None


def notes(playbook: Playbook | None, question: str, limit: int = 3) -> str:
    """The insights that best match a question, as plain lines for a prompt; empty if none."""
    if playbook is None:
        return ""
    found = playbook.search(question, limit=limit)
    lines = []
    for insight, _ in found:
        lines.append(f"- {insight.title}: {insight.summary}")
        lines += [f"  • {p}" for p in insight.points[:4]]
    return "\n".join(lines)
