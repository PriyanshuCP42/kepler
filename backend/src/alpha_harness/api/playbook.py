"""The Playbook: WorldQuant's own guidance and this account's lessons, searchable on any screen."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from ..agents.playbook import Insight
from ..schemas import Out
from .deps import State, refuse

router = APIRouter(prefix="/api/playbook", tags=["playbook"])


class InsightIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=2000)
    source: str = Field(min_length=1, max_length=300)
    date: str = ""
    topics: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    points: list[str] = Field(default_factory=list)
    figures: list[list[str]] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)


class InsightOut(Out):
    id: str
    title: str
    summary: str
    source: str
    date: str
    topics: list[str]
    places: list[str]
    points: list[str]
    figures: list[list[str]]
    actions: list[str]
    #: BM25 relevance to the question; absent when the list was not searched.
    score: float | None = None


class PlaybookView(Out):
    insights: list[InsightOut]
    topics: list[str]
    places: list[str]
    total: int
    #: Where the insights are kept, on this machine only.
    path: str


@router.get("")
async def search(
    state: State,
    q: str = "",
    topic: str | None = None,
    place: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> PlaybookView:
    """Insights for a question (BM25), a topic or a screen; with none, the newest first."""
    book = state.playbook
    found = book.search(q, topic=topic, place=place, limit=limit)
    return PlaybookView(
        insights=[InsightOut.model_validate(asdict(i) | {"score": s}) for i, s in found],
        topics=book.topics(),
        places=book.places(),
        total=len(book.all()),
        path=str(book.path),
    )


@router.put("/{insight_id}")
async def save(insight_id: str, body: InsightIn, state: State) -> InsightOut:
    """Add an insight, or replace the one with this id."""
    insight = Insight(id=insight_id, **body.model_dump())
    state.playbook.upsert(insight)
    return InsightOut.model_validate(asdict(insight))


@router.delete("/{insight_id}", status_code=204)
async def remove(insight_id: str, state: State) -> None:
    if not state.playbook.delete(insight_id):
        raise refuse(404, "not_found", f"No insight {insight_id}.")
