"""The Hypothesis Writer: write hypotheses, switch them on or off, and test them as a task."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..agents.hypothesis import NICHES, PER_NICHE
from ..schemas import Out
from .deps import State, refuse

router = APIRouter(prefix="/api/hypotheses", tags=["hypotheses"])


class HypothesisOut(Out):
    id: int
    created_at: str
    region: str
    delay: int
    universe: str
    category: str
    dataset_id: str
    title: str
    rationale: str
    expressions: list[str]
    model: str | None
    enabled: bool
    #: Written for it in a task, finished so far, and passed by the Critic.
    simulations: int
    finished: int
    passed: int
    best_sharpe: float | None


class Hypotheses(Out):
    #: idle | writing | failed
    status: str
    note: str | None
    hypotheses: list[HypothesisOut]


class WriteRequest(BaseModel):
    niches: int = Field(default=NICHES, ge=1, le=6)
    per_niche: int = Field(default=PER_NICHE, ge=1, le=8, alias="perNiche")


class Toggle(BaseModel):
    enabled: bool


class TaskRequest(BaseModel):
    cores: int = Field(default=4, ge=1, le=8)


class Started(Out):
    started: bool


class HypothesisTask(Out):
    id: int
    name: str
    hypotheses: int


@router.get("")
async def hypotheses(state: State) -> Hypotheses:
    writer = state.hypotheses
    return Hypotheses(
        status=writer.status,
        note=writer.note,
        hypotheses=[HypothesisOut.model_validate(h) for h in await writer.results()],
    )


@router.post("/write", status_code=202)
async def write(body: WriteRequest, state: State) -> Started:
    """Ask the LLM for hypotheses on the Strategist's top pyramids, in the background. Uses the
    LLM's budget, no simulation quota."""
    if not state.hypotheses.write_in_background(body.niches, body.per_niche):
        raise refuse(409, "busy", "The Hypothesis Writer is already writing.")
    return Started(started=True)


@router.patch("/{hypothesis_id}")
async def toggle(hypothesis_id: int, body: Toggle, state: State) -> Toggle:
    if not await state.hypotheses.set_enabled(hypothesis_id, body.enabled):
        raise refuse(404, "not_found", f"No hypothesis {hypothesis_id}.")
    return body


@router.post("/tasks", status_code=201)
async def add_task(body: TaskRequest, state: State) -> HypothesisTask:
    """Every switched-on hypothesis not yet tested, as one task in Tasks, not started."""
    added = await state.hypotheses.add_task(body.cores)
    if added is None:
        raise refuse(409, "nothing_to_test", "No switched-on hypothesis is waiting to be tested.")
    task, count = added
    return HypothesisTask(id=task.id, name=task.name, hypotheses=count)
