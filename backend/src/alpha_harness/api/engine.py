"""The template engine: the Dataset Doctor's view of a region, the scoreboard of recipes, and
writing engine tasks."""

from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from ..labs import engine, recipes
from ..schemas import Out
from .deps import State, refuse

router = APIRouter(prefix="/api/engine", tags=["engine"])


class Diagnosis(Out):
    dataset_id: str
    category_id: str
    shape: str
    fields: int
    priority: float
    notes: list[str]
    #: The recipes the engine can use for this shape.
    recipes: list[str]


class RecipeRecord(Out):
    recipe: str
    title: str
    shape: str
    region: str
    tries: int
    passes: int
    rate: float | None
    retired: bool


class Scoreboard(Out):
    records: list[RecipeRecord]
    retire_after: int


class PlanRequest(BaseModel):
    region: str
    count: int = Field(default=3, ge=1, le=10)


class Planned(Out):
    region: str
    added: list[int]


@router.get("/doctor")
async def doctor_view(
    state: State, region: Annotated[str, Query(description="USA, GLB, ASI or EUR")]
) -> list[Diagnosis]:
    """Each promising dataset in the region and the shape of its usable fields, best first."""
    if recipes.market_of(region) is None:
        raise refuse(422, "no_market_rules", f"The engine has no rules for {region} yet.")
    found = await engine.diagnose(state, region)
    return [
        Diagnosis(
            dataset_id=c.dataset_id,
            category_id=c.category_id,
            shape=str(c.shape),
            fields=len(c.field_ids),
            priority=round(c.priority, 3),
            notes=c.notes,
            recipes=[r.key for r in recipes.for_shape(c.shape)],
        )
        for c in found[:40]
    ]


@router.get("/scoreboard")
async def scoreboard(state: State) -> Scoreboard:
    """How each recipe has done for each shape and region: trials, passes, and whether it is
    retired there."""
    learned = await engine.records(state)
    rows = [
        RecipeRecord(
            recipe=key,
            title=recipes.BY_KEY[key].title if key in recipes.BY_KEY else key,
            shape=shape,
            region=region,
            tries=r.tries,
            passes=r.passes,
            rate=r.passes / r.tries if r.tries else None,
            retired=r.retired,
        )
        for (key, shape, region), r in learned.items()
    ]
    rows.sort(key=lambda r: (-(r.rate or 0), -r.tries))
    return Scoreboard(records=rows, retire_after=engine.RETIRE_TRIES)


@router.post("/plan")
async def plan(body: PlanRequest, state: State) -> Planned:
    """Write engine tasks for the region, not started. The Daily Autopilot starts them."""
    if recipes.market_of(body.region) is None:
        raise refuse(422, "no_market_rules", f"The engine has no rules for {body.region} yet.")
    added = await engine.plan(state, body.region, body.count)
    return Planned(region=body.region, added=[a.id for a in added])
