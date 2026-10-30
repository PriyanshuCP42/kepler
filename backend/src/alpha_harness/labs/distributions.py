"""A trial's Optuna distributions, each large one stored once and referenced by every trial.

A trial records what Optuna drew each value from, so its sampler can be rebuilt after a
restart. A categorical over a few thousand fields or expressions is the same JSON on every
trial of a task: 62,000 trials once took 16 GB to store a few hundred distinct values. So a
distribution longer than :data:`INLINE_LIMIT` is kept in ``trial_distribution`` under its
SHA-256 and the trial stores ``ref:<digest>`` in its place. Short ones stay inline, as does
``scheduler.FROM_SPACE``.

:func:`pack` turns a trial's distributions into the stored form, :func:`resolve` reads the
bodies a set of stored forms refer to, and :func:`unpack` puts them back.
"""

import hashlib
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert

from ..db.models import TrialDistribution

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

#: A stored distribution longer than this, in characters, is kept once and referenced.
INLINE_LIMIT = 512
#: Marks a reference. Stored distributions are JSON objects, so none begins with it.
REF = "ref:"


def digest(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def pack(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """The stored form of ``raw``, and the bodies it now refers to, by digest."""
    stored: dict[str, Any] = {}
    bodies: dict[str, str] = {}
    for name, value in raw.items():
        if isinstance(value, str) and len(value) > INLINE_LIMIT and not value.startswith(REF):
            key = digest(value)
            bodies[key] = value
            stored[name] = REF + key
        else:
            stored[name] = value
    return stored, bodies


def refs(raws: Iterable[dict[str, Any]]) -> set[str]:
    """Every digest these stored forms refer to."""
    return {
        value.removeprefix(REF)
        for raw in raws
        for value in raw.values()
        if isinstance(value, str) and value.startswith(REF)
    }


def unpack(raw: dict[str, Any], bodies: dict[str, str]) -> dict[str, Any]:
    """``raw`` with its references replaced. Empty when one names a body that is missing,
    which the readers treat as a trial they cannot replay."""
    out: dict[str, Any] = {}
    for name, value in raw.items():
        if isinstance(value, str) and value.startswith(REF):
            body = bodies.get(value.removeprefix(REF))
            if body is None:
                return {}
            out[name] = body
        else:
            out[name] = value
    return out


async def save(session: AsyncSession, bodies: dict[str, str]) -> None:
    """Store bodies not stored yet. A digest names its body, so an existing row is kept."""
    if not bodies:
        return
    await session.execute(
        insert(TrialDistribution)
        .values([{"digest": key, "body": body} for key, body in bodies.items()])
        .on_conflict_do_nothing(index_elements=[TrialDistribution.digest])
    )


async def resolve(session: AsyncSession, raws: Iterable[dict[str, Any]]) -> dict[str, str]:
    """The bodies these stored forms refer to, by digest."""
    wanted = list(refs(raws))
    found: dict[str, str] = {}
    for start in range(0, len(wanted), 500):
        rows = await session.execute(
            select(TrialDistribution.digest, TrialDistribution.body).where(
                TrialDistribution.digest.in_(wanted[start : start + 500])
            )
        )
        found.update(dict(rows.tuples().all()))
    return found
