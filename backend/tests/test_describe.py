"""Writing Power Pool descriptions for the Submission Desk's candidates, in bulk."""

import asyncio
from types import SimpleNamespace

from alpha_harness.api import alphas, desk


def _candidate(alpha_id: str, readiness: str, power: bool, status: str) -> SimpleNamespace:
    return SimpleNamespace(
        alpha_id=alpha_id,
        readiness=readiness,
        power_pool=power,
        submission=SimpleNamespace(status=status),
    )


def test_needs_writing_picks_power_pool_candidates_brain_has_not_passed():
    assert desk.needs_writing(_candidate("a", "fails_check", True, "fails"))
    assert desk.needs_writing(_candidate("b", "check", True, "not_run"))
    assert not desk.needs_writing(_candidate("c", "check", False, "not_run"))
    assert not desk.needs_writing(_candidate("d", "ready", True, "passes"))
    assert not desk.needs_writing(_candidate("e", "blocked", True, "fails"))


def test_the_job_keeps_long_descriptions_and_properties(monkeypatch):
    bodies = {
        "new": {"name": "Mine", "category": "RISK", "color": "BLUE", "tags": ["x"], "regular": {}},
        "old": {"regular": {"description": "y" * 150}},
    }
    saved: dict[str, object] = {}
    checked: list[str] = []

    async def fake_desk(state):
        return SimpleNamespace(
            candidates=[
                _candidate("new", "fails_check", True, "fails"),
                _candidate("old", "check", True, "not_run"),
            ]
        )

    async def alpha_body(alpha_id):
        return bodies[alpha_id]

    async def write_description(alpha_id, state):
        return SimpleNamespace(description="z" * 300)

    async def update_properties(alpha_id, props, state):
        saved[alpha_id] = props

    async def check_alpha(alpha_id, state):
        checked.append(alpha_id)
        result = "PASS" if alpha_id == "new" else "FAIL"
        return alphas.BrainPayload({"is": {"checks": [{"name": "LOW_SHARPE", "result": result}]}})

    captured = {}

    async def start_job(kind, label, work):
        captured["work"] = work
        return "job-1"

    async def update(task, **kwargs):
        return None

    monkeypatch.setattr(desk, "desk", fake_desk)
    monkeypatch.setattr(alphas, "write_description", write_description)
    monkeypatch.setattr(alphas, "update_properties", update_properties)
    monkeypatch.setattr(alphas, "check_alpha", check_alpha)
    state = SimpleNamespace(
        endpoints=SimpleNamespace(alpha_body=alpha_body),
        backfill=SimpleNamespace(start_job=start_job),
        tasks=SimpleNamespace(update=update),
    )

    started = asyncio.run(desk.write_descriptions(state))
    assert started.task_id == "job-1"
    summary = asyncio.run(captured["work"](object()))

    assert list(saved) == ["new"]  # the long description is kept
    props = saved["new"]
    assert (props.name, props.category, props.color, props.tags) == ("Mine", "RISK", "BLUE", ["x"])
    assert props.description == "z" * 300
    assert checked == ["new", "old"]
    assert "1 descriptions written" in summary
    assert "1 now pass" in summary
    assert "1 already described" in summary
