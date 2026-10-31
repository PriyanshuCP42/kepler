"""The Pre-screen agent: trains the model on the vault, keeps it fresh, and scores candidates.

See :mod:`alpha_harness.labs.prescreen` for the model. This owns its life: it loads the last
model from the data folder on start, retrains in the background when there is none or it is
older than :data:`RETRAIN_HOURS`, and hands the scheduler a :class:`Gate` only while the
model's measured lift is trusted. Without a gate, Template and Search Labs send exactly what
they always did.

It learns to predict a Critic pass, which is rarer and stricter than passing BRAIN's checks,
and falls back to BRAIN's checks when the Critic's target is too thin to filter on.

The gate draws :data:`OVERSAMPLE` candidates for every one it sends, sends the best by score,
and keeps :data:`EXPLORE` of each round for candidates chosen at random from the rest, so the
model keeps seeing outcomes it did not choose and cannot only confirm itself.
"""

import asyncio
import contextlib
import json
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import structlog

from ..brain.filters import platform_midnight
from ..labs import prescreen
from ..vault.yields import checks_of, verdict
from . import critic

if TYPE_CHECKING:
    from pathlib import Path

    from ..brain.schemas import SimulationRequest
    from ..db.duck import Catalog
    from .bus import AgentBus

log = structlog.get_logger(__name__)

OVERSAMPLE = 4
EXPLORE = 0.1
RETRAIN_HOURS = 24.0
#: How often the agent checks whether the model is due for retraining.
CHECK_SECONDS = 1800.0

Status = Literal["untrained", "training", "ready", "failed"]
Target = Literal["critic", "brain"]
#: How :meth:`Screener.reason` names each target.
TARGETS: dict[Target, str] = {"critic": "the Critic", "brain": "BRAIN's checks"}


@dataclass(frozen=True, slots=True)
class Gate:
    """What the scheduler uses to choose which candidates to send."""

    screener: Screener
    oversample: int = OVERSAMPLE
    explore: float = EXPLORE

    def choose(self, requests: list[SimulationRequest], want: int) -> list[int]:
        """Indexes of the ``want`` candidates to send: the best, plus a few at random."""
        if len(requests) <= want:
            return list(range(len(requests)))
        scores = self.screener.score(requests)
        ranked = sorted(range(len(requests)), key=lambda i: scores[i], reverse=True)
        explore = min(want, round(want * self.explore))
        best = ranked[: want - explore]
        rest = ranked[want - explore :]
        return best + random.sample(rest, min(explore, len(rest)))


class Screener:
    def __init__(self, folder: Path, catalog: Catalog, bus: AgentBus) -> None:
        self.folder = folder
        self.catalog = catalog
        self.bus = bus
        self.status: Status = "untrained"
        self.report: prescreen.Report | None = None
        #: What it learnt to predict: a Critic pass, or BRAIN's checks when Critic passes are
        #: too few or too unpredictable to filter on.
        self.target: Target = "critic"
        self.trained_at: datetime | None = None
        self.error: str | None = None
        self._fields: dict[str, tuple[str | None, str | None]] = {}
        self._day = platform_midnight()
        self.drawn_today = 0
        self.kept_today = 0
        self._task: asyncio.Task[None] | None = None
        self._training: asyncio.Task[None] | None = None

    # -- life ----------------------------------------------------------------

    async def start(self) -> None:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(self._load)
        self._report_state()
        self._task = asyncio.create_task(self._run(), name="prescreen")

    async def stop(self) -> None:
        for task in (self._task, self._training):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def _run(self) -> None:
        while True:
            age = (
                (datetime.now(UTC) - self.trained_at).total_seconds() / 3600
                if self.trained_at
                else None
            )
            if age is None or age >= RETRAIN_HOURS:
                await self.retrain()
            await asyncio.sleep(CHECK_SECONDS)

    def retrain_in_background(self) -> bool:
        """Start a retrain unless one is running. Returns whether it started."""
        if self._training is not None and not self._training.done():
            return False
        self._training = asyncio.create_task(self.retrain(), name="prescreen-train")
        return True

    async def retrain(self) -> None:
        self.status = "training"
        self.bus.say("prescreen", "Learning from every Alpha in the vault.")
        started = time.monotonic()
        try:
            fields = await self._field_map()
            rows = await self._labelled(fields)
            target: Target = "critic"
            report = await asyncio.to_thread(prescreen.train, [(n, c) for n, c, _ in rows])
            if not report.trusted:
                target = "brain"
                report = await asyncio.to_thread(prescreen.train, [(n, b) for n, _, b in rows])
        except Exception as exc:
            log.warning("prescreen.train_failed", exc_info=True)
            self.status, self.error = "failed", str(exc)[:300]
            self.bus.block("prescreen", f"Training failed: {self.error}")
            return
        self._fields, self.report, self.target = fields, report, target
        self.trained_at, self.status, self.error = datetime.now(UTC), "ready", None
        with contextlib.suppress(Exception):
            await asyncio.to_thread(self._save)
        log.info(
            "prescreen.trained",
            samples=report.samples,
            positives=report.positives,
            target=target,
            auc=report.auc,
            lift=report.lift,
            seconds=round(time.monotonic() - started, 1),
        )
        self.bus.block("prescreen", None)
        self._report_state()

    # -- the gate ------------------------------------------------------------

    def gate(self) -> Gate | None:
        """A gate while the model is trusted; ``None`` sends what the labs draw, unfiltered."""
        if self.report is None or self.report.model is None or not self.report.trusted:
            return None
        return Gate(self)

    def score(self, requests: list[SimulationRequest]) -> list[float]:
        model = self.report.model if self.report else None
        if model is None:
            return [0.0] * len(requests)
        out: list[float] = []
        for request in requests:
            settings = request.settings.model_dump(exclude_none=True)
            names = prescreen.features(request.regular, settings, self._fields)
            out.append(model.score(prescreen.hashed(names)))
        return out

    def record(self, drawn: int, kept: int) -> None:
        """Count one round of a lab's candidates, and show it in Research HQ."""
        today = platform_midnight()
        if today != self._day:
            self._day, self.drawn_today, self.kept_today = today, 0, 0
        self.drawn_today += drawn
        self.kept_today += kept
        if drawn:
            self.bus.handoff("writers", "prescreen", "candidates", drawn, "candidates written")
        self.bus.set_stat("prescreen", "screened today", self.drawn_today)
        self.bus.set_stat("prescreen", "sent today", self.kept_today)

    # -- reporting -----------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.gate() is not None

    def reason(self) -> str:
        if self.status == "training":
            return "Training on the vault now."
        if self.status == "failed":
            return f"Training failed: {self.error}"
        report = self.report
        if report is None:
            return "Not trained yet. It trains on the vault in the background."
        if report.lift is None:
            return "Too few judged Alphas in the vault to measure it yet; sending unfiltered."
        if report.trusted:
            return (
                f"Filtering Template and Search Lab sends: its top tenth passed "
                f"{TARGETS[self.target]} {report.lift:.1f}x as often as average on Alphas it "
                "never saw."
            )
        return (
            f"Watching, not filtering: its lift of {report.lift:.2f}x on held-out Alphas is "
            f"under {prescreen.MIN_LIFT:g}x."
        )

    def summary(self) -> dict[str, Any]:
        report = self.report
        return {
            "status": self.status,
            "active": self.active,
            "reason": self.reason(),
            "trainedAt": self.trained_at.isoformat() if self.trained_at else None,
            "samples": report.samples if report else 0,
            "positives": report.positives if report else 0,
            "auc": report.auc if report else None,
            "lift": report.lift if report else None,
            "target": self.target,
            "oversample": OVERSAMPLE,
            "exploreShare": EXPLORE,
            "screenedToday": self.drawn_today,
            "keptToday": self.kept_today,
        }

    def _report_state(self) -> None:
        report = self.report
        if report is not None and report.lift is not None:
            self.bus.set_stat("prescreen", "lift", f"{report.lift:.2f}x")
        if report is not None and report.auc is not None:
            self.bus.set_stat("prescreen", "AUC", round(report.auc, 3))
        self.bus.say("prescreen", self.reason())

    # -- data ----------------------------------------------------------------

    async def _field_map(self) -> dict[str, tuple[str | None, str | None]]:
        rows = await self.catalog.query(
            "SELECT field_id, any_value(dataset_id) AS d, any_value(category_id) AS c "
            "FROM data_field GROUP BY field_id"
        )
        return {str(r["field_id"]): (r["d"], r["c"]) for r in rows}

    async def _labelled(
        self, fields: dict[str, tuple[str | None, str | None]]
    ) -> list[tuple[list[str], bool, bool]]:
        """Each judged Alpha's features, whether the Critic passes it, and whether BRAIN's
        checks do, oldest first.

        The Critic's verdict is worked out here from the row rather than read back, so it is
        before the pool check: that depends on what was submitted since, not on the idea.
        """
        rows = await self.catalog.query(
            f"SELECT expression, neutralization, decay, truncation, {critic.COLUMNS} "  # noqa: S608
            "FROM alpha WHERE expression IS NOT NULL ORDER BY date_created"
        )

        def build() -> list[tuple[list[str], bool, bool]]:
            out: list[tuple[list[str], bool, bool]] = []
            for row in rows:
                judged = verdict(checks_of(row["checks"]), row["simulation_mode"])
                if judged is None:
                    continue
                settings = {k: row[k] for k in ("region", "delay", "universe", "neutralization")}
                settings |= {"decay": row["decay"], "truncation": row["truncation"]}
                names = prescreen.features(row["expression"], settings, fields)
                passed = critic.judge(row).verdict == "pass"
                out.append((names, passed, judged in ("submittable", "pending")))
            return out

        return await asyncio.to_thread(build)

    def _save(self) -> None:
        report = self.report
        if report is None or report.model is None or self.trained_at is None:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.folder / "model.npz", weights=report.model.weights)
        meta = {
            "bias": report.model.bias,
            "samples": report.samples,
            "positives": report.positives,
            "auc": report.auc,
            "lift": report.lift,
            "trainedAt": self.trained_at.isoformat(),
            "target": self.target,
            "fields": self._fields,
        }
        (self.folder / "model.json").write_text(json.dumps(meta))

    def _load(self) -> None:
        meta = json.loads((self.folder / "model.json").read_text())
        weights = np.load(self.folder / "model.npz")["weights"]
        if weights.shape != (prescreen.DIMENSIONS,):
            return
        model = prescreen.Model(weights, float(meta["bias"]))
        self.report = prescreen.Report(
            meta["samples"], meta["positives"], meta["auc"], meta["lift"], model
        )
        self._fields = {k: (v[0], v[1]) for k, v in meta["fields"].items()}
        self.trained_at = datetime.fromisoformat(meta["trainedAt"])
        self.target = "critic" if meta.get("target") == "critic" else "brain"
        self.status = "ready"
