"""The Pre-screen: which candidate Alphas are worth a simulation, learned from the vault.

Every Alpha the app has simulated is in the vault with BRAIN's verdict on it. The Pre-screen
learns from them which expressions and settings pass BRAIN's checks, then ranks a lab's
candidates before they are sent, so the day's quota goes to the likeliest ones.

The model is a logistic regression over hashed features: the settings, every operator, every
field with its dataset and category, and a few crossings (region with category, universe with
dataset). It is small, trains in seconds on tens of thousands of Alphas, and needs nothing
beyond numpy. Hashing is by CRC-32, so a model trained in one process scores the same in
another.

It is judged honestly before it is trusted: trained on the older four-fifths of the vault and
scored on the newest fifth, which it never saw. **Lift** is the pass rate of its top tenth
divided by the average pass rate there; the gate only filters sends when lift clears
:data:`MIN_LIFT`.
"""

import math
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .fastexpr import ParseError, data_fields, parse, walk

#: Hashed feature space. Collisions at this size cost almost nothing at a few thousand features.
DIMENSIONS = 1 << 18
#: Below this lift on held-out Alphas the model watches but does not filter.
MIN_LIFT = 1.3
#: Held-out share, newest first.
HOLDOUT = 0.2
EPOCHS = 4
LEARNING_RATE = 0.2
L2 = 1e-6
#: Positives are rare (about 2%); this weight keeps the model from ignoring them.
POSITIVE_WEIGHT = 5.0


def _bucket(value: float | None, edges: tuple[float, ...]) -> str:
    if value is None:
        return "none"
    for edge in edges:
        if value <= edge:
            return f"<={edge:g}"
    return f">{edges[-1]:g}"


def features(
    expression: str | None,
    settings: dict[str, Any],
    fields: dict[str, tuple[str | None, str | None]],
) -> list[str]:
    """The feature names of one candidate. ``fields`` maps a field id to its dataset and
    category, from the catalog; an unknown field still counts as itself."""
    region = str(settings.get("region") or "")
    universe = str(settings.get("universe") or "")
    out = [
        f"region={region}",
        f"delay={settings.get('delay')}",
        f"universe={universe}",
        f"neutralization={settings.get('neutralization')}",
        f"decay={_bucket(_float(settings.get('decay')), (0, 4, 10, 20))}",
        f"truncation={_bucket(_float(settings.get('truncation')), (0.01, 0.05, 0.1))}",
    ]
    if not expression:
        return [*out, "no-expression"]
    try:
        tree = parse(expression)
    except ParseError:
        return [*out, "unparsable"]
    nodes = [node for _, node in walk(tree)]
    operators = [n.value for n in nodes if n.kind in ("call", "binary", "unary", "ternary")]
    out.extend(f"op={name}" for name in dict.fromkeys(operators))
    out.append(f"operators={_bucket(float(len(operators)), (1, 2, 3, 5, 8))}")
    if tree.kind == "call":
        out.append(f"outer={tree.value}")
    numbers = [n.value for n in nodes if n.kind == "num"]
    out.extend(f"n={_bucket(_float(v), (1, 5, 10, 22, 66, 126, 252))}" for v in set(numbers))
    names = data_fields(tree)
    out.append(f"fields={min(len(names), 4)}")
    for name in names:
        dataset, category = fields.get(name, (None, None))
        out.append(f"field={name}")
        if dataset:
            out += [f"dataset={dataset}", f"{region}|dataset={dataset}", f"{universe}|{dataset}"]
        if category:
            out += [f"category={category}", f"{region}|category={category}"]
    return out


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except TypeError, ValueError:
        return None
    return number if math.isfinite(number) else None


def hashed(names: list[str]) -> np.ndarray:
    return np.array(sorted({zlib.crc32(n.encode()) % DIMENSIONS for n in names}), dtype=np.int64)


@dataclass(slots=True)
class Model:
    weights: np.ndarray
    bias: float

    def score(self, index: np.ndarray) -> float:
        """Probability-like score in (0, 1)."""
        z = self.bias + float(self.weights[index].sum())
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def fit(samples: list[np.ndarray], labels: np.ndarray, *, seed: int = 0) -> Model:
    """Logistic regression by SGD with per-weight AdaGrad steps."""
    rng = np.random.default_rng(seed)
    weights = np.zeros(DIMENSIONS)
    squares = np.full(DIMENSIONS, 1e-8)
    base = float(labels.mean()) if len(labels) else 0.5
    bias = math.log(max(base, 1e-6) / max(1 - base, 1e-6))
    bias_square = 1e-8
    order = np.arange(len(samples))
    for _ in range(EPOCHS):
        rng.shuffle(order)
        for i in order:
            index, label = samples[i], labels[i]
            z = bias + float(weights[index].sum())
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            gradient = (p - label) * (POSITIVE_WEIGHT if label else 1.0)
            squares[index] += gradient * gradient
            weights[index] -= LEARNING_RATE * (
                gradient / np.sqrt(squares[index]) + L2 * weights[index]
            )
            bias_square += gradient * gradient
            bias -= LEARNING_RATE * gradient / math.sqrt(bias_square)
    return Model(weights, bias)


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Area under the ROC curve, by ranks; ``None`` when one class is missing."""
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = scores.argsort().argsort() + 1.0
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def lift(scores: np.ndarray, labels: np.ndarray, top: float = 0.1) -> float | None:
    """Pass rate of the top ``top`` share by score, over the average pass rate."""
    if not len(labels) or not labels.any():
        return None
    count = max(1, int(len(labels) * top))
    best = scores.argsort()[::-1][:count]
    return float(labels[best].mean() / labels.mean())


@dataclass(slots=True)
class Report:
    samples: int
    positives: int
    auc: float | None
    lift: float | None
    model: Model | None = field(default=None, repr=False)

    @property
    def trusted(self) -> bool:
        return self.lift is not None and self.lift >= MIN_LIFT


def train(rows: list[tuple[list[str], bool]]) -> Report:
    """Fit on the older rows, judge on the newest :data:`HOLDOUT`, then refit on all of them.

    ``rows`` must be oldest first.
    """
    samples = [hashed(names) for names, _ in rows]
    labels = np.array([1.0 if passed else 0.0 for _, passed in rows])
    split = int(len(rows) * (1 - HOLDOUT))
    measured_auc = measured_lift = None
    if split and split < len(rows):
        held = fit(samples[:split], labels[:split])
        scores = np.array([held.score(s) for s in samples[split:]])
        measured_auc, measured_lift = auc(scores, labels[split:]), lift(scores, labels[split:])
    model = fit(samples, labels) if len(rows) else None
    return Report(len(rows), int(labels.sum()), measured_auc, measured_lift, model)
