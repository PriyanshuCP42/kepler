"""compress: binding the original to ``alpha`` never redefines a name BRAIN already has."""

from __future__ import annotations

import pytest

from alpha_harness.tools.correlation_breaker import compress

TAIL = "rank(alpha)"


@pytest.mark.parametrize(
    ("expression", "program"),
    [
        ("rank(x)", "alpha = rank(x);\nrank(alpha)"),
        ("alpha = rank(x)", "alpha = rank(x);\nrank(alpha)"),
        (
            "alpha = winsorize(x, std=4); rank(alpha)",
            "alpha_0 = winsorize(x, std=4);\nalpha = rank(alpha_0);\nrank(alpha)",
        ),
        (
            "alpha = rank(x); b = alpha * 2; b",
            "alpha_0 = rank(x);\nb = alpha_0 * 2;\nalpha = b;\nrank(alpha)",
        ),
        (
            "alpha_0 = 1; alpha = x + alpha_0; alpha * 2",
            "alpha_0 = 1;\nalpha_1 = x + alpha_0;\nalpha = alpha_1 * 2;\nrank(alpha)",
        ),
    ],
)
def test_alpha_is_assigned_once(expression: str, program: str) -> None:
    out = compress(expression).program(TAIL)
    assert out == program
    assert sum(line.startswith("alpha =") for line in out.splitlines()) == 1
