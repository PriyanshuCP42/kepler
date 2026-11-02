"""Statistical traps: expressions BRAIN simulates whose result is quietly meaningless."""

import pytest

from alpha_harness.labs.fastexpr import lint, parse


@pytest.mark.parametrize(
    ("expression", "says"),
    [
        ("ts_backfill(rank(x), 5)", "clean raw data first"),
        ("winsorize(rank(x), std=3)", "clean raw data first"),
        ("ts_zscore(rank(x), 20)", "rank(ts_zscore(x, d))"),
        ("sqrt(x)", "signed_power(x, 0.5)"),
        ("log(x)", "s_log_1p"),
        ("power(x, 2)", "signed_power(x, 2)"),
        ("ts_corr(x, y, 2)", "at least 20"),
        ("ts_zscore(x, 1)", "at least 5"),
        ("ts_product(x, 252)", "exp(ts_mean(log(x), d))"),
        ("ts_returns(eps, 5)", "ts_delta"),
        ("ts_decay_linear(ts_delta(x, 5) > 0, 10)", "trade_when filter"),
        ("rank(is_in_index == 1)", "trade_when filter"),
        ("trade_when(x >= ts_sum(x, 5) / 35, rank(-y), -1)", "Divide by 5"),
    ],
)
def test_each_trap_is_named_with_its_fix(expression, says):
    found = lint(parse(expression))
    assert len(found) == 1
    assert says in found[0]


@pytest.mark.parametrize(
    "expression",
    [
        "rank(ts_backfill(x, 5))",
        "rank(winsorize(x, std=3))",
        "rank(ts_zscore(x, 20))",
        "signed_power(x, 0.5)",
        "sqrt(abs(x))",
        "ts_corr(x, y, 20)",
        "ts_returns(close, 5)",
        "ts_decay_linear(ts_delta(x, 5), 10)",
        "trade_when(x >= ts_sum(x, 5) / 5, rank(-y), -1)",
        "trade_when(is_in_index == 1, rank(x), -1)",
    ],
)
def test_the_fixes_are_clean(expression):
    assert lint(parse(expression)) == []


def test_a_trap_split_across_statements_is_still_found():
    found = lint(parse("a = rank(x);\nb = ts_backfill(a, 21);\ngroup_neutralize(b, industry)"))
    assert found and "clean raw data first" in found[0]


def test_template_variables_are_left_alone():
    assert lint(parse("a = ts_backfill($field, 63);\n$sign * rank($op(a, $w))")) == []
