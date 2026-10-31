"""Crowding: datasets judged by the Production Correlation of the alphas that read them."""

from __future__ import annotations

from alpha_harness.agents.crowding import expected, reckon


def test_crowded_and_clean_datasets_are_told_apart() -> None:
    measured = {f"c{i}": 0.85 for i in range(5)} | {f"k{i}": 0.40 for i in range(5)}
    datasets_of = {f"c{i}": {"techindi"} for i in range(5)} | {
        f"k{i}": {"insider"} for i in range(5)
    }
    table = reckon(measured, datasets_of)
    assert table["techindi"].crowded and not table["techindi"].clean
    assert table["insider"].clean and not table["insider"].crowded


def test_price_and_volume_is_background() -> None:
    measured = {f"a{i}": 0.9 for i in range(6)}
    datasets_of = {f"a{i}": {"pv1", "techindi"} for i in range(6)}
    table = reckon(measured, datasets_of, {"pv1"})
    assert "pv1" not in table
    assert table["techindi"].crowded


def test_too_few_measured_says_nothing() -> None:
    table = reckon({"a": 0.95, "b": 0.95}, {"a": {"x"}, "b": {"x"}})
    assert not table["x"].crowded
    assert expected({"x"}, table) is None


def test_expected_takes_the_most_crowded_dataset() -> None:
    measured = {f"a{i}": 0.8 for i in range(5)} | {f"b{i}": 0.3 for i in range(5)}
    datasets_of = {f"a{i}": {"hot"} for i in range(5)} | {f"b{i}": {"cold"} for i in range(5)}
    assert expected({"hot", "cold"}, reckon(measured, datasets_of)) == 0.8
