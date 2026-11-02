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


def test_a_search_is_crowded_once_its_first_passes_all_reach_brains_limit() -> None:
    from alpha_harness.agents.crowding import SURVIVORS, search_crowded

    assert search_crowded([0.81, 0.92]) is None  # too early
    assert search_crowded([0.81, 0.92, 0.7][:SURVIVORS]) is True
    assert search_crowded([0.81, 0.55]) is False  # one clean pass keeps it going


def test_the_crowded_message_names_the_values() -> None:
    from alpha_harness.agents.crowding import crowded_message

    message = crowded_message([0.92, 0.81, 0.77])
    assert "0.77, 0.81, 0.92" in message and "0.7" in message


async def test_the_watch_sends_unmeasured_passes_only_while_free(monkeypatch) -> None:
    from types import SimpleNamespace

    from alpha_harness.agents import crowding

    sent: list[list[str]] = []
    watch = crowding.ProdWatch(
        SimpleNamespace(
            auth=SimpleNamespace(session=SimpleNamespace(authenticated=True)),
            backfill=SimpleNamespace(busy=True),
        )
    )

    async def unmeasured() -> list[str]:
        return ["a", "b"]

    watch.unmeasured = unmeasured  # type: ignore[method-assign]
    assert await watch.round() == []  # another job holds BRAIN's rate limit

    import alpha_harness.api.lab_tasks as lab_tasks

    async def start(_state, ids):
        sent.append(ids)
        return "t1"

    monkeypatch.setattr(lab_tasks, "start_prod_correlation", start)
    watch.state.backfill.busy = False
    assert await watch.round() == ["a", "b"]
    assert sent == [["a", "b"]]
