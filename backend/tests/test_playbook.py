"""The Playbook: insights kept on this machine, found by the words a question shares with them."""

from alpha_harness.agents import playbook
from alpha_harness.agents.playbook import Insight, Playbook


def _book(tmp_path) -> Playbook:
    book = Playbook(tmp_path)
    book.upsert(
        Insight(
            id="osmosis-shortlist",
            title="Osmosis score shortlists consultants for full-time roles",
            summary="Qualify with 100,000 points over at least 10 Alphas in 3 regions.",
            source="Webinar",
            date="2026-11-05",
            topics=["osmosis"],
            places=["genius"],
            figures=[["Sharpe above 2.0", "160 of 5,799"]],
        )
    )
    book.upsert(
        Insight(
            id="smoothing",
            title="Smoothing operators raise margin",
            summary="ts_target_tvr_hump cuts turnover 16% and lifts Fitness 14%.",
            source="Webinar",
            date="2026-11-04",
            topics=["operators"],
            places=["labs"],
        )
    )
    return book


def test_a_question_finds_the_insight_that_shares_its_words(tmp_path):
    found = _book(tmp_path).search("how do I get a full time job through osmosis")
    assert found[0][0].id == "osmosis-shortlist"
    assert found[0][1] > 0


def test_shorthand_is_read_as_the_words_insights_use(tmp_path):
    found = _book(tmp_path).search("lower tvr")
    assert [i.id for i, _ in found] == ["smoothing"]


def test_a_screen_gets_its_own_insights_newest_first(tmp_path):
    book = _book(tmp_path)
    assert [i.id for i, _ in book.search(place="genius")] == ["osmosis-shortlist"]
    assert [i.id for i, _ in book.search()] == ["osmosis-shortlist", "smoothing"]


def test_insights_survive_a_reload_and_can_be_replaced_or_removed(tmp_path):
    _book(tmp_path)
    again = Playbook(tmp_path)
    assert {i.id for i in again.all()} == {"osmosis-shortlist", "smoothing"}
    again.upsert(Insight(id="smoothing", title="Changed", summary="x", source="me"))
    assert next(i for i in again.all() if i.id == "smoothing").title == "Changed"
    assert again.delete("smoothing") and not again.delete("smoothing")
    assert [i.id for i in Playbook(tmp_path).all()] == ["osmosis-shortlist"]


def test_an_empty_playbook_gives_no_notes(tmp_path):
    assert playbook.notes(Playbook(tmp_path), "anything") == ""
    assert "Osmosis score" in playbook.notes(_book(tmp_path), "osmosis regions")


def test_a_lever_gathers_the_insights_that_move_one_genius_bar(tmp_path):
    book = _book(tmp_path)
    book.upsert(Insight(id="pyr", title="Pyramids", summary="x", source="s", levers=["pyramids"]))
    assert [i.id for i, _ in book.search(lever="pyramids")] == ["pyr"]
    assert book.search(lever="signals") == []
    assert playbook.LEVERS[:3] == ("signals", "pyramids", "performance")
