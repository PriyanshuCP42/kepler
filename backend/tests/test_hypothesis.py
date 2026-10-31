"""The Hypothesis Writer: reading the model's answer, and screening its expressions."""

import json

from alpha_harness.agents.guard import skeleton
from alpha_harness.agents.hypothesis import parse_hypotheses, screen
from alpha_harness.agents.overlap import Overlap
from alpha_harness.labs.fastexpr import operator_table
from alpha_harness.labs.power_pool import Context, Field

OPERATORS = [
    {"name": "rank", "definition": "rank(x)"},
    {"name": "ts_rank", "definition": "ts_rank(x, d)"},
    {"name": "ts_delta", "definition": "ts_delta(x, d)"},
    {"name": "group_rank", "definition": "group_rank(x, group)"},
    {"name": "vec_avg", "definition": "vec_avg(x)"},
    {"name": "add", "definition": "add(x, y)"},
    {"name": "subtract", "definition": "subtract(x, y)"},
    {"name": "divide", "definition": "divide(x, y)"},
    {"name": "multiply", "definition": "multiply(x, y)"},
]


def _ctx() -> Context:
    fields = (Field("si_days", "MATRIX", 0.9, "Days to cover", None),)
    basics = (Field("close", "MATRIX", 1.0, "Close", None),)
    return Context(
        id="risk59",
        name="Short interest",
        category="Risk",
        description="",
        fields=fields,
        basics=basics,
        groups=("industry",),
        types={"si_days": "MATRIX", "close": "MATRIX", "industry": "GROUP"},
        own=frozenset({"si_days"}),
        names=frozenset({"si_days", "close", "industry"}),
        held={},
    )


def _overlap(pool: dict[str, list[frozenset[str]]] | None = None) -> Overlap:
    overlap = Overlap(None, None)  # type: ignore[arg-type]
    overlap.pool = pool or {}
    return overlap


class TestParse:
    def test_reads_the_json_answer(self):
        text = json.dumps(
            {"hypotheses": [{"title": "T", "rationale": "R", "expressions": ["rank(close)"]}]}
        )
        assert parse_hypotheses(text)[0]["title"] == "T"

    def test_reads_a_fenced_block_and_skips_broken_items(self):
        body = json.dumps({"hypotheses": [{"title": "T", "expressions": ["x"]}, {"title": ""}]})
        assert len(parse_hypotheses(f"Here you go:\n```json\n{body}\n```")) == 1

    def test_nonsense_is_nothing(self):
        assert parse_hypotheses("no json here") == []


class TestScreen:
    def test_keeps_a_good_expression(self):
        kept, dropped = screen(
            ["ts_rank(si_days, 60)"], _ctx(), operator_table(OPERATORS), "USA", set(), _overlap()
        )
        assert kept == ["ts_rank(si_days, 60)"]
        assert dropped == []

    def test_drops_made_up_operators_and_fields(self):
        kept, dropped = screen(
            ["magic(si_days)", "rank(nonexistent_field)"],
            _ctx(),
            operator_table(OPERATORS),
            "USA",
            set(),
            _overlap(),
        )
        assert kept == []
        assert len(dropped) == 2

    def test_drops_what_uses_none_of_the_dataset(self):
        kept, _ = screen(
            ["rank(close)"], _ctx(), operator_table(OPERATORS), "USA", set(), _overlap()
        )
        assert kept == []

    def test_drops_what_the_vault_already_tried_with_other_numbers(self):
        known = {skeleton("ts_rank(si_days, 20)")}
        kept, dropped = screen(
            ["ts_rank(si_days, 60)"], _ctx(), operator_table(OPERATORS), "USA", known, _overlap()
        )
        assert kept == []
        assert "already simulated" in dropped[0]

    def test_drops_a_repeat_of_your_pool(self):
        pool = _overlap({"USA": [frozenset({"si_days"})]})
        kept, dropped = screen(
            ["ts_rank(si_days, 60)"], _ctx(), operator_table(OPERATORS), "USA", set(), pool
        )
        assert kept == []
        assert "your pool" in dropped[0]

    def test_a_repeat_within_one_answer_is_kept_once(self):
        kept, _ = screen(
            ["ts_rank(si_days, 60)", "ts_rank(si_days, 120)"],
            _ctx(),
            operator_table(OPERATORS),
            "USA",
            set(),
            _overlap(),
        )
        assert kept == ["ts_rank(si_days, 60)"]
