"""Evolution Lab children are expressions BRAIN accepts, multi-statement seeds included.

On 2026-11-08 a task spent 420 simulations on children BRAIN refused: crossover spliced a
call from one program into another, leaving assignments nothing read, names read before
(or without) their assignment, and groups where numbers go. These expressions are synthetic.
"""

import random

import pytest

from alpha_harness.labs import ga
from alpha_harness.labs.fastexpr import (
    graft_sites,
    grafts,
    operator_table,
    parse,
    prune,
    reads,
    render,
    scope_problems,
    statements,
    unit_problems,
    validate,
    walk,
)


def _op(name: str, category: str, definition: str) -> dict[str, object]:
    return {
        "name": name,
        "category": category,
        "definition": definition,
        "scope": ["REGULAR"],
        "level": "ALL",
    }


OPERATORS = [
    _op("rank", "Cross Sectional", "rank(x, rate=2)"),
    _op("zscore", "Cross Sectional", "zscore(x)"),
    _op("ts_rank", "Time Series", "ts_rank(x, d, constant = 0)"),
    _op("ts_mean", "Time Series", "ts_mean(x, d)"),
    _op("ts_delta", "Time Series", "ts_delta(x, d)"),
    _op("ts_decay_linear", "Time Series", "ts_decay_linear(x, d, dense = false)"),
    _op("ts_backfill", "Time Series", "ts_backfill(x, lookback = d, k=1)"),
    _op("abs", "Arithmetic", "abs(x)"),
    _op("reverse", "Arithmetic", "reverse(x)"),
    _op("signed_power", "Arithmetic", "signed_power(x, y)"),
    _op("densify", "Arithmetic", "densify(x)"),
    _op("min", "Arithmetic", "min(x, y ..)"),
    _op("max", "Arithmetic", "max(x, y, ..)"),
    _op("group_rank", "Group", "group_rank(x, group)"),
    _op("group_zscore", "Group", "group_zscore(x, group)"),
    _op("group_neutralize", "Group", "group_neutralize(x, group)"),
    _op("group_mean", "Group", "group_mean(x, weight, group)"),
    _op("group_cartesian_product", "Group", "group_cartesian_product(g1, g2)"),
    _op("vec_avg", "Vector", "vec_avg(x)"),
    _op("vec_sum", "Vector", "vec_sum(x)"),
    _op("bucket", "Transformational", 'bucket(rank(x), range="0, 1, 0.1")'),
    _op("trade_when", "Transformational", "trade_when(x, y, z)"),
]


def _field(field_id: str, dataset: str, kind: str = "MATRIX") -> dict[str, str]:
    return {"field_id": field_id, "dataset_id": dataset, "field_type": kind}


FIELDS = [
    *(_field(f"fnd_alpha_{i}", "fnd") for i in range(4)),
    *(_field(f"est_beta_{i}", "est") for i in range(3)),
    _field("close", "pv"),
    _field("returns", "pv"),
    _field("cap", "pv"),
    _field("news_vec", "nws", "VECTOR"),
    _field("flow_vec", "nws", "VECTOR"),
    _field("cls_custom", "cls", "GROUP"),
    *(_field(g, "pv", "GROUP") for g in ("market", "sector", "industry", "subindustry")),
]

MARKET = ga.build_market(OPERATORS, FIELDS, ["SUBINDUSTRY", "INDUSTRY", "SLOW"])

#: Programs in the shapes the failing task's seeds had: helpers, an ``alpha`` and a final
#: call; a group made once and used twice; and plain single expressions.
SEEDS = [
    "df = ts_backfill(vec_avg(news_vec), 21); alpha = ts_rank(df, 63) - rank(fnd_alpha_0); "
    "group_neutralize(alpha, subindustry)",
    "r = ts_backfill(fnd_alpha_1, 21); d = ts_backfill(vec_sum(flow_vec), 21); "
    "s = group_rank(ts_delta(r, 5), densify(group_cartesian_product(sector, industry))) "
    "- rank(d); signed_power(group_neutralize(s, subindustry) - 0.5, 1.5)",
    "g = densify(group_cartesian_product(sector, industry)); a = group_rank(ts_mean("
    "est_beta_0, 21), g); c = group_zscore(ts_rank(est_beta_1, 42), g); "
    "signal = group_neutralize(0.5 * a + 0.5 * c, industry); ts_decay_linear(signal, 10)",
    "rank(ts_delta(fnd_alpha_2, 10)) * rank(-ts_delta(close, 1))",
    "group_rank(ts_mean(est_beta_2, 63), densify(cls_custom))",
    "trade_when(ts_rank(returns, 21) > 0.5, zscore(ts_mean(fnd_alpha_3, 42)), -1)",
]


def _parents() -> list[ga.Parent]:
    settings = {"neutralization": "SUBINDUSTRY", "decay": 0, "truncation": 0.08}
    return [p for i, e in enumerate(SEEDS) if (p := ga.Parent.of(f"seed{i}", e, settings))]


def test_seeds_are_valid_to_begin_with():
    for expression in SEEDS:
        assert MARKET.problems(parse(expression)) == [], expression


def test_every_child_is_an_expression_brain_accepts():
    parents = _parents()
    shapes = {p.tree: ga.Shape.of(p.tree, MARKET) for p in parents}
    rng = random.Random(20261108)
    made: list[str] = []
    for _ in range(600):
        a, b = rng.choice(parents), rng.choice(parents)
        child = ga.child(rng, a, b, MARKET, 0.1, shapes)
        if child is None:
            continue
        tree, _ = child
        text = render(tree)
        made.append(text)
        assert ga.sendable(text, MARKET) == [], text
        program = statements(tree)
        defined: set[str] = set()
        for index, statement in enumerate(program):
            value = statement.args[0] if statement.kind == "assign" else statement
            assigned = {s.value for s in program if s.kind == "assign"}
            assert reads(value) & assigned <= defined, text
            if statement.kind == "assign":
                assert any(statement.value in reads(s) for s in program[index + 1 :]), text
                defined.add(statement.value)
        assert program[-1].kind != "assign", text
    assert len(made) > 500
    # Breeding still reaches across programs rather than only mutating one.
    assert len(set(made)) > 300
    assert any(parse(text).kind == "seq" for text in made)


def test_a_graft_from_a_program_carries_what_its_names_hold():
    program = parse("df = ts_backfill(fnd_alpha_0, 21); alpha = rank(df); ts_mean(alpha, 5)")
    taken = grafts(program)
    assert parse("ts_mean(rank(ts_backfill(fnd_alpha_0, 21)), 5)") in taken
    for call in taken:
        assert not reads(call) & {"df", "alpha"}


def test_a_program_is_never_grafted_over_a_whole_statement():
    program = parse("a = rank(fnd_alpha_0); ts_mean(a, 5)")
    assert all(len(path) > 1 for path in graft_sites(program))


def test_groups_are_neither_grafted_nor_grafted_over():
    tree = parse("group_rank(ts_mean(x, 5), densify(group_cartesian_product(sector, industry)))")
    sites = graft_sites(tree)
    taken = grafts(tree)
    assert [render(c) for c in taken] == [
        "group_rank(ts_mean(x, 5), densify(group_cartesian_product(sector, industry)))",
        "ts_mean(x, 5)",
    ]
    assert sites == [(0,)]


@pytest.mark.parametrize(
    ("before", "after"),
    [
        # The failing task's shape: a whole final statement replaced, helpers left unread.
        (
            "df = ts_backfill(x, 21); alpha = rank(df); ts_decay_linear(group_rank(y, sector), 10)",
            "ts_decay_linear(group_rank(y, sector), 10)",
        ),
        # Pruning one assignment leaves the one it read unread: both go.
        ("r = ts_mean(x, 5); d = rank(r); s = rank(y); s", "s = rank(y); s"),
        ("a = rank(x); b = rank(a); b - a", "a = rank(x); b = rank(a); b - a"),
        # A name assigned again is live up to the assignment that reads it.
        ("a = rank(x); a = ts_mean(a, 5); a", "a = rank(x); a = ts_mean(a, 5); a"),
        ("rank(x)", "rank(x)"),
    ],
)
def test_prune_drops_only_assignments_nothing_reads(before, after):
    assert render(prune(parse(before))) == after


@pytest.mark.parametrize(
    ("expression", "says"),
    [
        ("alpha = rank(alpha); alpha", "alpha is read before it is assigned."),
        ("a = rank(b); b = rank(x); a", "b is read before it is assigned."),
        ("df = rank(x); alpha = rank(y); alpha", "df is assigned but never read."),
        ("a = rank(x)", "ends with an assignment"),
    ],
)
def test_scope_problems_name_what_brain_refuses(expression, says):
    assert any(says in p for p in scope_problems(parse(expression)))


def test_a_clean_program_has_no_scope_problems():
    assert scope_problems(parse(SEEDS[1])) == []


@pytest.mark.parametrize(
    ("expression", "says"),
    [
        ("densify(ts_delta(close, 5))", "densify takes a group at input 1."),
        ("rank(x); group_cartesian_product(sector, industry)", "ends with a group"),
        ("abs(densify(sector))", "abs takes a number, not a group at input 1."),
        ("group_rank(x, ts_mean(y, 5))", "group_rank takes a group at input 2."),
        ("g = densify(sector); rank(g)", "rank takes a number, not a group at input 1."),
        ("rank(x) - sector", "A group cannot go into -."),
    ],
)
def test_unit_problems_find_groups_out_of_place(expression, says):
    table = operator_table(OPERATORS)
    assert any(says in p for p in unit_problems(parse(expression), table))


@pytest.mark.parametrize(
    "expression",
    [
        "group_mean(x, cap, sector)",
        'g = bucket(rank(cap), range="0, 1, 0.1"); group_neutralize(rank(x), g)',
        "group_rank(x, densify(group_cartesian_product(market, subindustry)))",
    ],
)
def test_groups_in_their_places_pass(expression):
    assert unit_problems(parse(expression), operator_table(OPERATORS)) == []


def test_a_markets_own_group_fields_are_groups():
    assert MARKET.problems(parse("group_rank(rank(close), densify(cls_custom))")) == []
    assert MARKET.problems(parse("rank(cls_custom)"))


def test_a_vector_field_is_read_only_through_a_vec_operator():
    assert MARKET.problems(parse("rank(vec_avg(news_vec))")) == []
    assert MARKET.problems(parse("rank(news_vec)")) == [
        "news_vec is a vector field: read it through a vec_ operator."
    ]


def test_an_ellipsis_written_after_an_input_keeps_that_input_required():
    table = operator_table(OPERATORS)
    assert table["min"].required == 2
    assert table["max"].required == 2
    assert validate(parse("min(cap)"), table, {"cap"}) == ["min does not take 1 inputs."]
    assert validate(parse("min(cap, close, returns)"), table, {"cap", "close", "returns"}) == []


def test_an_operator_returning_a_group_is_never_swapped():
    tree = parse("group_rank(rank(close), densify(sector))")
    densify = next(n for _, n in walk(tree) if n.kind == "call" and n.value == "densify")
    assert MARKET.peers(densify) == []
    assert "densify" not in {o.name for ops in MARKET.swappable.values() for o in ops}


def test_a_refused_child_is_never_sent():
    assert ga.sendable("df = rank(close); rank(returns)", MARKET) == [
        "df is assigned but never read."
    ]
    assert ga.sendable("rank(close", MARKET)[0].startswith("Its expression could not be read")


def test_a_seed_brain_would_refuse_says_why():
    assert ga.unreadable("a = rank(close); rank(a) + rank(b); b = rank(cap)", MARKET)
