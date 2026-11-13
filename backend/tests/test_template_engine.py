"""The template engine: names it can read back, learning from results, and the tasks it writes."""

import random

from alpha_harness.labs import engine, recipes
from alpha_harness.labs.doctor import Candidate, Shape

GOOD = {"sharpe": 2.0, "fitness": 1.05, "margin": 0.0006, "returns": 0.04, "drawdown": 0.02}
WEAK = {"sharpe": 0.6, "fitness": 0.2, "margin": 0.0002, "returns": 0.01, "drawdown": 0.02}


def test_a_task_name_carries_what_it_tested():
    name = engine.task_name("own-history-neutral", "count", "ASI", "mmp_nlp_sentiment")
    assert engine.parse_name(name) == ("own-history-neutral", "count", "ASI", "mmp_nlp_sentiment")
    assert engine.parse_name("Library L3 USA: something") is None
    assert engine.parse_name(None) is None


def test_results_are_tallied_by_recipe_shape_and_region():
    name = engine.task_name("surprise-drift", "surprise", "USA", "analyst_consensus")
    other = engine.task_name("surprise-drift", "surprise", "GLB", "analyst_consensus")
    rows = [
        (name, {"stats": GOOD}),
        (name, {"stats": WEAK}),
        (name, {"stats": {}}),  # no figures yet: not counted
        (other, {"stats": WEAK}),
        ("Library L3 USA", {"stats": GOOD}),  # not an engine task
    ]
    tally = engine.tally(rows)
    assert (
        tally[("surprise-drift", "surprise", "USA")].tries,
        tally[("surprise-drift", "surprise", "USA")].passes,
    ) == (2, 1)
    assert tally[("surprise-drift", "surprise", "GLB")].passes == 0
    assert len(tally) == 2


def test_a_trial_passes_only_on_every_local_test():
    assert engine.passes(GOOD)
    assert not engine.passes(GOOD | {"margin": 0.0003})  # loses money after costs
    assert not engine.passes(GOOD | {"drawdown": 0.04})  # returns only equal to drawdown
    assert engine.passes(GOOD, {"LOW_SHARPE": 1.25, "LOW_FITNESS": 1.0})


def test_a_recipe_that_never_passes_is_retired_and_skipped():
    record = engine.Record(tries=engine.RETIRE_TRIES, passes=0)
    assert record.retired and not engine.Record(tries=engine.RETIRE_TRIES, passes=1).retired
    c = Candidate("news_a", "news", Shape.SURPRISE, ["f1"], priority=5.0)
    learned = {("surprise-drift", "surprise", "USA"): record}
    # Surprise has one recipe; retired in USA, nothing is chosen there, but GLB still is.
    assert engine.choose([c], learned, "USA", 3, random.Random(1)) == []
    assert engine.choose([c], learned, "GLB", 3, random.Random(1))[0].recipe.key == "surprise-drift"


def test_recipes_that_work_are_chosen_more_often():
    c = Candidate("news_a", "news", Shape.COUNT, ["f1"], priority=5.0)
    learned = {
        (r.key, "count", "USA"): engine.Record(tries=60, passes=1)
        for r in recipes.for_shape(Shape.COUNT)
    }
    learned[("own-history-neutral", "count", "USA")] = engine.Record(tries=60, passes=15)
    rng = random.Random(7)
    picked = [engine.choose([c], learned, "USA", 1, rng)[0].recipe.key for _ in range(200)]
    assert picked.count("own-history-neutral") > 180


def test_one_search_per_dataset_most_valuable_first():
    cs = [
        Candidate("a", "news", Shape.COUNT, ["f"], priority=9.0),
        Candidate("a", "news", Shape.LEVEL, ["g"], priority=8.0),
        Candidate("b", "model", Shape.LEVEL, ["h"], priority=2.0),
    ]
    chosen = engine.choose(cs, {}, "USA", 3, random.Random(3))
    assert [ch.candidate.dataset_id for ch in chosen] == ["a", "b"]


def test_a_choice_becomes_a_task_with_its_regions_rules():
    c = Candidate("mmp_nlp_sentiment", "other", Shape.COUNT, ["count_x", "count_y"], priority=1.0)
    task = engine.task_for(engine.Choice(c, recipes.BY_KEY["own-history-neutral"], 1.0), "ASI")
    # MINVOL10M: the universe of BRAIN's 2x ASI D1 theme.
    assert task.region == "ASI" and task.universes == ["MINVOL10M"]
    assert task.investability == ["max_trade"]  # ASI's investability rule
    assert task.neutralizations == ["STATISTICAL", "SLOW"]
    assert task.variables["f"].filter.field_ids == ["count_x", "count_y"]
    assert engine.parse_name(task.template_name) == (
        "own-history-neutral",
        "count",
        "ASI",
        "mmp_nlp_sentiment",
    )
    # No recipe carries a reversion term.
    assert all("returns" not in r.template for r in recipes.RECIPES)


async def test_the_engine_declines_niches_it_does_not_search():
    # Delay 0, or a region without market rules: the Strategist's Search Lab takes them.
    assert await engine.plan_niche(None, "USA", 0, "news", 100, 2) is None
    assert await engine.plan_niche(None, "JPN", 1, "news", 100, 2) is None


def test_own_history_recipes_are_slow_enough_to_beat_trading_costs():
    # Fast versions reached Sharpe 2.2 but traded so much their margin stayed near 3 bps.
    for key in ("own-history-neutral", "own-history-ranked"):
        decays = [int(d) for d in recipes.BY_KEY[key].values["d"].split(",")]
        assert min(decays) >= 20, key


def test_every_recipe_fits_power_pool_and_playbook_ones_say_where_they_came_from():
    from alpha_harness.labs.fastexpr import data_fields, operator_count
    from alpha_harness.labs.template import program

    for r in recipes.RECIPES:
        text = r.template.replace("$f", "some_field")
        for name, values in r.values.items():
            text = text.replace(f"${name}", values.split(",")[0].strip())
        tree = program(text)
        assert operator_count(tree) <= 8, r.key
        assert len(data_fields(tree)) <= 3, r.key
    turnover = recipes.BY_KEY["own-history-target-turnover"]
    assert turnover.playbook == "operators-slide" and "ts_target_tvr_hump" in turnover.template
    # One target sits inside the GLB 15-30% turnover theme band.
    assert any(0.15 <= float(t) <= 0.3 for t in turnover.values["t"].split(","))
