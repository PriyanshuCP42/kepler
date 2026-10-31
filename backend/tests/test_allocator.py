"""The learning layer: the yield posterior and the day's split of the budget."""

import random

from alpha_harness.agents.allocator import (
    EXPLORE,
    HYPOTHESIS,
    REPAIR,
    SEARCH,
    TEMPLATE,
    Arm,
    posterior,
    split,
)


def _arms(**over: dict) -> list[Arm]:
    base = {
        SEARCH: {"simulations": 2000, "good": 1},
        REPAIR: {"simulations": 400, "good": 20},
        HYPOTHESIS: {"simulations": 0, "good": 0},
        TEMPLATE: {"simulations": 4000, "good": 20, "manual": True},
    }
    return [
        Arm(key=k, lab=k, action=k, **(base[k] | over.get(k, {})))  # type: ignore[arg-type]
        for k in base
    ]


def test_posterior_starts_from_the_prior():
    a, b = posterior(0, 0)
    assert a / (a + b) == 1 / 200


def test_more_good_alphas_raise_the_yield():
    low, high = posterior(1, 1000), posterior(30, 1000)
    assert low[0] / sum(low) < high[0] / sum(high)


def test_the_whole_budget_is_spent():
    arms = _arms()
    split(arms, 3000, random.Random(1))
    assert sum(a.planned for a in arms) == 3000


def test_the_best_arm_gets_the_most():
    arms = _arms()
    split(arms, 3000, random.Random(1))
    by = {a.key: a for a in arms}
    assert by[REPAIR].planned == max(a.planned for a in arms)


def test_every_arm_keeps_its_exploration_share():
    arms = _arms()
    split(arms, 3000, random.Random(1))
    assert all(a.planned >= EXPLORE * 3000 - 1 for a in arms)


def test_capacity_is_respected_and_the_rest_goes_on():
    arms = _arms(repair={"capacity": 300}, hypothesis={"capacity": 0})
    split(arms, 3000, random.Random(1))
    by = {a.key: a for a in arms}
    assert by[REPAIR].planned <= 300
    assert by[HYPOTHESIS].planned == 0
    assert sum(a.planned for a in arms) == 3000


def test_the_same_day_gives_the_same_plan():
    first, second = _arms(), _arms()
    split(first, 3000, random.Random(7))
    split(second, 3000, random.Random(7))
    assert [a.planned for a in first] == [a.planned for a in second]


def test_no_budget_plans_nothing():
    arms = _arms()
    split(arms, 0, random.Random(1))
    assert all(a.planned == 0 for a in arms)
