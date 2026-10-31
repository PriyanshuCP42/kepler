"""The Strategist: value, yield and stalls decide the plan; history is counted per pyramid."""

import random
from datetime import UTC, datetime

from alpha_harness.agents import strategist as st


def _niche(region="DEU", delay=0, category="earnings", **kw):
    niche = st.Niche(region, delay, category, category.title(), synced=True)
    for key, value in kw.items():
        setattr(niche, key, value)
    return niche


def test_need_rewards_nearly_complete_pyramids_and_discounts_complete_ones():
    assert st.need(2) > st.need(1) > st.need(0) > st.need(3)


def test_a_productive_niche_outranks_a_dry_one_of_equal_value():
    good = _niche(category="risk", multiplier=1.2, simulations=500, families=15)
    dry = _niche(category="pv", multiplier=1.2, simulations=2000, families=0)
    wins = sum(st.assess([good, dry], random.Random(i))[0].category == "risk" for i in range(200))
    assert wins > 190


def test_a_stalled_niche_is_cooled_and_an_undownloaded_one_never_planned():
    stalled = _niche(simulations=900, recent_simulations=600, recent_passed=0, families=2)
    missing = _niche(category="news", synced=False, multiplier=1.9)
    ranked = st.assess([stalled, missing], random.Random(0))
    assert stalled.status == "stalled"
    assert missing.status == "not_downloaded" and missing.score == 0
    assert missing not in st.choose(ranked, 5)


def test_the_plan_keeps_a_place_for_an_untried_niche_and_shares_add_up():
    tried = [
        _niche(category=f"c{i}", multiplier=1.0, simulations=400, families=8) for i in range(6)
    ]
    untried = _niche(category="fresh", multiplier=1.9, simulations=0)
    picked = st.choose(st.assess([*tried, untried], random.Random(3)), 4)
    assert untried in picked
    assert len(picked) == 4
    assert abs(sum(n.share for n in picked) - 1) < 1e-9


def test_history_counts_each_alpha_against_its_first_fields_category():
    alphas = [
        {
            "alpha_id": "a",
            "expression": "rank(est_eps) + close",
            "region": "USA",
            "delay": 1,
            "date_created": datetime.now(UTC),
        },
        {
            "alpha_id": "b",
            "expression": "rank(est_eps)",
            "region": "USA",
            "delay": 1,
            "date_created": datetime(2020, 1, 1, tzinfo=UTC),
        },
        {"alpha_id": "c", "expression": "rank(", "region": "USA", "delay": 1, "date_created": None},
    ]
    fields = {"est_eps": "analyst", "close": "pv"}
    verdicts = {"a": ("pass", "a"), "b": ("pass", "a")}
    record = st._tally(alphas, fields, verdicts)[("USA", 1, "analyst")]
    assert record == {
        "simulations": 2,
        "passed": 2,
        "families": 1,
        "recent_simulations": 1,
        "recent_passed": 1,
    }


async def test_the_plan_shown_is_the_plan_written():
    niches = [_niche(category=f"c{i}", multiplier=1.0 + i / 10) for i in range(8)]

    class Fixed(st.Strategist):
        async def niche_map(self):
            return [
                st.Niche(
                    **{
                        k: getattr(n, k)
                        for k in (
                            "region",
                            "delay",
                            "category",
                            "category_name",
                            "multiplier",
                            "synced",
                        )
                    }
                )
                for n in niches
            ]

    strategist = Fixed(state=None, bus=None)
    first = [n.category for n in await strategist.assessed()]
    second = [n.category for n in await strategist.assessed()]
    assert first == second
