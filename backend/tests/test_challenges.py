"""A consultant challenge's standing: which Alphas count, the ranking metric, daily rules."""

from datetime import UTC, date, datetime

from alpha_harness.agents import challenges as ch

RULES = ch.Challenge(
    id="t",
    name="Test",
    start=date(2026, 11, 7),
    end=date(2026, 11, 10),
    regions=frozenset({"USA", "GLB"}),
    min_alphas=4,
    min_daily_simulations=1000,
    min_daily_submissions=1,
    theme_region="GLB",
    theme_delay=1,
    min_theme_alphas=1,
)


def _a(alpha_id, region, day, prod, theme=None):
    return ch.Submitted(alpha_id, region, day, prod, theme_match=theme)


def test_only_usa_glb_alphas_under_the_ceiling_inside_the_window_count():
    subs = [
        _a("usa", "USA", date(2026, 11, 7), 0.40),
        _a("glb", "GLB", date(2026, 11, 8), 0.60, theme=True),
        _a("crowded", "USA", date(2026, 11, 8), 0.70),  # at the ceiling: not below it
        _a("eur", "EUR", date(2026, 11, 8), 0.30),
        _a("early", "USA", date(2026, 11, 6), 0.30),
        _a("unmeasured", "GLB", date(2026, 11, 8), None),
    ]
    st = ch.standing(RULES, subs, {}, date(2026, 11, 8))
    assert st.eligible == 2
    assert st.average_prod == 0.5  # the ranking metric: (0.40 + 0.60) / 2
    assert st.theme_alphas == 1
    assert st.unmeasured == 1
    reasons = {e.alpha_id: e.reasons for e in st.entries}
    assert reasons["crowded"] == [ch.PROD_TOO_HIGH]
    assert reasons["eur"] == [ch.WRONG_REGION]
    assert reasons["unmeasured"] == [ch.NOT_MEASURED]
    assert "early" not in reasons  # before the window: not an entry at all


def test_a_past_day_short_of_either_daily_rule_is_missed_and_today_is_still_open():
    subs = [_a("a", "USA", date(2026, 11, 7), 0.5), _a("b", "GLB", date(2026, 11, 9), 0.5)]
    sims = {date(2026, 11, 7): 1200, date(2026, 11, 8): 2000, date(2026, 11, 9): 300}
    st = ch.standing(RULES, subs, sims, date(2026, 11, 9))
    # 7 Nov met both rules; 8 Nov ran enough but submitted nothing; 9 Nov is today.
    assert st.missed_days == [date(2026, 11, 8)]
    assert [d.met for d in st.days] == [None, False, True]  # newest first
    assert st.simulations_needed_today == 700
    assert st.submissions_needed_today == 0
    # Two of four alphas still needed over today and tomorrow.
    assert (st.alphas_needed, st.days_left, st.alphas_per_day) == (2, 2, 1)


def test_a_brain_timestamp_is_read_on_brains_calendar():
    # 03:30 UTC on 8 Nov is still 7 Nov in New York.
    assert ch.platform_day(datetime(2026, 11, 8, 3, 30, tzinfo=UTC)) == date(2026, 11, 7)
    assert ch.platform_day("2026-11-07T23:10:00-05:00") == date(2026, 11, 7)
    assert ch.platform_day("2026-11-07T23:10:00") == date(2026, 11, 7)  # BRAIN's local time
    assert ch.platform_day(None) is None


def test_adding_a_low_correlation_alpha_lowers_the_average():
    assert round(ch.average_after([0.6, 0.6], 0.3), 2) == 0.5


def test_a_candidate_qualifies_only_within_power_pool_limits_and_lifting_both_pools():
    from alpha_harness.api.competitions import PoolEffect, assess

    up = PoolEffect(members=10, before=7.0, after=7.1)
    down = PoolEffect(members=10, before=7.0, after=6.9)
    ok = {"power_ok": True, "regular": False, "theme_match": True}
    assert assess(**ok, whole=up, power=up) == (True, [])
    # Lowers the Power Pool combo: not wanted however low its correlation.
    assert assess(**ok, whole=up, power=down) == (False, ["LOWERS_A_POOL"])
    # A pure Power Pool Alpha outside the theme cannot be submitted.
    off = {**ok, "theme_match": False}
    assert assess(**off, whole=up, power=up)[1] == ["NEEDS_REGULAR_OR_THEME"]
    # A regular one needs no theme; one over the limits never qualifies.
    assert assess(**{**off, "regular": True}, whole=up, power=up)[1] == []
    big = {**ok, "power_ok": False}
    assert assess(**big, whole=up, power=up)[1] == ["OUTSIDE_POWER_POOL_LIMITS"]
    assert assess(**ok, whole=None, power=up) == (None, ["EFFECT_NOT_MEASURED"])
