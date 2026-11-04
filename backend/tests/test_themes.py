"""This month's Power Pool themes, and whether BRAIN can take an alpha."""

import json

from alpha_harness.agents import themes
from alpha_harness.agents.repair import _outcome
from alpha_harness.api.desk import BrainComparison, PowerPoolRoute, judge
from alpha_harness.db.models import Trial, TrialState

THEMES_CHECK = {
    "name": "MATCHES_THEMES",
    "result": "WARNING",
    "themes": [
        {"id": "a", "multiplier": 1.0, "name": "GLB/D1 Power Pool Nov`26"},
        {"id": "b", "multiplier": 2.0, "name": "ASI/D1 MINVOL10M Theme"},
    ],
}
SHARPE_WARN = {"name": "LOW_SHARPE", "result": "WARNING", "value": 1.17, "limit": 1.58}
SHARPE_PASS = {"name": "LOW_SHARPE", "result": "PASS", "value": 1.9, "limit": 1.58}
POWER_POOL = {"name": "POWER_POOL_CORRELATION", "result": "PASS", "value": 0.2, "limit": 0.5}


def _themes() -> list[themes.Theme]:
    return themes.from_checks([THEMES_CHECK])


def test_themes_are_read_from_a_check():
    glb, asi = _themes()
    assert (glb.region, glb.delay, glb.multiplier, glb.universe) == ("GLB", 1, 1.0, None)
    assert (asi.region, asi.delay, asi.multiplier, asi.universe) == ("ASI", 1, 2.0, "MINVOL10M")
    assert "'" in glb.name and "`" not in glb.name


def test_a_universe_theme_covers_only_its_universe():
    _, asi = _themes()
    assert asi.covers("ASI", 1, "MINVOL10M")
    assert not asi.covers("ASI", 1, "TOP500")
    assert not asi.covers("ASI", 0, "MINVOL10M")


def test_an_alpha_over_the_regular_limits_needs_no_theme():
    row = {"region": "USA", "delay": 1, "checks": json.dumps([SHARPE_PASS])}
    assert themes.takeable(row, _themes())


def test_a_pure_power_pool_alpha_needs_a_theme_market():
    usa = {"region": "USA", "delay": 1, "checks": json.dumps([SHARPE_WARN, POWER_POOL])}
    glb = {"region": "GLB", "delay": 1, "checks": json.dumps([SHARPE_WARN, POWER_POOL])}
    assert not themes.takeable(usa, _themes())
    assert themes.takeable(glb, _themes())


def test_a_theme_holds_an_alpha_that_misses_what_it_asks_beyond_the_market():
    glb, asi = _themes()
    assert glb.requires == ("LOW_GLB_AMER_SHARPE",) and asi.requires == ()
    weak = {"name": "LOW_GLB_AMER_SHARPE", "result": "FAIL", "value": 0.62, "limit": 1}
    warned = {**weak, "result": "WARNING"}
    strong = {"name": "LOW_GLB_AMER_SHARPE", "result": "PASS", "value": 1.21, "limit": 1}
    for check, takeable in ((weak, False), (warned, False), (strong, True)):
        row = {"region": "GLB", "delay": 1, "checks": json.dumps([SHARPE_WARN, POWER_POOL, check])}
        assert themes.takeable(row, _themes()) is takeable


def test_under_the_limits_and_not_power_pool_is_never_takeable():
    row = {"region": "GLB", "delay": 1, "checks": json.dumps([SHARPE_WARN])}
    assert not themes.takeable(row, _themes())


def test_unknown_themes_hold_nothing_back():
    usa = {"region": "USA", "delay": 1, "checks": json.dumps([SHARPE_WARN, POWER_POOL])}
    assert themes.takeable(usa, [])


def test_the_desk_holds_an_off_theme_alpha():
    route = PowerPoolRoute(missed=["LOW_SHARPE"], theme=None)
    readiness, reasons = judge(0.3, 0.2, [], None, 1.5, None, route)
    assert readiness == "off_theme"
    assert "Held, not dropped" in reasons[0]


def test_a_theme_market_alpha_is_told_to_add_its_description():
    route = PowerPoolRoute(missed=["LOW_SHARPE"], theme="GLB/D1 Power Pool Nov'26")
    lifts = BrainComparison("EQUITY:USA:1", {"sharpe": 2.0}, {"sharpe": 2.2})
    readiness, reasons = judge(0.3, 0.2, [], None, 1.5, None, route, lifts)
    assert readiness == "ready"
    assert any("description" in r for r in reasons)


def test_a_repair_brain_cannot_take_is_not_fixed():
    trial = Trial(state=TrialState.COMPLETE, alpha_id="x")
    assert _outcome(trial, {"x": "pass"}, {"x"}, blocked={"x"}) == "not_fixed"
    assert _outcome(trial, {"x": "pass"}, {"x"}) == "fixed"
