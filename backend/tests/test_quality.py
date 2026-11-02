"""Quality search: three objectives and four constraints, under constrained NSGA-II."""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
from optuna.distributions import CategoricalDistribution, distribution_to_json

from alpha_harness.labs import quality
from alpha_harness.labs import study as study_module


def _days(values, start=date(2014, 1, 1)):
    return {start + timedelta(days=i): float(v) for i, v in enumerate(values)}


def test_the_elite_pool_is_a_fifth_of_the_fields_within_bounds():
    assert quality.population(50, 3000) == quality.MIN_POPULATION
    assert quality.population(1500, 3000) == 300
    assert quality.population(9000, 9000) == quality.MAX_POPULATION
    # A small task still gets three generations.
    assert quality.population(50, 150) == 50


def test_hard_checks_and_errors_are_counted_others_are_not():
    checks = [
        {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"},
        {"name": "LOW_SHARPE", "result": "FAIL"},
        {"name": "CONCENTRATED_WEIGHT", "result": "ERROR"},
        {"name": "HIGH_TURNOVER", "result": "PASS"},
    ]
    assert quality.hard_failures(checks) == 2


def test_the_cheap_constraints_are_met_at_zero_or_less():
    assert quality.cheap_constraints(1.4, 0.0008, []) == [-0.4, 0.0, -3.0]
    sharpe, checks, margin = quality.cheap_constraints(0.7, 0.0002, [])
    assert sharpe > 0 and checks == 0 and margin > 0


def test_a_steady_climb_beats_a_jump_on_k_ratio_with_test_years_unseen():
    rng = np.random.default_rng(0)
    steady = rng.normal(loc=1.0, scale=1.0, size=2500)
    jumpy = np.zeros(2500)
    jumpy[::250] = 250.0
    k_steady, calmar = quality.train_curve(_days(steady), None)
    k_jumpy, _ = quality.train_curve(_days(jumpy), None)
    assert k_steady > k_jumpy and calmar > 0
    # Losses after the split change nothing: the test years are not read.
    lossy = np.concatenate([steady[:1800], -abs(steady[1800:]) * 50])
    split = (date(2014, 1, 1) + timedelta(days=1800)).isoformat()
    assert quality.train_curve(_days(lossy), split) == quality.train_curve(_days(steady), split)


def test_a_crowded_candidate_is_infeasible_and_an_unfetched_one_scores_nothing():
    crowded = quality.score(1.6, 1.2, 0.001, [], (2.0, 1.5), pool_correlation=0.75)
    assert not crowded.feasible and crowded.constraints[3] > 0
    clean = quality.score(1.6, 1.2, 0.001, [], (2.0, 1.5), pool_correlation=0.3)
    assert clean.feasible and clean.values == [2.0, 1.5, 1.2]
    unread = quality.score(0.4, 0.3, 0.001, [])
    assert unread.values[:2] == [0.0, 0.0] and not unread.feasible


def test_nsga2_runs_on_the_lab_space_and_a_restart_keeps_its_generations():
    row = SimpleNamespace(objectives=list(quality.OBJECTIVES), max_trials=60)
    fields = tuple(f"f{i}" for i in range(12))
    slots = {"field": fields}
    sampler = study_module._sampler(row, slots)
    assert sampler.population_size == 20
    study = study_module._rebuild(sampler, [], study_module._Distributions(slots), ["maximize"] * 3)

    rng = np.random.default_rng(1)
    finished = []
    for number in range(45):
        trial = study.ask()
        field = trial.suggest_categorical("field", fields)
        good = int(field[1:]) % 2 == 0
        scored = quality.score(1.5 if good else 0.5, rng.random(), 0.001, [], (rng.random(), 1.0))
        stored = SimpleNamespace(id=number, number=number, params={"field": field})
        summary = {"constraints": scored.constraints, "generation": None}
        summary["generation"] = study_module._generation(trial)
        finished.append((trial, stored, scored.values, summary))
        study_module._tell_many(study, [finished[-1]], {}, study_module._Distributions(slots))
    generations = {t.system_attrs.get(quality.GENERATION_KEY) for t in study.trials}
    assert {0, 1} <= generations
    assert all("constraints:sharpe" in t.system_attrs for t in study.trials)

    # Rebuilt from the stored rows, each trial keeps its constraints and generation.
    history = [
        {
            "params": {"field": stored.params["field"]},
            "distributions": {"field": distribution_to_json(CategoricalDistribution(fields))},
            "values": values,
            "state": "COMPLETE",
            "system": study_module._system_attrs(summary),
        }
        for _, stored, values, summary in finished
    ]
    loader = study_module._Distributions(slots)
    fresh = study_module._sampler(row, slots)
    again = study_module._rebuild(fresh, history, loader, ["maximize"] * 3)
    assert [t.system_attrs.get(quality.GENERATION_KEY) for t in again.trials] == [
        t.system_attrs.get(quality.GENERATION_KEY) for t in study.trials
    ]
    assert again.trials[0].system_attrs["constraints:pool"] == 0.0
    nxt = again.ask()
    nxt.suggest_categorical("field", fields)
    assert again.sampler.get_trial_generation(again, nxt._cached_frozen_trial) >= 1
