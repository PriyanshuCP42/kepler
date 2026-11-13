"""The Dataset Doctor gives each field the shape a recipe needs."""

from alpha_harness.labs.doctor import Shape, candidates, classify


def _f(field_id, description="", category="other", **over):
    return {"field_id": field_id, "description": description, "category_id": category} | over


def test_fields_get_the_shape_that_decides_their_recipe():
    assert (
        classify(_f("count_very_positive_sentiment", "Count of very positive sentiment"))
        is Shape.COUNT
    )
    assert classify(_f("negative_word_count_mtnews", "Number of negative words")) is Shape.COUNT
    assert classify(_f("mean_surprise_value_gps", "Mean earnings surprise")) is Shape.SURPRISE
    assert (
        classify(_f("oth296_prob_pos", "Raw probability the topic is positive"))
        is Shape.PROBABILITY
    )
    assert (
        classify(_f("anl69_ebit_best_ebit_4wk_up", "EBIT estimates revised up", "analyst"))
        is Shape.ESTIMATE
    )
    assert classify(_f("latent_feature57_value", "Value of the 57th latent feature")) is Shape.LEVEL


def test_identifiers_timestamps_and_flags_are_never_signals():
    for f in (
        _f("mdl239_isin", "ISIN Security Identifier"),
        _f("oth296_call_time", "UTC date and time of the earnings call"),
        _f("oth699_executiontimestamp", "Timestamp when recorded"),
        _f("anl10_entpast_det_excflag_6310", "Exclusion flag for estimates"),
    ):
        assert classify(f) is Shape.SKIP, f["field_id"]


def test_candidates_keep_usable_uncrowded_fields_and_rank_valuable_datasets_first():
    datasets = [
        {
            "dataset_id": "news_a",
            "category_id": "news",
            "value_score": 6,
            "pyramid_multiplier": 1.4,
        },
        {
            "dataset_id": "model_b",
            "category_id": "model",
            "value_score": 3,
            "pyramid_multiplier": 1.1,
        },
        {"dataset_id": "techindi_model", "category_id": "other", "value_score": 9},
    ]
    fields = [
        _f(
            "count_pos",
            "Count of positive items",
            "news",
            dataset_id="news_a",
            coverage=0.9,
            alpha_count=0,
        ),
        _f(
            "count_neg",
            "Count of negative items",
            "news",
            dataset_id="news_a",
            coverage=0.4,
            alpha_count=0,
        ),
        _f("score_x", "Model score", "model", dataset_id="model_b", coverage=0.8, alpha_count=2),
        _f("score_y", "Model score", "model", dataset_id="model_b", coverage=0.8, alpha_count=99),
        _f("label_z", "Predicted label", "other", dataset_id="techindi_model", coverage=1.0),
    ]
    found = candidates(datasets, fields)
    assert [(c.dataset_id, c.shape) for c in found] == [
        ("news_a", Shape.COUNT),
        ("model_b", Shape.LEVEL),
    ]
    assert found[0].field_ids == ["count_pos"]  # 40% coverage is left out
    assert found[1].field_ids == ["score_x"]  # 99 alphas is crowded
    assert candidates(datasets, fields, avoid=frozenset({"news_a"}))[0].dataset_id == "model_b"


def test_words_inside_other_words_do_not_mislead_it():
    # "missing data filled" is no miss, "emissions" no surprise.
    ratio = _f("buy_sell_ratio_all_20d_filled", "Ratio of purchases to sales, missing data filled")
    assert classify(ratio) is Shape.LEVEL
    assert (
        classify(_f("oth696_ghgemissions_positive_82", "greenhouse gas emissions")) is Shape.LEVEL
    )
    # A time series is a signal; an as-of date in a description is not a date field.
    delta = _f("mdl239_combo_deltaactivity", "Normalized day-over-day time-series change")
    assert classify(delta) is Shape.LEVEL
    volume = _f("daily_option_contracts_traded_4", "Trailing 5-day, as-of-date option volume")
    assert classify(volume) is Shape.LEVEL
    # Actual against predicted, written with underscores, is a surprise.
    gap = _f("pct_diff_actual_vs_predicted_annual_eps_long", "")
    assert classify(gap) is Shape.SURPRISE
    # "Forecast" in a text dataset is a text score, not an analyst estimate.
    assert classify(_f("min_forecast_score_text", "Minimum forecast score", "other")) is Shape.LEVEL


def test_a_currency_adjusted_value_is_a_signal_but_a_currency_code_is_not():
    value = _f("median_surprise_value_fxadj_cfo_annual12", "Currency-adjusted median surprise")
    assert classify(value) is Shape.SURPRISE
    assert classify(_f("listing_ccy", "Currency code of the listing")) is Shape.SKIP
