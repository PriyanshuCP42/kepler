"""Gross vs estimated after-cost portfolio figures.

Reproduces the relationship seen on the Portfolio page (2026-10-30 snapshot): at 5 bps the
combined portfolio's gross margin (4.89 bps) is below the cost, so net margin is negative
(-0.11 bps) and net returns are roughly gross returns less ``turnover * 2 * bps * 250``.
Synthetic data only; no BRAIN access.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from alpha_harness.vault import metrics

BPS = 5.0


def _days(n: int) -> list[date]:
    out, day = [], date(2014, 1, 1)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


@pytest.fixture
def book() -> tuple[list[date], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    n = 1000
    # Daily PnL sized so gross margin is a few bps of traded dollars at 30% turnover.
    pnl = rng.normal(loc=2_900.0, scale=9_000.0, size=n)
    turnover = np.full(n, 0.30)
    return _days(n), pnl, turnover


def test_after_cost_subtracts_turnover_times_book_times_bps(book) -> None:
    _, pnl, turnover = book
    net = metrics.after_cost(pnl, turnover, BPS)
    expected_daily_cost = 0.30 * metrics.BOOK_SIZE * BPS / 10_000
    assert np.allclose(pnl - net, expected_daily_cost)


def test_zero_cost_is_identity(book) -> None:
    _, pnl, turnover = book
    assert np.array_equal(metrics.after_cost(pnl, turnover, 0.0), pnl)


def test_net_margin_is_gross_margin_minus_cost(book) -> None:
    """Margin is PnL per dollar traded, so it is the break-even cost: net = gross - bps."""
    _, pnl, turnover = book
    gross = metrics.stats(pnl, turnover)
    net = metrics.stats(metrics.after_cost(pnl, turnover, BPS), turnover)
    assert gross is not None and net is not None
    assert net.margin == pytest.approx(gross.margin - BPS / 10_000, abs=1e-12)


def test_net_returns_drop_by_two_times_turnover_times_bps_annualised(book) -> None:
    """Returns are on half the book (long side), traded dollars on the whole book."""
    _, pnl, turnover = book
    gross = metrics.stats(pnl, turnover)
    net = metrics.stats(metrics.after_cost(pnl, turnover, BPS), turnover)
    assert gross is not None and net is not None
    drop = 2 * 0.30 * (BPS / 10_000) * metrics.TRADING_DAYS
    assert gross.returns - net.returns == pytest.approx(drop, rel=1e-9)


def test_turnover_is_unchanged_by_cost(book) -> None:
    _, pnl, turnover = book
    gross = metrics.stats(pnl, turnover)
    net = metrics.stats(metrics.after_cost(pnl, turnover, BPS), turnover)
    assert gross is not None and net is not None
    assert gross.turnover == net.turnover


def test_cost_above_margin_gives_negative_net_pnl(book) -> None:
    _, pnl, turnover = book
    gross = metrics.stats(pnl, turnover)
    assert gross is not None
    breakeven_bps = gross.margin * 10_000
    assert breakeven_bps > 0
    below = metrics.stats(metrics.after_cost(pnl, turnover, breakeven_bps * 0.9), turnover)
    above = metrics.stats(metrics.after_cost(pnl, turnover, breakeven_bps * 1.1), turnover)
    assert below is not None and above is not None
    assert below.pnl > 0 > above.pnl


def test_windows_share_the_split_day(book) -> None:
    dates, pnl, turnover = book
    split = dates[800]
    w = metrics.windows(dates, pnl, turnover, split)
    assert w["train"] is not None and w["test"] is not None and w["in_sample"] is not None
    assert w["train"].days == 801 and w["test"].days == 200
