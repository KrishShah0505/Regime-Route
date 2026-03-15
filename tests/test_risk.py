"""tests/test_risk.py"""
import pytest
import numpy as np
import pandas as pd
from backtester.risk import (
    sharpe_ratio, max_drawdown, cagr, win_rate,
    regime_attribution, total_return,
)


def make_returns(n=500, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(np.random.randn(n) * 0.01 + 0.0003, index=dates)


def test_sharpe_positive_drift():
    r = make_returns()
    s = sharpe_ratio(r)
    assert isinstance(s, float)
    assert -5 < s < 10   # reasonable range


def test_max_drawdown_negative():
    r = make_returns()
    eq = (1 + r).cumprod() * 100_000
    mdd = max_drawdown(eq)
    assert mdd <= 0, "Max drawdown must be negative or zero"


def test_max_drawdown_flat():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    eq = pd.Series(100_000.0, index=dates)
    assert max_drawdown(eq) == 0.0


def test_cagr_reasonable():
    r = make_returns()
    eq = (1 + r).cumprod() * 100_000
    c = cagr(eq)
    assert -0.5 < c < 1.0   # between -50% and +100% CAGR


def test_win_rate_bounds():
    r = make_returns()
    wr = win_rate(r)
    assert 0 <= wr <= 1


def test_regime_attribution():
    r = make_returns()
    regimes = pd.Series(np.tile([0, 1, 2], len(r) // 3 + 1)[:len(r)], index=r.index)
    result = regime_attribution(r, regimes)
    assert len(result) > 0
    assert "sharpe" in result.columns
