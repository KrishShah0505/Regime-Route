"""
tests/test_engine.py
Tests for the backtest engine — most critical module to test.
"""
import pytest
import numpy as np
import pandas as pd
from backtester.engine import BacktestEngine, BacktestConfig


def make_dummy_features(n_days=500, n_tickers=5):
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    np.random.seed(42)
    prices = pd.DataFrame(
        (1 + np.random.randn(n_days, n_tickers) * 0.01).cumprod(axis=0) * 100,
        index=dates, columns=tickers,
    )
    returns = prices.pct_change()
    return {
        "close": prices,
        "returns": returns,
        "realized_vol": returns.rolling(20).std() * np.sqrt(252),
    }


def make_dummy_weights(features):
    close = features["close"]
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    weights.iloc[100:] = 0.1   # simple: long 10% each ticker from day 100
    return weights


def test_engine_runs():
    features = make_dummy_features()
    weights = make_dummy_weights(features)
    regimes = np.zeros(len(features["close"]), dtype=int)

    engine = BacktestEngine()
    result = engine.run(weights, features, regimes)

    assert "equity_curve" in result
    assert "returns" in result
    assert len(result["equity_curve"]) > 0


def test_equity_starts_at_capital():
    features = make_dummy_features()
    weights = make_dummy_weights(features)
    regimes = np.zeros(len(features["close"]), dtype=int)

    config = BacktestConfig(capital=50_000)
    engine = BacktestEngine(config)
    result = engine.run(weights, features, regimes)

    # First nonzero equity value should be close to capital
    eq = result["equity_curve"].dropna()
    assert abs(eq.iloc[0] - 50_000) / 50_000 < 0.05


def test_no_lookahead_bias():
    """
    Prove no lookahead: if we shift weights by 1, we should get the same result
    as the engine (which internally shifts by 1).
    """
    features = make_dummy_features()
    weights = make_dummy_weights(features)
    regimes = np.zeros(len(features["close"]), dtype=int)

    engine = BacktestEngine()
    result = engine.run(weights, features, regimes)

    # Manually compute expected returns with shift
    returns = features["returns"]
    expected = (weights.shift(1).fillna(0) * returns).sum(axis=1)
    actual_gross = result["gross_returns"]

    pd.testing.assert_series_equal(
        expected.reindex(actual_gross.index),
        actual_gross,
        check_names=False,
        atol=1e-10,
    )


def test_zero_weights_flat_equity():
    """Zero weights = no positions = flat equity (minus no costs)."""
    features = make_dummy_features()
    weights = pd.DataFrame(0.0, index=features["close"].index, columns=features["close"].columns)
    regimes = np.zeros(len(features["close"]), dtype=int)

    engine = BacktestEngine(BacktestConfig(capital=100_000))
    result = engine.run(weights, features, regimes)

    eq = result["equity_curve"].dropna()
    assert (eq == 100_000).all(), "Zero weights should produce flat equity"


def test_position_cap():
    """No position should exceed max_position_size."""
    features = make_dummy_features()
    weights = pd.DataFrame(0.5, index=features["close"].index, columns=features["close"].columns)
    regimes = np.zeros(len(features["close"]), dtype=int)

    config = BacktestConfig(max_position_size=0.20)
    engine = BacktestEngine(config)
    result = engine.run(weights, features, regimes)

    assert result["positions"].abs().max().max() <= 0.201   # small float tolerance
