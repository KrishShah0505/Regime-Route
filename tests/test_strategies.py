"""tests/test_strategies.py"""
import pytest
import numpy as np
import pandas as pd
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.trend_filter import TrendFilterStrategy


def make_features(n_days=600, n_tickers=10):
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    tickers = [f"S{i}" for i in range(n_tickers)]
    np.random.seed(0)
    prices = pd.DataFrame(
        (1 + np.random.randn(n_days, n_tickers) * 0.012).cumprod(axis=0) * 50,
        index=dates, columns=tickers,
    )
    returns = prices.pct_change()
    log_ret = np.log(prices / prices.shift(1))
    rv = log_ret.rolling(20).std() * np.sqrt(252)
    rolling_mean = prices.rolling(20).mean()
    rolling_std = prices.rolling(20).std()
    bb_zscore = (prices - rolling_mean) / rolling_std

    return {
        "close": prices,
        "returns": returns,
        "realized_vol": rv,
        "bb_zscore": bb_zscore,
        "volume": pd.DataFrame(1e6, index=dates, columns=tickers),
    }


def test_momentum_signal_shape():
    features = make_features()
    regimes = np.zeros(len(features["close"]), dtype=int)
    s = MomentumStrategy()
    signals = s.entry_signals(features, regimes)
    assert signals.shape == features["close"].shape
    assert signals.isin([-1.0, 0.0, 1.0]).all().all()


def test_momentum_no_early_signals():
    """No signals in the burn-in period."""
    features = make_features()
    regimes = np.zeros(len(features["close"]), dtype=int)
    s = MomentumStrategy()
    signals = s.entry_signals(features, regimes)
    assert (signals.iloc[:252] == 0).all().all()


def test_mean_reversion_signal_shape():
    features = make_features()
    regimes = np.ones(len(features["close"]), dtype=int)
    s = MeanReversionStrategy()
    signals = s.entry_signals(features, regimes)
    assert signals.shape == features["close"].shape


def test_trend_filter_signal_shape():
    features = make_features()
    regimes = np.full(len(features["close"]), 2, dtype=int)
    s = TrendFilterStrategy()
    signals = s.entry_signals(features, regimes)
    assert signals.shape == features["close"].shape


def test_position_sizes_bounded():
    """Position sizes should be between -1 and 1."""
    features = make_features()
    regimes = np.zeros(len(features["close"]), dtype=int)
    s = MomentumStrategy()
    signals = s.entry_signals(features, regimes)
    weights = s.position_size(signals, features, capital=100_000)
    assert (weights.abs() <= 1.0 + 1e-9).all().all()
