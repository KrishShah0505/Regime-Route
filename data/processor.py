"""
data/processor.py
-----------------
Feature engineering on raw OHLCV data.
All transformations are fully vectorized using Pandas/NumPy.
No iterrows(), no apply() with Python loops.

Features produced:
    - Realized volatility (rolling)
    - Log returns
    - Rolling momentum signals
    - Volume ratio (abnormal volume detection)
    - VIX z-score (for regime features)
"""

import pandas as pd
import numpy as np


# ── Return Engineering ────────────────────────────────────────────────────────

def compute_log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns: ln(P_t / P_{t-1})
    More statistically well-behaved than simple returns for modelling.
    Fully vectorized via np.log and shift.
    """
    return np.log(close / close.shift(1))


def compute_simple_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Simple percentage returns. Used for P&L calculation."""
    return close.pct_change()


# ── Volatility Features ───────────────────────────────────────────────────────

def compute_realized_vol(
    returns: pd.DataFrame,
    window: int = 20,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Rolling realized volatility = rolling std of returns.
    Annualized by default (multiply by sqrt(252)).

    Vectorized: .rolling().std() operates on entire DataFrame at once.
    """
    rv = returns.rolling(window=window, min_periods=window // 2).std()
    if annualize:
        rv = rv * np.sqrt(252)
    return rv


def compute_vol_ratio(
    realized_vol: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 63,
) -> pd.DataFrame:
    """
    Short-term vol / long-term vol ratio.
    Ratio > 1 = vol is rising (regime transition signal).
    Ratio < 1 = vol is compressing (calm regime).

    Both rolling windows computed vectorized simultaneously.
    """
    short_vol = realized_vol.rolling(short_window, min_periods=2).mean()
    long_vol = realized_vol.rolling(long_window, min_periods=10).mean()
    return short_vol / long_vol.replace(0, np.nan)


# ── Momentum Features ─────────────────────────────────────────────────────────

def compute_momentum(
    close: pd.DataFrame,
    lookback: int = 252,
    skip_recent: int = 21,
) -> pd.DataFrame:
    """
    Time-series momentum signal: return over lookback period, skipping
    the most recent 'skip_recent' days to avoid short-term reversal.

    Formula: close[t - skip_recent] / close[t - lookback] - 1

    This is the standard academic momentum factor (Jegadeesh & Titman).
    Vectorized shift operations — no loops.
    """
    # Price lookback days ago
    past_price = close.shift(lookback)
    # Price skip_recent days ago (avoid last month reversal)
    recent_price = close.shift(skip_recent)

    momentum = recent_price / past_price - 1
    return momentum


def compute_short_momentum(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Short-term momentum for mean reversion detection (inverse signal)."""
    return close.pct_change(window)


# ── Volume Features ───────────────────────────────────────────────────────────

def compute_volume_ratio(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Volume ratio: today's volume / rolling average volume.
    > 1.5 = abnormally high volume (potential breakout or reversal).
    Vectorized rolling mean across all tickers simultaneously.
    """
    avg_volume = volume.rolling(window=window, min_periods=5).mean()
    return volume / avg_volume.replace(0, np.nan)


# ── VIX Features ──────────────────────────────────────────────────────────────

def compute_vix_zscore(vix: pd.Series, window: int = 63) -> pd.Series:
    """
    Rolling z-score of VIX.
    z > 1.5 = high vol regime
    z < -0.5 = low vol regime
    between = transitional

    Used as fallback regime classifier and as HMM input feature.
    """
    rolling_mean = vix.rolling(window=window, min_periods=10).mean()
    rolling_std = vix.rolling(window=window, min_periods=10).std()
    return (vix - rolling_mean) / rolling_std.replace(0, np.nan)


def compute_vix_percentile(vix: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling percentile rank of VIX over past 'window' days.
    More stable than z-score for labelling absolute vol levels.
    0 = historically low vol, 1 = historically high vol.
    """
    return vix.rolling(window=window, min_periods=50).rank(pct=True)


# ── Moving Averages ───────────────────────────────────────────────────────────

def compute_ema(close: pd.DataFrame, span: int) -> pd.DataFrame:
    """Exponential moving average. Vectorized via pandas ewm."""
    return close.ewm(span=span, adjust=False).mean()


def compute_sma(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple moving average. Vectorized via rolling mean."""
    return close.rolling(window=window, min_periods=window // 2).mean()


def compute_bollinger_bands(
    close: pd.DataFrame,
    window: int = 20,
    n_std: float = 2.0,
) -> tuple:
    """
    Bollinger Bands: (upper, middle, lower)
    All three are full DataFrames, vectorized.

    Returns
    -------
    tuple: (upper_band, middle_band, lower_band)
    """
    middle = close.rolling(window=window, min_periods=window // 2).mean()
    std = close.rolling(window=window, min_periods=window // 2).std()
    upper = middle + n_std * std
    lower = middle - n_std * std
    return upper, middle, lower


def compute_bollinger_zscore(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    (Price - Rolling Mean) / Rolling Std
    Positive = price above mean, Negative = price below mean.
    Used as the raw mean-reversion signal.
    """
    middle = close.rolling(window=window, min_periods=window // 2).mean()
    std = close.rolling(window=window, min_periods=window // 2).std()
    return (close - middle) / std.replace(0, np.nan)


# ── Master Feature Builder ────────────────────────────────────────────────────

def build_all_features(data: dict) -> dict:
    """
    Run all feature engineering steps on raw fetched data.
    Call this after fetch_universe_data().

    Parameters
    ----------
    data : dict from fetcher.fetch_universe_data()

    Returns
    -------
    dict with all original keys plus:
        'log_returns', 'realized_vol', 'vol_ratio',
        'momentum', 'short_momentum', 'volume_ratio',
        'vix_zscore', 'vix_percentile',
        'ema_50', 'ema_200', 'sma_200',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_zscore'
    """
    close = data["close"]
    volume = data["volume"]
    returns = data["returns"]
    vix = data["vix"]

    log_returns = compute_log_returns(close)
    realized_vol = compute_realized_vol(log_returns, window=20)

    features = {
        **data,
        "log_returns": log_returns,
        "realized_vol": realized_vol,
        "vol_ratio": compute_vol_ratio(realized_vol),
        "momentum": compute_momentum(close, lookback=252, skip_recent=21),
        "short_momentum": compute_short_momentum(close, window=20),
        "volume_ratio": compute_volume_ratio(volume),
        "vix_zscore": compute_vix_zscore(vix),
        "vix_percentile": compute_vix_percentile(vix),
        "ema_50": compute_ema(close, span=50),
        "ema_200": compute_ema(close, span=200),
        "sma_200": compute_sma(close, window=200),
    }

    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(close)
    features["bb_upper"] = bb_upper
    features["bb_middle"] = bb_middle
    features["bb_lower"] = bb_lower
    features["bb_zscore"] = compute_bollinger_zscore(close)

    return features
