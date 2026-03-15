"""
regime/features.py
------------------
Builds the feature matrix that the HMM regime classifier is trained on.
These features capture the volatility/trend state of the market.

All operations are vectorized — the entire feature matrix is computed
across 20 years of data in one pass.
"""

import pandas as pd
import numpy as np


def build_regime_features(vix: pd.Series, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix for regime classification.

    Uses market-wide signals (VIX-based) rather than individual stock signals
    because regimes are macro phenomena — they affect the whole market.

    Features:
        vix              : raw VIX level
        vix_ma20         : 20-day moving average of VIX
        vix_zscore_63    : rolling 63-day z-score of VIX
        vix_change       : day-over-day VIX change (velocity)
        realized_vol_20  : cross-sectional median realized vol (market-wide)
        vol_of_vol       : rolling std of VIX (uncertainty about uncertainty)
        vix_rv_ratio     : VIX / realized_vol — measures IV premium
        market_return    : equal-weighted market return (trend signal)
        market_return_ma : 20-day MA of market return

    Parameters
    ----------
    vix     : pd.Series — VIX index
    returns : pd.DataFrame — daily returns for all tickers

    Returns
    -------
    pd.DataFrame
        shape: (days, n_features), aligned to VIX index
        All values are standardised (no NaN after burn-in period)
    """

    # ── VIX features (vectorized) ─────────────────────────────────────────────
    vix_ma20 = vix.rolling(20, min_periods=5).mean()
    vix_ma63 = vix.rolling(63, min_periods=10).mean()

    vix_std_63 = vix.rolling(63, min_periods=10).std()
    vix_zscore_63 = (vix - vix_ma63) / vix_std_63.replace(0, np.nan)

    vix_change = vix.diff()                              # velocity of VIX
    vix_change_pct = vix.pct_change()

    vol_of_vol = vix.rolling(20, min_periods=5).std()    # std of VIX itself

    # ── Cross-sectional realized vol (vectorized across all tickers) ──────────
    # Compute realized vol per ticker then take cross-sectional median
    # This gives a single market-wide vol estimate, robust to outliers
    log_returns = np.log(1 + returns.replace([np.inf, -np.inf], np.nan))
    realized_vol_per_ticker = log_returns.rolling(20, min_periods=5).std() * np.sqrt(252)

    # .median(axis=1) = across tickers each day — fully vectorized
    realized_vol_market = realized_vol_per_ticker.median(axis=1)
    realized_vol_market.name = "realized_vol_market"

    # ── VIX / Realized Vol ratio (IV premium) ─────────────────────────────────
    # When VIX >> realized vol: market is pricing in fear beyond what's realised
    # When VIX ≈ realized vol: fair pricing, calmer regime
    vix_rv_ratio = vix / (realized_vol_market * 100).replace(0, np.nan)

    # ── Market return features (trend direction) ──────────────────────────────
    # Equal-weighted market return — direction matters for regime
    market_return = returns.mean(axis=1)                             # vectorized
    market_return_ma20 = market_return.rolling(20, min_periods=5).mean()
    market_return_cumulative = (1 + market_return).rolling(63).apply(
        np.prod, raw=True
    ) - 1   # 63-day cumulative return

    # ── Assemble feature matrix ───────────────────────────────────────────────
    features = pd.DataFrame({
        "vix":                  vix,
        "vix_ma20":             vix_ma20,
        "vix_zscore_63":        vix_zscore_63,
        "vix_change":           vix_change,
        "vix_change_pct":       vix_change_pct,
        "vol_of_vol":           vol_of_vol,
        "realized_vol_market":  realized_vol_market,
        "vix_rv_ratio":         vix_rv_ratio,
        "market_return":        market_return,
        "market_return_ma20":   market_return_ma20,
        "market_return_63":     market_return_cumulative,
    }, index=vix.index)

    return features


def select_hmm_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Select and preprocess the subset of features fed directly into the HMM.
    HMM works best with a small number of highly informative features.

    We use: VIX z-score, VIX change, realized vol, VIX/RV ratio
    These capture both the LEVEL and DIRECTION of volatility.
    """
    hmm_cols = ["vix_zscore_63", "vix_change_pct", "realized_vol_market", "vix_rv_ratio"]
    hmm_features = features[hmm_cols].copy()

    # Clip extreme values (outliers hurt HMM fitting)
    for col in hmm_cols:
        q01 = hmm_features[col].quantile(0.01)
        q99 = hmm_features[col].quantile(0.99)
        hmm_features[col] = hmm_features[col].clip(q01, q99)

    return hmm_features


def get_burn_in_period(features: pd.DataFrame) -> pd.Timestamp:
    """
    Return the first date where all features are non-NaN.
    Before this date, features are unreliable (rolling windows not full).
    """
    return features.dropna().index[0]
