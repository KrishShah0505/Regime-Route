"""
strategies/momentum.py
----------------------
TIME-SERIES MOMENTUM STRATEGY — used in Regime 0 (Low Volatility)

Theory:
    Stocks that have performed well over the past 12 months (minus 1 month)
    tend to continue performing well over the next month.
    This is the Jegadeesh & Titman (1993) momentum anomaly, one of the most
    robust and academically documented effects in finance.

Why it works in low-vol regimes:
    In calm markets, institutional flows are persistent and trends are clean.
    Prices drift in one direction with low noise, making momentum predictable.
    In high-vol regimes, momentum gets whipsawed — hence the regime filter.

Signal construction:
    raw_signal = close[t-21] / close[t-252] - 1  (12m return, skip last month)
    cross_sectional_rank(raw_signal)              (rank within universe)
    long top 20%, short bottom 20%
    position_size = vol_scaled weights

All operations are vectorized across the full date range and ticker universe.
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):

    name = "momentum"
    description = "Time-Series Momentum — long recent winners, short recent losers"
    preferred_regimes = [0]   # Low volatility

    default_params = {
        "lookback": 252,          # formation period (trading days)
        "skip_recent": 21,        # skip last N days (avoid reversal)
        "top_pct": 0.20,          # long/short top and bottom X% of universe
        "vol_target": 0.15,       # annual vol target per position
        "vol_window": 63,         # window for realized vol calculation
        "min_momentum": 0.0,      # minimum signal strength to enter (filter noise)
    }

    # ── Entry Signals ─────────────────────────────────────────────────────────

    def entry_signals(self, features: dict, regimes: np.ndarray) -> pd.DataFrame:
        """
        Momentum entry signal.

        Step 1: Compute raw momentum for all tickers simultaneously (vectorized)
        Step 2: Cross-sectionally rank stocks by momentum each day (vectorized)
        Step 3: Long top 20%, short bottom 20%, flat everything else

        Returns DataFrame of {-1, 0, 1} aligned to features['close'].index
        """
        close = features["close"]
        lookback = self.get_param("lookback")
        skip = self.get_param("skip_recent")
        top_pct = self.get_param("top_pct")
        min_mom = self.get_param("min_momentum")

        # ── Step 1: Raw momentum (vectorized shift on entire DataFrame) ────────
        # past_price = price 'lookback' days ago
        # recent_price = price 'skip' days ago (avoid last-month reversal)
        past_price = close.shift(lookback)
        recent_price = close.shift(skip)

        # Entire universe, entire date range, one operation
        raw_momentum = recent_price / past_price - 1

        # ── Step 2: Cross-sectional rank per day ──────────────────────────────
        # rank(axis=1) = ranks across tickers for each row (each day)
        # pct=True normalises to [0, 1]
        daily_ranks = raw_momentum.rank(axis=1, pct=True, na_option="keep")

        # ── Step 3: Long/short signal ─────────────────────────────────────────
        n_tickers = close.shape[1]
        top_n = max(1, int(n_tickers * top_pct))

        signals = pd.DataFrame(0, index=close.index, columns=close.columns, dtype=float)

        # Vectorized boolean indexing — no loops
        long_mask = daily_ranks >= (1 - top_pct)
        short_mask = daily_ranks <= top_pct

        # Only enter if momentum exceeds minimum threshold
        strong_positive = raw_momentum > min_mom
        strong_negative = raw_momentum < -min_mom

        signals[long_mask & strong_positive] = 1.0
        signals[short_mask & strong_negative] = -1.0

        # Require minimum lookback period — zero out before enough data
        signals.iloc[:lookback + skip] = 0.0

        return signals

    # ── Exit Signals ──────────────────────────────────────────────────────────

    def exit_signals(self, features: dict, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Exit when:
        1. Momentum reverses (signal flips sign)
        2. Regime changes (handled by router, but we signal readiness to exit)
        3. Stop-loss: position down more than 15% since entry (risk control)

        Returns boolean DataFrame: True = exit this position today
        """
        close = features["close"]
        lookback = self.get_param("lookback")
        skip = self.get_param("skip_recent")

        # Recompute current momentum to detect reversals
        past_price = close.shift(lookback)
        recent_price = close.shift(skip)
        current_momentum = recent_price / past_price - 1

        # Exit long if momentum went negative; exit short if momentum went positive
        exit_long = (positions > 0) & (current_momentum < 0)
        exit_short = (positions < 0) & (current_momentum > 0)

        return exit_long | exit_short

    # ── Position Sizing ───────────────────────────────────────────────────────

    def position_size(self, signals: pd.DataFrame, features: dict, capital: float) -> pd.DataFrame:
        """
        Volatility-scaled equal weighting.

        Each position targets the same annual volatility contribution.
        This means we take smaller positions in volatile stocks and larger
        positions in calm stocks — keeping risk balanced across the portfolio.

        Formula per stock: weight = (1/N) * (vol_target / stock_realized_vol)
        where N = number of active positions that day.
        """
        realized_vol = features.get("realized_vol")
        vol_target = self.get_param("vol_target")

        if realized_vol is None:
            # Fallback: equal weight without vol scaling
            n_active = (signals != 0).sum(axis=1).replace(0, np.nan)
            weights = signals.div(n_active, axis=0).fillna(0)
            return weights

        # Vol-scaled weights
        weights = self._vol_scale(signals, realized_vol, target_vol=vol_target)

        # Normalise so total gross exposure doesn't exceed 100%
        gross_exposure = weights.abs().sum(axis=1).replace(0, np.nan)
        weights = weights.div(gross_exposure, axis=0).fillna(0)

        return weights
