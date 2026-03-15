"""
strategies/rsi_divergence.py
-----------------------------
RSI Divergence strategy for High Volatility regime.

Replaces Bollinger Band Mean Reversion which had Sharpe -0.908
in High Vol periods.

Logic:
    Bullish Divergence  → BUY
        Price: lower low over lookback window
        RSI:   higher low over same window
        → momentum turning up before price confirms

    Bearish Divergence  → SELL (short)
        Price: higher high over lookback window
        RSI:   lower high over same window
        → momentum fading before price confirms

Why this works better in High Vol:
    Bollinger Bands fade every 2σ move — in high vol this means
    fading genuine breakouts, which bleeds badly.
    RSI Divergence requires structural confirmation, filtering
    out noise and only trading genuine reversals.
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class RSIDivergenceStrategy(BaseStrategy):

    name = "rsi_divergence"
    description = "RSI Divergence mean reversion for High Volatility regime"
    preferred_regimes = [1]  # High Vol
    default_params = {
        "rsi_period":    14,
        "lookback":      15,
        "rsi_threshold": 35.0,   # stricter — only trade extreme divergences
        "top_pct":       0.2,
        "breakout_window": 20,    # fewer positions
    }

    # ── Abstract Method Implementations ───────────────────────────────────────

    def entry_signals(self, features: dict, regimes: np.ndarray) -> pd.DataFrame:
        """
        Volatility breakout — trade WITH momentum in High Vol.
        Buy when price breaks above 20-day high with RSI confirming.
        Short when price breaks below 20-day low with RSI confirming.
        """
        close    = features["close"]
        rsi      = self._compute_rsi(close)
        window   = self.get_param("breakout_window")
        thresh   = self.get_param("rsi_threshold")

        # Rolling high/low excluding current bar (shift 1 to avoid lookahead)
        rolling_high = close.shift(1).rolling(window).max()
        rolling_low  = close.shift(1).rolling(window).min()

        # Breakout conditions
        long_breakout  = (close > rolling_high) & (rsi > 50) & (rsi < 80)
        short_breakout = (close < rolling_low)  & (rsi < 50) & (rsi > 20)

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        signals[long_breakout]  =  1.0
        signals[short_breakout] = -1.0

        return signals

    def exit_signals(self, features: dict, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Exit when RSI crosses back to neutral (50 zone).
        Vectorized: compare RSI to neutral band.
        """
        close = features["close"]
        rsi   = self._compute_rsi(close)

        # Exit long when RSI > 50, exit short when RSI < 50
        long_exit  = (positions > 0) & (rsi > 50)
        short_exit = (positions < 0) & (rsi < 50)

        return long_exit | short_exit

    def position_size(
        self,
        signals: pd.DataFrame,
        features: dict,
        capital: float,
    ) -> pd.DataFrame:
        """
        Equal weight among divergence signals.
        Use cross-sectional ranking to select top signals only.
        """
        top_pct = self.get_param("top_pct")
        return self._equal_weight_universe(signals, top_pct=top_pct)

    # ── RSI Helper ────────────────────────────────────────────────────────────

    def _compute_rsi(self, close: pd.DataFrame) -> pd.DataFrame:
        """Vectorized Wilder's RSI across all tickers simultaneously."""
        period   = self.get_param("rsi_period")
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)