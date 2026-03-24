"""
strategies/breakout.py
-----------------------
Volatility Breakout Strategy.

Logic:
    BUY  when price breaks above N-day high with volume confirmation
    SELL when price breaks below N-day low with volume confirmation

Why it works:
    Breakouts signal the start of new trends. In transitional regimes
    where direction is unclear, a confirmed breakout gives early
    trend entry with defined risk.

Volume confirmation filters false breakouts — a breakout on low
volume is likely noise, a breakout on high volume is conviction.

All operations fully vectorized across all tickers simultaneously.
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class BreakoutStrategy(BaseStrategy):

    name = "breakout"
    description = "Volatility breakout with volume confirmation"
    preferred_regimes = [2]  # Transitional
    default_params = {
        "window":          20,    # lookback for high/low
        "volume_threshold": 1.5,  # volume must be X times average
        "top_pct":         0.3,   # fraction of universe to trade
    }

    def entry_signals(self, features: dict, regimes: np.ndarray) -> pd.DataFrame:
        """
        Detect breakouts with volume confirmation.

        Long:  close > N-day high (shifted 1 to avoid lookahead)
               AND volume > volume_threshold * avg_volume

        Short: close < N-day low (shifted 1 to avoid lookahead)
               AND volume > volume_threshold * avg_volume
        """
        close  = features["close"]
        volume = features["volume"]
        window = self.get_param("window")
        vol_thresh = self.get_param("volume_threshold")

        # Rolling high/low — shift by 1 to exclude current bar
        rolling_high = close.shift(1).rolling(window).max()
        rolling_low  = close.shift(1).rolling(window).min()

        # Volume confirmation — is today's volume above average?
        avg_volume   = volume.rolling(window).mean()
        high_volume  = volume > (vol_thresh * avg_volume)

        # Breakout signals
        long_breakout  = (close > rolling_high) & high_volume
        short_breakout = (close < rolling_low)  & high_volume

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        signals[long_breakout]  =  1.0
        signals[short_breakout] = -1.0

        return signals

    def exit_signals(self, features: dict, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Exit when price reverts back inside the range.
        Long exits when price drops below rolling midpoint.
        Short exits when price rises above rolling midpoint.
        """
        close  = features["close"]
        window = self.get_param("window")

        rolling_high = close.shift(1).rolling(window).max()
        rolling_low  = close.shift(1).rolling(window).min()
        midpoint     = (rolling_high + rolling_low) / 2

        long_exit  = (positions > 0) & (close < midpoint)
        short_exit = (positions < 0) & (close > midpoint)

        return long_exit | short_exit

    def position_size(
        self,
        signals: pd.DataFrame,
        features: dict,
        capital: float,
    ) -> pd.DataFrame:
        """Equal weight among breakout signals."""
        top_pct = self.get_param("top_pct")
        return self._equal_weight_universe(signals, top_pct=top_pct)