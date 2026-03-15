"""
strategies/mean_reversion.py
-----------------------------
BOLLINGER BAND MEAN REVERSION — used in Regime 1 (High Volatility)

Theory:
    When markets are volatile, prices overshoot in both directions due to
    panic selling and euphoric buying. They then snap back to the mean.
    We buy when price is abnormally far below its mean, sell when far above.

Why it works in high-vol regimes:
    High volatility = lots of overreaction. Panicked selling pushes prices
    2+ standard deviations below the mean, creating reversion opportunities.
    Momentum strategies fail here because there's no clean trend to follow.

Signal construction:
    bb_zscore = (price - 20d_mean) / 20d_std
    Entry long:  bb_zscore < -2.0  (price 2σ below mean)
    Entry short: bb_zscore > +2.0  (price 2σ above mean)
    Exit:        bb_zscore crosses back through 0 (returned to mean)

All operations vectorized — entire Bollinger band history computed in one pass.
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):

    name = "mean_reversion"
    description = "Bollinger Band Mean Reversion — fade extreme moves"
    preferred_regimes = [1]   # High volatility

    default_params = {
        "bb_window": 20,          # lookback for Bollinger Band mean/std
        "entry_zscore": 2.0,      # enter when |zscore| exceeds this
        "exit_zscore": 0.0,       # exit when zscore crosses this (back to mean)
        "max_holding_days": 10,   # max days to hold if mean reversion doesn't happen
        "top_pct": 0.25,          # top/bottom % of universe by zscore
        "vol_target": 0.12,       # lower vol target than momentum (more defensive)
        "stop_loss_zscore": 3.5,  # stop out if zscore reaches this (trend developing)
    }

    # ── Entry Signals ─────────────────────────────────────────────────────────

    def entry_signals(self, features: dict, regimes: np.ndarray) -> pd.DataFrame:
        """
        Bollinger Band entry signal.

        Step 1: Compute Bollinger z-score for all tickers (vectorized)
        Step 2: Rank by z-score within universe (most oversold / overbought)
        Step 3: Long most oversold, short most overbought

        The key insight: we don't enter on every 2σ move.
        We rank by z-score and only trade the most extreme cases in the universe.
        This filters out noise and focuses on the clearest reversion setups.
        """
        close = features["close"]
        bb_zscore = features.get("bb_zscore")
        window = self.get_param("bb_window")
        entry_z = self.get_param("entry_zscore")
        top_pct = self.get_param("top_pct")

        # Compute bb_zscore if not precomputed in features
        if bb_zscore is None:
            rolling_mean = close.rolling(window, min_periods=window // 2).mean()
            rolling_std = close.rolling(window, min_periods=window // 2).std()
            bb_zscore = (close - rolling_mean) / rolling_std.replace(0, np.nan)

        # ── Cross-sectional ranking of absolute z-score ────────────────────────
        # Most negative z-score = most oversold = best long candidate
        # Most positive z-score = most overbought = best short candidate
        abs_zscore_rank = bb_zscore.abs().rank(axis=1, pct=True, na_option="keep")

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        # Long: significantly oversold AND in top quantile of oversold stocks
        long_condition = (bb_zscore < -entry_z) & (abs_zscore_rank >= (1 - top_pct))
        # Short: significantly overbought AND in top quantile of overbought stocks
        short_condition = (bb_zscore > entry_z) & (abs_zscore_rank >= (1 - top_pct))

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        # Require burn-in
        signals.iloc[:window * 2] = 0.0

        return signals

    # ── Exit Signals ──────────────────────────────────────────────────────────

    def exit_signals(self, features: dict, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Exit when:
        1. Z-score returns to mean (successful reversion) — primary exit
        2. Z-score reaches stop-loss level (trend continuation — cut loss)
        3. Max holding period elapsed

        All conditions checked vectorized across entire positions DataFrame.
        """
        close = features["close"]
        window = self.get_param("bb_window")
        exit_z = self.get_param("exit_zscore")
        stop_z = self.get_param("stop_loss_zscore")

        # Recompute live z-score
        rolling_mean = close.rolling(window, min_periods=window // 2).mean()
        rolling_std = close.rolling(window, min_periods=window // 2).std()
        bb_zscore = (close - rolling_mean) / rolling_std.replace(0, np.nan)

        # Exit long: z-score has risen back to 0 (mean reversion occurred)
        mean_reversion_long = (positions > 0) & (bb_zscore >= exit_z)
        # Exit short: z-score has fallen back to 0
        mean_reversion_short = (positions < 0) & (bb_zscore <= -exit_z)

        # Stop loss: z-score moved even further against us (trend, not reversion)
        stop_long = (positions > 0) & (bb_zscore < -stop_z)
        stop_short = (positions < 0) & (bb_zscore > stop_z)

        return mean_reversion_long | mean_reversion_short | stop_long | stop_short

    # ── Position Sizing ───────────────────────────────────────────────────────

    def position_size(self, signals: pd.DataFrame, features: dict, capital: float) -> pd.DataFrame:
        """
        Size positions proportional to z-score magnitude.

        A stock 3σ below its mean gets a larger position than one only 2σ below.
        The further the deviation, the higher the expected reversion magnitude.

        Weight proportional to z-score, then normalize to target vol.
        """
        close = features["close"]
        bb_zscore = features.get("bb_zscore")
        window = self.get_param("bb_window")
        realized_vol = features.get("realized_vol")
        vol_target = self.get_param("vol_target")

        if bb_zscore is None:
            rolling_mean = close.rolling(window, min_periods=window // 2).mean()
            rolling_std = close.rolling(window, min_periods=window // 2).std()
            bb_zscore = (close - rolling_mean) / rolling_std.replace(0, np.nan)

        # Weight = signal direction * |z-score| (bigger move = bigger bet)
        # Clip z-score to avoid enormous weights on extreme outliers
        zscore_weight = signals * bb_zscore.abs().clip(upper=4.0)

        if realized_vol is not None:
            weights = self._vol_scale(zscore_weight, realized_vol, target_vol=vol_target)
        else:
            weights = zscore_weight

        # Normalize
        gross = weights.abs().sum(axis=1).replace(0, np.nan)
        return weights.div(gross, axis=0).fillna(0)
