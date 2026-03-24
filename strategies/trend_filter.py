"""
strategies/trend_filter.py
--------------------------
TREND FILTER WITH REDUCED EXPOSURE — used in Regime 2 (Transitional)

Theory:
    Transitional regimes are uncertain — the market is shifting between calm
    and chaos. Neither momentum nor mean reversion reliably works here.
    The defensive play: only trade in the direction of the long-term trend
    and cut position sizes in half to limit damage from false signals.

Why reduced exposure:
    False signals in transitional regimes are expensive. Better to be small
    and wrong than large and wrong. This is classic risk management.

Signal construction:
    Trend filter: price vs 200-day EMA (above = uptrend, below = downtrend)
    Signal: long if price > EMA200 AND recent 20d momentum is positive
            short if price < EMA200 AND recent 20d momentum is negative
    Position size: 50% of what momentum strategy would take

All operations vectorized.
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class TrendFilterStrategy(BaseStrategy):

    name = "trend_filter"
    description = "Trend Filter — long-term direction only, reduced exposure"
    preferred_regimes = [2]   # Transitional

    default_params = {
        "ema_long": 200,          # long-term trend filter (EMA)
        "ema_short": 50,          # medium-term trend confirmation
        "momentum_window": 20,    # short-term momentum confirmation
        "size_scalar": 0.5,       # position size multiplier (50% of normal)
        "vol_target": 0.10,       # conservative vol target
        "top_pct": 0.25,          # universe top/bottom % to trade
    }

    # ── Entry Signals ─────────────────────────────────────────────────────────

    def entry_signals(self, features: dict, regimes: np.ndarray) -> pd.DataFrame:
        """
        Trend-filtered entry signal.

        Three conditions must all align for an entry:
        1. Price above/below 200-day EMA (macro trend direction)
        2. Price above/below 50-day EMA (medium trend confirmation)
        3. Positive/negative 20-day momentum (recent direction matches trend)

        When all three agree: clear directional bias → take a small position.
        When they disagree: conflicting signals → stay flat.

        This triple confirmation eliminates most false signals in choppy markets.
        """
        close = features["close"]
        ema_long = self.get_param("ema_long")
        ema_short = self.get_param("ema_short")
        mom_window = self.get_param("momentum_window")
        top_pct = self.get_param("top_pct")

        # EMA200 and EMA50 — vectorized ewm across all tickers simultaneously
        ema200 = close.ewm(span=ema_long, adjust=False).mean()
        ema50 = close.ewm(span=ema_short, adjust=False).mean()

        # Short-term momentum — vectorized pct_change
        momentum_20 = close.pct_change(mom_window)

        # ── Triple filter conditions ──────────────────────────────────────────
        # Long: price above both EMAs AND recent momentum positive
        long_trend = (close > ema200) & (close > ema50)
        long_momentum = momentum_20 > 0
        long_signal = long_trend & long_momentum

        # Short: price below both EMAs AND recent momentum negative
        short_trend = (close < ema200) & (close < ema50)
        short_momentum = momentum_20 < 0
        short_signal = short_trend & short_momentum

        # Rank by strength of trend (distance from EMA200) within universe
        trend_strength = ((close - ema200) / ema200).abs()
        trend_rank = trend_strength.rank(axis=1, pct=True, na_option="keep")

        # Only trade the clearest trends (top quartile by strength)
        strong_trend = trend_rank >= (1 - top_pct)

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        signals[long_signal & strong_trend] = 1.0
        signals[short_signal & strong_trend] = -1.0

        # Require burn-in for EMA200
        signals.iloc[:ema_long + 10] = 0.0

        return signals

    # ── Exit Signals ──────────────────────────────────────────────────────────

    def exit_signals(self, features: dict, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Exit when trend breaks:
        - Long exit: price crosses back below EMA50 (trend weakening)
        - Short exit: price crosses back above EMA50

        Quicker exit than the trend filter entry to protect capital fast.
        """
        close = features["close"]
        ema_short = self.get_param("ema_short")

        ema50 = close.ewm(span=ema_short, adjust=False).mean()

        # Trend broken for longs
        exit_long = (positions > 0) & (close < ema50)
        # Trend broken for shorts
        exit_short = (positions < 0) & (close > ema50)

        return exit_long | exit_short

    # ── Position Sizing ───────────────────────────────────────────────────────

    def position_size(self, signals: pd.DataFrame, features: dict, capital: float) -> pd.DataFrame:
        """
        Reduced-size vol-targeted positions.

        Takes the vol-scaled weights from the parent class and then
        applies the size_scalar (default 0.5) to halve all positions.
        This is the core risk management feature of the transitional regime.
        """
        realized_vol = features.get("realized_vol")
        vol_target = self.get_param("vol_target")
        scalar = self.get_param("size_scalar")

        if realized_vol is not None:
            weights = self._vol_scale(signals, realized_vol, target_vol=vol_target)
        else:
            weights = signals.copy()

        # Normalize
        gross = weights.abs().sum(axis=1).replace(0, np.nan)
        weights = weights.div(gross, axis=0).fillna(0)

        # Apply size reduction — the defining feature of this regime's strategy
        weights = weights * scalar

        return weights


