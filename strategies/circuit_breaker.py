"""
strategies/circuit_breaker.py
------------------------------
Circuit Breaker — go completely flat in a regime.

Used as the High Vol strategy by default.
Returns zero weights for every day it is active.

The research finding: going flat in High Vol improved
total return from +6.3% to +10.0% and reduced max
drawdown from -24.3% to -23.2%.
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class CircuitBreakerStrategy(BaseStrategy):

    name = "circuit_breaker"
    description = "Go flat — capital preservation mode. Optimal in High Volatility regime."
    preferred_regimes = [1]  # High Vol
    default_params = {}

    def entry_signals(self, features: dict, regimes: np.ndarray) -> pd.DataFrame:
        """Always return zero — no positions."""
        close = features["close"]
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

    def exit_signals(self, features: dict, positions: pd.DataFrame) -> pd.DataFrame:
        """Always exit — close all positions immediately."""
        return pd.DataFrame(True, index=positions.index, columns=positions.columns)

    def position_size(self, signals: pd.DataFrame, features: dict, capital: float) -> pd.DataFrame:
        """Zero weight always."""
        return signals * 0.0