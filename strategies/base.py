"""
strategies/base.py
------------------
Abstract base class for all trading strategies.

DESIGN PRINCIPLE — Open/Closed:
    This file never changes. Adding a new strategy = create a new file,
    inherit from BaseStrategy, implement 3 methods, register in router.py.
    Zero changes to existing code.

VECTORIZATION CONTRACT:
    All three abstract methods MUST return pandas objects aligned to
    data.index. No loops over rows. The backtester assumes vectorized outputs.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    Abstract base class for all QuantRegime strategies.

    Every strategy must implement:
        entry_signals()  : when to open a position
        exit_signals()   : when to close a position
        position_size()  : how much capital to allocate

    Strategies receive the full feature dictionary and regime array,
    but they should ONLY use data up to time T when computing signal for T.
    The backtester enforces the 1-day execution lag via .shift(1).
    """

    # Subclasses set these class attributes
    name: str = "base"
    description: str = ""
    preferred_regimes: list = []          # which regimes this strategy targets
    default_params: dict = {}             # default hyperparameters

    def __init__(self, params: dict = None):
        """
        Parameters
        ----------
        params : override default_params. E.g. {'lookback': 180, 'top_pct': 0.2}
        """
        self.params = {**self.default_params, **(params or {})}

    # ── Abstract Methods (must implement) ─────────────────────────────────────

    @abstractmethod
    def entry_signals(
        self,
        features: dict,
        regimes: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute entry signals for all tickers across all dates.

        Returns
        -------
        pd.DataFrame
            shape: (n_days, n_tickers)
            values: 1 (long), -1 (short), 0 (flat/no trade)
            index: same DatetimeIndex as features['close']

        Rules:
            - NEVER look ahead. Signal on day T uses only data through T.
            - The backtester applies .shift(1) before execution. You don't.
            - Vectorized only. No iterrows(). No apply() with Python loops.
        """
        pass

    @abstractmethod
    def exit_signals(
        self,
        features: dict,
        positions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute exit signals — when to close an open position.

        Returns
        -------
        pd.DataFrame
            shape: (n_days, n_tickers)
            values: True = exit this position today, False = hold
        """
        pass

    @abstractmethod
    def position_size(
        self,
        signals: pd.DataFrame,
        features: dict,
        capital: float,
    ) -> pd.DataFrame:
        """
        Scale raw signals into position weights.

        Returns
        -------
        pd.DataFrame
            shape: (n_days, n_tickers)
            values: float weights, typically between -1.0 and 1.0
                    where 1.0 = 100% of capital in that stock

        Common approaches:
            - Equal weight: just pass signals through (each position gets 1/n)
            - Vol scaling: signals / realized_vol (equalises risk per position)
            - Kelly: signals * edge / variance
        """
        pass

    # ── Shared Utilities (available to all subclasses) ─────────────────────────

    def get_param(self, key: str):
        return self.params.get(key, self.default_params.get(key))

    def _cross_sectional_rank(self, signal: pd.DataFrame) -> pd.DataFrame:
        """
        Rank tickers by signal strength on each day (cross-sectional).
        Returns percentile ranks: 0 = weakest, 1 = strongest signal.

        Vectorized: .rank(axis=1, pct=True) operates on all rows at once.
        """
        return signal.rank(axis=1, pct=True, na_option="keep")

    def _vol_scale(
        self,
        signals: pd.DataFrame,
        realized_vol: pd.DataFrame,
        target_vol: float = 0.15,
    ) -> pd.DataFrame:
        """
        Scale position sizes so each position targets the same annual volatility.
        This keeps risk constant regardless of how volatile a stock is.

        Formula: weight = signal * (target_vol / stock_realized_vol)

        Vectorized: pure DataFrame division.
        """
        # Avoid division by zero
        rv = realized_vol.replace(0, np.nan).reindex(signals.index)
        scaling = target_vol / rv
        # Cap scaling to avoid huge positions in very low-vol stocks
        scaling = scaling.clip(upper=3.0)
        return signals * scaling

    def _equal_weight_universe(self, signals: pd.DataFrame, top_pct: float = 0.2) -> pd.DataFrame:
        """
        For cross-sectional strategies: take top X% long, bottom X% short.
        Assigns equal weight within long and short books.

        Vectorized: rank → threshold → sign → normalize, all at DataFrame level.
        """
        ranks = self._cross_sectional_rank(signals.abs() * np.sign(signals))
        n = signals.shape[1]
        top_n = max(1, int(n * top_pct))

        # Long: top top_pct of positive signals
        # Short: bottom top_pct of negative signals
        long_threshold = 1 - top_pct
        short_threshold = top_pct

        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        weights[ranks >= long_threshold] = 1.0 / top_n
        weights[ranks <= short_threshold] = -1.0 / top_n

        return weights

    def __repr__(self):
        return f"{self.__class__.__name__}(params={self.params})"
