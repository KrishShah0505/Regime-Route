"""
strategies/router.py
--------------------
The Regime Router — the brain of QuantRegime.

Responsibilities:
    1. Map each regime label to a strategy instance
    2. Every day: look up today's regime → dispatch to correct strategy
    3. Combine signals from multiple strategies into a single signal array
    4. Support blended mode (weighted combination using HMM probabilities)

EXTENSIBILITY:
    Adding a new strategy = import it + add one line to STRATEGY_REGISTRY.
    The router and backtester require zero changes.

MODES:
    'hard'    : single winning strategy per regime (default)
    'blended' : weighted combination using HMM state probabilities (V2 feature)
"""

import numpy as np
import pandas as pd
import logging
from strategies.base import BaseStrategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.trend_filter import TrendFilterStrategy
from strategies.rsi_divergence import RSIDivergenceStrategy
from strategies.circuit_breaker import CircuitBreakerStrategy
from strategies.pairs_trading import PairsTradingStrategy
from strategies.breakout import BreakoutStrategy
logger = logging.getLogger(__name__)


# ── Strategy Registry ─────────────────────────────────────────────────────────
# Add new strategies here. One line. Router handles everything else.

STRATEGY_REGISTRY = {
    "momentum":       MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "trend_filter":   TrendFilterStrategy,
    # Future additions:
    "rsi_divergence":     RSIDivergenceStrategy,
    "circuit_breaker":CircuitBreakerStrategy,
     "breakout":           BreakoutStrategy,
     "pairs_trading":      PairsTradingStrategy,
}


# ── Default Regime → Strategy Mapping ────────────────────────────────────────

DEFAULT_REGIME_MAP = {
    0: "momentum",         # Low vol  → Momentum
    1: "circuit_breaker",   # High vol → Mean Reversion
    2: "trend_filter",     # Transit  → Trend Filter
}


# ── Router ────────────────────────────────────────────────────────────────────

class RegimeRouter:
    """
    Routes daily signals to the appropriate strategy based on detected regime.

    Parameters
    ----------
    regime_map    : dict mapping regime int → strategy name
                    defaults to DEFAULT_REGIME_MAP
    strategy_params : dict of {strategy_name: {param: value}} for overrides
    mode          : 'hard' (one strategy per regime) or 'blended' (weighted mix)
    """

    def __init__(
        self,
        regime_map: dict = None,
        strategy_params: dict = None,
        mode: str = "hard",
    ):
        self.regime_map = regime_map or DEFAULT_REGIME_MAP
        self.mode = mode
        self.strategy_params = strategy_params or {}

        # Instantiate all strategies referenced in regime_map
        self._strategies = self._build_strategies()

        logger.info(
            f"RegimeRouter initialized | mode={mode} | "
            f"mapping={self.regime_map}"
        )

    def _build_strategies(self) -> dict:
        """Instantiate strategy objects for every strategy in the regime map."""
        instances = {}
        for regime, strategy_name in self.regime_map.items():
            if strategy_name not in STRATEGY_REGISTRY:
                raise ValueError(
                    f"Unknown strategy '{strategy_name}'. "
                    f"Available: {list(STRATEGY_REGISTRY.keys())}"
                )
            params = self.strategy_params.get(strategy_name, {})
            instances[strategy_name] = STRATEGY_REGISTRY[strategy_name](params=params)
            logger.debug(f"Instantiated {strategy_name} for regime {regime}")
        return instances

    def get_strategy(self, regime: int) -> BaseStrategy:
        """Return the strategy instance for a given regime."""
        strategy_name = self.regime_map.get(regime, "trend_filter")
        return self._strategies[strategy_name]

    # ── Main Signal Generation ─────────────────────────────────────────────────

    def generate_signals(
        self,
        features: dict,
        regimes: np.ndarray,
    ) -> pd.DataFrame:
        """
        Generate the composite signal by routing each day to its strategy.

        In HARD mode (default):
            - Compute signals from ALL strategies (vectorized, fast)
            - For each day, select the signal from the strategy mapped to that day's regime
            - Selection is done via vectorized numpy indexing — no day-by-day loop

        Returns
        -------
        pd.DataFrame
            shape: (n_days, n_tickers)
            values: position weights (float, between -1 and 1)
        """
        if self.mode == "hard":
            return self._hard_route(features, regimes)
        elif self.mode == "blended":
            return self._blended_route(features, regimes)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

    def _hard_route(self, features: dict, regimes: np.ndarray) -> pd.DataFrame:
        """
        Hard routing: one strategy per regime.

        Implementation:
        1. Compute all strategy signals upfront (each is fully vectorized)
        2. Stack them into a 3D array (n_strategies x n_days x n_tickers)
        3. Use regime array to index into axis 0 — selects the right strategy per day
        4. No Python loop over days

        This is the vectorized approach — pure NumPy indexing, not iteration.
        """
        close = features["close"]
        n_days, n_tickers = close.shape
        index = close.index
        columns = close.columns

        # ── Step 1: Compute all strategy signals ──────────────────────────────
        all_signals = {}
        all_weights = {}

        unique_strategies = set(self.regime_map.values())
        for strategy_name in unique_strategies:
            strategy = self._strategies[strategy_name]
            logger.info(f"Computing signals for: {strategy_name}")

            raw = strategy.entry_signals(features, regimes)
            weights = strategy.position_size(raw, features, capital=100_000)

            all_signals[strategy_name] = raw
            all_weights[strategy_name] = weights

        # ── Step 2: Build strategy-index array ────────────────────────────────
        # ordered list of strategy names matching regime keys 0,1,2,...
        max_regime = max(self.regime_map.keys())
        strategy_order = [
            self.regime_map.get(r, "trend_filter") for r in range(max_regime + 1)
        ]

        # Stack weight arrays: shape (n_strategies, n_days, n_tickers)
        weight_stack = np.stack(
            [all_weights[s].values for s in strategy_order],
            axis=0
        )

        # ── Step 3: Vectorized selection via regime array ─────────────────────
        # regimes[i] gives the strategy index for day i
        # We use numpy advanced indexing to select from the stack
        clipped_regimes = np.clip(regimes, 0, max_regime).astype(int)
        # Shape: (n_days, n_tickers) — one row per day, selected from correct strategy
        selected = weight_stack[clipped_regimes, np.arange(n_days), :]

        composite_weights = pd.DataFrame(selected, index=index, columns=columns)

        # Log regime distribution in this run
        regime_counts = {
            self.regime_map.get(r, "?"): int((regimes == r).sum())
            for r in np.unique(regimes)
        }
       

        logger.info(f"Signal generation complete | Regime days: {regime_counts}")

        return composite_weights
        
    def _blended_route(self, features: dict, regimes: np.ndarray) -> pd.DataFrame:
        """
        Blended routing (V2): weight strategies by HMM state probabilities.
        For now, falls back to hard routing. Implement after HMM probabilities
        are exposed from the classifier.
        """
        logger.info("Blended mode not yet implemented, falling back to hard routing")
        return self._hard_route(features, regimes)

    # ── Introspection ─────────────────────────────────────────────────────────

    def list_strategies(self) -> dict:
        """Return metadata for all registered strategies."""
        return {
            name: {
                "class": cls.__name__,
                "description": cls.description,
                "preferred_regimes": cls.preferred_regimes,
                "default_params": cls.default_params,
            }
            for name, cls in STRATEGY_REGISTRY.items()
        }

    def get_regime_assignments(self) -> dict:
        """Return current regime → strategy mapping."""
        return {
            regime: self.regime_map[regime]
            for regime in sorted(self.regime_map.keys())
        }

    def update_regime_map(self, new_map: dict):
        """
        Hot-swap the regime → strategy mapping without restarting.
        Useful for the dashboard's manual override feature.
        """
        for regime, strategy_name in new_map.items():
            if strategy_name not in STRATEGY_REGISTRY:
                raise ValueError(f"Unknown strategy: {strategy_name}")
        self.regime_map = new_map
        self._strategies = self._build_strategies()
        logger.info(f"Regime map updated: {new_map}")
    def _apply_circuit_breaker(
        self,
        weights: pd.DataFrame,
        regimes: np.ndarray,
        high_vol_scale: float = 0.0,
    ) -> pd.DataFrame:
        """
        Drawdown circuit breaker — reduce position size in High Vol regime.

        In High Vol (regime 1), scale all positions down to high_vol_scale
        of their original size. Default 25% — still in the market but
        heavily defensive.

        This is the correct response to systemic market stress:
        no strategy performs well in genuine crashes, so reduce exposure.

        Vectorized: multiply entire weight matrix by scalar mask.
        """
        regime_scale = np.where(regimes == 1, high_vol_scale, 1.0)
        scale_df = pd.DataFrame(
            regime_scale[:, None] * np.ones((1, weights.shape[1])),
            index=weights.index,
            columns=weights.columns,
        )
        return weights * scale_df
