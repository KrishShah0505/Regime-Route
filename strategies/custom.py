"""
strategies/custom.py
--------------------
Custom strategy compiler.

Takes user-defined rules from the UI and compiles them into
vectorized NumPy signals that plug directly into the backtest engine.

Rule format:
    {
        "indicator": "rsi",
        "operator":  "<",
        "value":     30,
        "action":    "buy"   # or "sell"
    }

Supported indicators:
    price, returns, rsi, ema_50, ema_200, sma_200,
    bb_zscore, volume_ratio, momentum, vix, vix_zscore

Supported operators:
    <, >, <=, >=, ==

Signal output:
    +1  = long
    -1  = short (if allow_short)
     0  = flat
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


# ── Available Indicators ──────────────────────────────────────────────────────

INDICATOR_MAP = {
    "price":        lambda f: f["close"],
    "returns":      lambda f: f["returns"],
    "rsi":          lambda f: _compute_rsi(f["close"]),
    "ema_50":       lambda f: f["ema_50"],
    "ema_200":      lambda f: f["ema_200"],
    "sma_200":      lambda f: f["sma_200"],
    "bb_zscore":    lambda f: f["bb_zscore"],
    "volume_ratio": lambda f: f["volume_ratio"],
    "momentum":     lambda f: f["momentum"],
    "vix":          lambda f: f["vix"].values.reshape(-1, 1) * np.ones((1, f["close"].shape[1])),
    "vix_zscore":   lambda f: f["vix_zscore"].values.reshape(-1, 1) * np.ones((1, f["close"].shape[1])),
}

OPERATOR_MAP = {
    "<":  lambda a, b: a < b,
    ">":  lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: np.isclose(a, b),
}

# Human readable labels for UI
INDICATOR_LABELS = {
    "price":        "Price ($)",
    "returns":      "Daily Return (%)",
    "rsi":          "RSI (0-100)",
    "ema_50":       "EMA 50",
    "ema_200":      "EMA 200",
    "sma_200":      "SMA 200",
    "bb_zscore":    "Bollinger Band Z-Score",
    "volume_ratio": "Volume Ratio",
    "momentum":     "12M Momentum",
    "vix":          "VIX Level",
    "vix_zscore":   "VIX Z-Score",
}


# ── Rule Compiler ─────────────────────────────────────────────────────────────

class CustomStrategy:
    """
    Compiles user-defined rules into vectorized signals.

    Parameters
    ----------
    rules       : list of rule dicts from the UI
    allow_short : whether to allow short positions
    regime_filter : None = all regimes, 0/1/2 = only that regime
    """

    def __init__(
        self,
        rules: List[Dict[str, Any]],
        allow_short: bool = True,
        regime_filter: int = None,
        position_size: float = 1.0,
    ):
        self.rules = rules
        self.allow_short = allow_short
        self.regime_filter = regime_filter
        self.position_size = position_size

        self._validate_rules()

    def _validate_rules(self):
        """Check all rules reference valid indicators and operators."""
        for rule in self.rules:
            if rule["indicator"] not in INDICATOR_MAP:
                raise ValueError(
                    f"Unknown indicator: {rule['indicator']}. "
                    f"Valid: {list(INDICATOR_MAP.keys())}"
                )
            if rule["operator"] not in OPERATOR_MAP:
                raise ValueError(
                    f"Unknown operator: {rule['operator']}. "
                    f"Valid: {list(OPERATOR_MAP.keys())}"
                )
            if rule["action"] not in ("buy", "sell"):
                raise ValueError(
                    f"Unknown action: {rule['action']}. Valid: buy, sell"
                )

    def generate_signals(
        self,
        features: dict,
        regimes: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compile rules into weight DataFrame.

        Steps:
            1. Compute each indicator across all tickers
            2. Apply operator to get boolean mask per rule
            3. AND all buy conditions → buy signal
            4. AND all sell conditions → sell signal
            5. buy=+1, sell=-1, conflict/neither=0
            6. Apply regime filter if set
            7. Apply position sizing

        Returns
        -------
        pd.DataFrame shape (n_days, n_tickers) — portfolio weights
        """
        index   = features["close"].index
        columns = features["close"].columns
        n_days  = len(index)
        n_tickers = len(columns)

        # Start with all True masks — AND conditions together
        buy_mask  = np.ones((n_days, n_tickers), dtype=bool)
        sell_mask = np.ones((n_days, n_tickers), dtype=bool)

        has_buy  = any(r["action"] == "buy"  for r in self.rules)
        has_sell = any(r["action"] == "sell" for r in self.rules)

        # Reset to False if no rules of that type
        if not has_buy:
            buy_mask[:] = False
        if not has_sell:
            sell_mask[:] = False

        for rule in self.rules:
            indicator_fn = INDICATOR_MAP[rule["indicator"]]
            operator_fn  = OPERATOR_MAP[rule["operator"]]

            # Get indicator values — shape (n_days, n_tickers)
            raw = indicator_fn(features)
            if isinstance(raw, pd.DataFrame):
                values = raw.reindex(index=index, columns=columns).ffill().values
            elif isinstance(raw, pd.Series):
                values = raw.reindex(index).ffill().values[:, None] * np.ones((1, n_tickers))
            else:
                values = raw

            # Apply condition
            condition = operator_fn(values, float(rule["value"]))

            if rule["action"] == "buy":
                buy_mask  = buy_mask  & condition
            else:
                sell_mask = sell_mask & condition

        # Build signal matrix
        # Buy = +1, Sell = -1, Both/Neither = 0
        signals = np.zeros((n_days, n_tickers))
        signals[buy_mask  & ~sell_mask] =  self.position_size
        signals[sell_mask & ~buy_mask]  = -self.position_size if self.allow_short else 0

        # Apply regime filter — zero out days outside target regime
        if self.regime_filter is not None:
            regime_mask = (regimes == self.regime_filter)[:, None]
            signals = signals * regime_mask

        weights = pd.DataFrame(signals, index=index, columns=columns)
        return weights


# ── RSI Helper ────────────────────────────────────────────────────────────────

def _compute_rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Vectorized RSI computation across all tickers.
    RSI = 100 - 100 / (1 + avg_gain / avg_loss)
    """
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)  # fill NaN with neutral 50


# ── Available Indicators For UI ───────────────────────────────────────────────

def get_available_indicators() -> list:
    """Return indicator metadata for the frontend rule builder."""
    return [
        {"id": "rsi",          "label": "RSI",              "default_value": 30,   "range": [0, 100]},
        {"id": "price",        "label": "Price ($)",         "default_value": 100,  "range": [0, 10000]},
        {"id": "ema_50",       "label": "EMA 50",            "default_value": 100,  "range": [0, 10000]},
        {"id": "ema_200",      "label": "EMA 200",           "default_value": 100,  "range": [0, 10000]},
        {"id": "bb_zscore",    "label": "BB Z-Score",        "default_value": -2,   "range": [-4, 4]},
        {"id": "momentum",     "label": "12M Momentum",      "default_value": 0,    "range": [-1, 2]},
        {"id": "volume_ratio", "label": "Volume Ratio",      "default_value": 1.5,  "range": [0, 5]},
        {"id": "vix",          "label": "VIX Level",         "default_value": 20,   "range": [5, 80]},
        {"id": "vix_zscore",   "label": "VIX Z-Score",       "default_value": 0,    "range": [-3, 3]},
        {"id": "returns",      "label": "Daily Return",      "default_value": 0,    "range": [-0.1, 0.1]},
    ]