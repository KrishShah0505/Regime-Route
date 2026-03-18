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

    def _get_values(self, rule, features, index, columns, n_tickers):
        """Extract indicator values as (n_days, n_tickers) numpy array."""
        indicator_fn = INDICATOR_MAP[rule["indicator"]]
        raw = indicator_fn(features)
        if isinstance(raw, pd.DataFrame):
            return raw.reindex(index=index, columns=columns).ffill().values
        elif isinstance(raw, pd.Series):
            return raw.reindex(index).ffill().values[:, None] * np.ones((1, n_tickers))
        else:
            return raw

    def generate_signals(
        self,
        features: dict,
        regimes: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compile rules into weight DataFrame.

        KEY FIX: Buy rules and sell rules are processed independently.
        Buy conditions are ANDed only with other buy conditions.
        Sell conditions are ANDed only with other sell conditions.
        The two masks never mix during construction.

        Returns
        -------
        pd.DataFrame shape (n_days, n_tickers) — portfolio weights
        """
        index     = features["close"].index
        columns   = features["close"].columns
        n_days    = len(index)
        n_tickers = len(columns)

        # Separate rules by action type
        buy_rules  = [r for r in self.rules if r["action"] == "buy"]
        sell_rules = [r for r in self.rules if r["action"] == "sell"]

        # ── Buy mask — AND all buy conditions independently ───────────────
        if buy_rules:
            buy_mask = np.ones((n_days, n_tickers), dtype=bool)
            for rule in buy_rules:
                operator_fn = OPERATOR_MAP[rule["operator"]]
                values      = self._get_values(rule, features, index, columns, n_tickers)
                buy_mask    = buy_mask & operator_fn(values, float(rule["value"]))
        else:
            buy_mask = np.zeros((n_days, n_tickers), dtype=bool)

        # ── Sell mask — AND all sell conditions independently ─────────────
        if sell_rules:
            sell_mask = np.ones((n_days, n_tickers), dtype=bool)
            for rule in sell_rules:
                operator_fn = OPERATOR_MAP[rule["operator"]]
                values      = self._get_values(rule, features, index, columns, n_tickers)
                sell_mask   = sell_mask & operator_fn(values, float(rule["value"]))
        else:
            sell_mask = np.zeros((n_days, n_tickers), dtype=bool)

        # ── Build signal matrix ───────────────────────────────────────────
        # Buy = +1, Sell = -1, Both/Neither = 0
        signals = np.zeros((n_days, n_tickers))
        signals[buy_mask  & ~sell_mask] =  self.position_size
        signals[sell_mask & ~buy_mask]  = -self.position_size if self.allow_short else 0

        # ── Apply regime filter ───────────────────────────────────────────
        if self.regime_filter is not None:
            regime_mask = (regimes == self.regime_filter)[:, None]
            signals     = signals * regime_mask

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

    return rsi.fillna(50)


# ── Available Indicators For UI ───────────────────────────────────────────────

def get_available_indicators() -> list:
    """Return indicator metadata for the frontend rule builder."""
    return [
        {"id": "rsi",          "label": "RSI",            "default_value": 30,    "range": [0, 100]},
        {"id": "price",        "label": "Price ($)",       "default_value": 150,   "range": [1, 10000]},
        {"id": "ema_50",       "label": "EMA 50",          "default_value": 150,   "range": [1, 10000]},
        {"id": "ema_200",      "label": "EMA 200",         "default_value": 150,   "range": [1, 10000]},
        {"id": "bb_zscore",    "label": "BB Z-Score",      "default_value": -2.0,  "range": [-4, 4]},
        {"id": "momentum",     "label": "12M Momentum",    "default_value": 0.0,   "range": [-0.5, 2.0]},
        {"id": "volume_ratio", "label": "Volume Ratio",    "default_value": 1.5,   "range": [0, 5]},
        {"id": "vix",          "label": "VIX Level",       "default_value": 20,    "range": [5, 80]},
        {"id": "vix_zscore",   "label": "VIX Z-Score",     "default_value": 0.0,   "range": [-3, 3]},
        {"id": "returns",      "label": "Daily Return",    "default_value": 0.01,  "range": [-0.1, 0.1]},
    ]