"""
strategies/pairs_trading.py
----------------------------
Statistical Pairs Trading Strategy.

Logic:
    Find pairs of stocks with historically correlated prices.
    When the spread between them diverges beyond Z standard deviations,
    bet on mean reversion:
        - Long the underperformer
        - Short the outperformer

This is market-neutral — the direction of the overall market
doesn't matter, only the relative performance of the pair.

Implementation:
    Rather than finding dynamic pairs (complex, slow), we use
    predefined sector pairs from the universe:
        JPM / BAC   — large cap banks
        XOM / CVX   — major oil (if both in universe)
        AAPL / MSFT — mega cap tech

    Spread = log(price_A) - log(price_B)
    Z-score of spread = (spread - rolling_mean) / rolling_std

    Entry:  |z| > entry_threshold  (default 2.0)
    Exit:   |z| < exit_threshold   (default 0.5)

All vectorized — spread computation across all pairs simultaneously.
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


# Predefined pairs — both tickers must be in the universe
PREDEFINED_PAIRS = [
    # Banks
    ("JPM",  "BAC"),
    ("GS",   "MS"),
    ("WFC",  "BAC"),
    # Tech mega cap
    ("AAPL", "MSFT"),
    ("AMZN", "GOOGL"),
    ("META", "GOOGL"),
    # Semiconductors
    ("NVDA", "AMD"),
    ("INTC", "AMD"),
    # Energy
    ("XOM",  "CVX"),
    ("XOM",  "COP"),
    # Consumer
    ("KO",   "PEP"),
    ("WMT",  "COST"),
    # Payments
    ("V",    "MA"),
    # Healthcare
    ("JNJ",  "PFE"),
    ("UNH",  "CVS"),
]


class PairsTradingStrategy(BaseStrategy):

    name = "pairs_trading"
    description = "Statistical pairs trading — long/short correlated stock pairs on spread divergence"
    preferred_regimes = [1]  # High Vol — market neutral works well here
    default_params = {
        "lookback":         60,   # rolling window for spread z-score
        "entry_threshold":  2.0,  # z-score to enter
        "exit_threshold":   0.5,  # z-score to exit
        "position_size":    0.5,  # fraction per leg
    }

    def entry_signals(self, features: dict, regimes: np.ndarray) -> pd.DataFrame:
        """
        Generate long/short signals for each valid pair.

        For each pair (A, B):
            spread   = log(A) - log(B)
            z        = (spread - rolling_mean) / rolling_std

            z > +threshold → A expensive vs B → SHORT A, LONG B
            z < -threshold → A cheap vs B    → LONG A, SHORT B
        """
        close    = features["close"]
        lookback = self.get_param("lookback")
        entry_z  = self.get_param("entry_threshold")

        signals  = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        tickers  = set(close.columns.tolist())

        for ticker_a, ticker_b in PREDEFINED_PAIRS:
            # Skip pair if either ticker not in universe
            if ticker_a not in tickers or ticker_b not in tickers:
                continue

            # Compute log spread
            spread = np.log(close[ticker_a]) - np.log(close[ticker_b])

            # Rolling z-score
            roll_mean = spread.rolling(lookback).mean()
            roll_std  = spread.rolling(lookback).std().replace(0, np.nan)
            z_score   = (spread - roll_mean) / roll_std

            # Entry signals
            # z > +threshold: A overpriced vs B → short A, long B
            overpriced_a  = z_score > entry_z
            # z < -threshold: A underpriced vs B → long A, short B
            underpriced_a = z_score < -entry_z

            signals.loc[overpriced_a,  ticker_a] = -self.get_param("position_size")
            signals.loc[overpriced_a,  ticker_b] =  self.get_param("position_size")
            signals.loc[underpriced_a, ticker_a] =  self.get_param("position_size")
            signals.loc[underpriced_a, ticker_b] = -self.get_param("position_size")

        return signals

    def exit_signals(self, features: dict, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Exit when spread reverts toward zero (z < exit_threshold).
        """
        close    = features["close"]
        lookback = self.get_param("lookback")
        exit_z   = self.get_param("exit_threshold")
        tickers  = set(close.columns.tolist())

        exit_mask = pd.DataFrame(False, index=close.index, columns=close.columns)

        for ticker_a, ticker_b in PREDEFINED_PAIRS:
            if ticker_a not in tickers or ticker_b not in tickers:
                continue

            spread    = np.log(close[ticker_a]) - np.log(close[ticker_b])
            roll_mean = spread.rolling(lookback).mean()
            roll_std  = spread.rolling(lookback).std().replace(0, np.nan)
            z_score   = (spread - roll_mean) / roll_std

            # Exit when spread has reverted
            reverted = z_score.abs() < exit_z

            exit_mask.loc[reverted, ticker_a] = True
            exit_mask.loc[reverted, ticker_b] = True

        return exit_mask

    def position_size(
        self,
        signals: pd.DataFrame,
        features: dict,
        capital: float,
    ) -> pd.DataFrame:
        """
        Signals already have position sizes embedded (±0.5 per leg).
        Each pair is dollar-neutral: +0.5 one leg, -0.5 the other.
        """
        return signals