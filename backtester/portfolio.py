"""
backtester/portfolio.py
-----------------------
Portfolio state tracking during backtesting.

Tracks:
    - Current positions per ticker
    - Cash balance
    - Daily portfolio value
    - Position history for trade extraction

All operations vectorized where possible.
Used by the engine for position management logic.
"""

import numpy as np
import pandas as pd


class Portfolio:
    """
    Tracks portfolio state across a backtest.

    Parameters
    ----------
    capital   : float — starting capital
    tickers   : list  — universe of tickers
    index     : pd.DatetimeIndex — trading day index
    """

    def __init__(
        self,
        capital: float,
        tickers: list,
        index: pd.DatetimeIndex,
    ):
        self.initial_capital = capital
        self.capital = capital
        self.tickers = tickers
        self.index = index
        self.n_days = len(index)
        self.n_tickers = len(tickers)

        # Position matrix — shape (n_days, n_tickers)
        # Values are weights (fraction of portfolio in each stock)
        self.positions = pd.DataFrame(
            0.0,
            index=index,
            columns=tickers,
        )

        # Daily portfolio value — shape (n_days,)
        self.portfolio_value = pd.Series(
            capital,
            index=index,
            name="portfolio_value",
        )

        # Daily cash — shape (n_days,)
        self.cash = pd.Series(
            capital,
            index=index,
            name="cash",
        )

    # ── Position Updates ──────────────────────────────────────────────────────

    def update_positions(self, weights: pd.DataFrame):
        """
        Set positions from weight DataFrame.
        Vectorized assignment across all tickers and days.
        """
        self.positions = weights.reindex(
            index=self.index,
            columns=self.tickers,
        ).fillna(0)

    # ── Portfolio Value ───────────────────────────────────────────────────────

    def compute_value(
        self,
        returns: pd.DataFrame,
        costs: pd.Series,
    ) -> pd.Series:
        """
        Compute daily portfolio value from positions and returns.

        Vectorized:
            stock_pnl  = positions * returns  (element-wise)
            port_ret   = stock_pnl.sum(axis=1)  (aggregate each day)
            net_ret    = port_ret - costs
            equity     = capital * (1 + net_ret).cumprod()
        """
        # Shift positions by 1 — execute tomorrow what we signal today
        executed = self.positions.shift(1).fillna(0)

        # Gross return each day
        stock_pnl = executed * returns.reindex(
            index=self.index,
            columns=self.tickers,
        ).fillna(0)

        portfolio_returns = stock_pnl.sum(axis=1)
        net_returns = portfolio_returns - costs.reindex(self.index).fillna(0)

        self.portfolio_value = self.initial_capital * (1 + net_returns).cumprod()
        return self.portfolio_value

    # ── Turnover ──────────────────────────────────────────────────────────────

    def compute_turnover(self) -> pd.Series:
        """
        Daily turnover = sum of absolute position changes.
        Vectorized: diff().abs().sum(axis=1)
        Only days where positions change incur transaction costs.
        """
        return self.positions.diff().abs().sum(axis=1)

    def compute_costs(
        self,
        commission: float,
        slippage: float,
    ) -> pd.Series:
        """
        Daily transaction costs = turnover * (commission + slippage).
        Vectorized multiplication across entire series.
        """
        turnover = self.compute_turnover()
        return turnover * (commission + slippage)

    # ── Summary Stats ─────────────────────────────────────────────────────────

    def get_position_summary(self) -> pd.DataFrame:
        """
        Summary of position activity per ticker.
        Returns DataFrame with days_held, avg_weight, max_weight per ticker.
        """
        return pd.DataFrame({
            "days_long":  (self.positions > 0).sum(),
            "days_short": (self.positions < 0).sum(),
            "days_flat":  (self.positions == 0).sum(),
            "avg_weight": self.positions.abs().mean().round(4),
            "max_weight": self.positions.abs().max().round(4),
        })

    def get_concentration(self) -> pd.Series:
        """
        Daily portfolio concentration = sum of squared weights (HHI).
        0 = perfectly diversified, 1 = entire portfolio in one stock.
        Vectorized: (positions ** 2).sum(axis=1)
        """
        return (self.positions ** 2).sum(axis=1)

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self):
        return (
            f"Portfolio("
            f"capital=${self.initial_capital:,.0f}, "
            f"tickers={self.n_tickers}, "
            f"days={self.n_days})"
        )