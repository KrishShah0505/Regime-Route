"""
backtester/risk.py
------------------
Risk metrics and performance attribution.

All metrics computed from the equity curve and returns Series.
No loops — pure vectorized NumPy/Pandas operations.

Metrics:
    - Total return, CAGR
    - Sharpe Ratio, Sortino Ratio
    - Max Drawdown, Calmar Ratio
    - Win Rate, Profit Factor
    - Regime Attribution (the unique feature)
    - Monthly/Annual returns heatmap data
"""

import numpy as np
import pandas as pd
from typing import Optional

TRADING_DAYS = 252


# ── Core Metrics ──────────────────────────────────────────────────────────────

def total_return(equity_curve: pd.Series) -> float:
    """Total return from start to end of backtest."""
    return equity_curve.iloc[-1] / equity_curve.iloc[0] - 1


def cagr(equity_curve: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    n_years = len(equity_curve) / TRADING_DAYS
    if n_years == 0:
        return 0.0
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Annualized Sharpe Ratio.
    Assumes daily returns input.
    """
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Sortino Ratio — like Sharpe but only penalises downside volatility.
    Better metric for strategies with asymmetric return distributions.
    """
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = returns - daily_rf
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    downside_std = downside.std() * np.sqrt(TRADING_DAYS)
    return (excess.mean() * TRADING_DAYS) / downside_std


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum drawdown: largest peak-to-trough decline.
    Vectorized: running max then ratio, no loop.

    Returns a negative number (e.g. -0.35 = -35% drawdown).
    """
    rolling_peak = equity_curve.cummax()
    drawdown = (equity_curve - rolling_peak) / rolling_peak
    return float(drawdown.min())


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Full drawdown series — used for plotting."""
    rolling_peak = equity_curve.cummax()
    return (equity_curve - rolling_peak) / rolling_peak


def calmar_ratio(equity_curve: pd.Series) -> float:
    """CAGR / abs(max_drawdown). Higher = better risk-adjusted return."""
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return cagr(equity_curve) / mdd


def win_rate(returns: pd.Series) -> float:
    """Fraction of days with positive returns."""
    positive = (returns > 0).sum()
    total = (returns != 0).sum()
    return float(positive / total) if total > 0 else 0.0


def profit_factor(returns: pd.Series) -> float:
    """
    Sum of winning returns / abs(sum of losing returns).
    > 1 = strategy makes more than it loses on average.
    """
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return float(wins / losses) if losses > 0 else float("inf")


def volatility(returns: pd.Series) -> float:
    """Annualized volatility of daily returns."""
    return float(returns.std() * np.sqrt(TRADING_DAYS))


# ── Regime Attribution ────────────────────────────────────────────────────────

def regime_attribution(
    returns: pd.Series,
    regime_series: pd.Series,
    regime_names: dict = None,
) -> pd.DataFrame:
    """
    THE UNIQUE FEATURE — break down performance by regime.

    Answers: "Is the strategy's edge coming from low-vol periods?
              Or does it work across all regimes?"

    For each regime: compute Sharpe, mean return, vol, win rate, trade count.

    This is what separates QuantRegime from generic backtesting frameworks.
    """
    if regime_names is None:
        regime_names = {0: "Low Vol", 1: "High Vol", 2: "Transitional"}

    combined = pd.DataFrame({
        "return": returns,
        "regime": regime_series,
    }).dropna()

    rows = []
    for regime_id, name in regime_names.items():
        mask = combined["regime"] == regime_id
        regime_returns = combined.loc[mask, "return"]

        if len(regime_returns) < 10:
            continue

        equity = (1 + regime_returns).cumprod()
        rows.append({
            "regime": name,
            "days": int(mask.sum()),
            "pct_of_time": round(mask.mean(), 3),
            "mean_daily_return": round(regime_returns.mean(), 6),
            "annualized_return": round(regime_returns.mean() * TRADING_DAYS, 4),
            "annualized_vol": round(regime_returns.std() * np.sqrt(TRADING_DAYS), 4),
            "sharpe": round(sharpe_ratio(regime_returns), 3),
            "win_rate": round(win_rate(regime_returns), 3),
            "max_drawdown": round(max_drawdown(equity), 4),
            "total_return": round(total_return(equity), 4),
        })

    return pd.DataFrame(rows).set_index("regime")


# ── Trade-Level Metrics ───────────────────────────────────────────────────────

def trade_statistics(trades: pd.DataFrame) -> dict:
    """Compute trade-level statistics from the trades DataFrame."""
    if trades.empty:
        return {}

    pnl = trades["pnl_pct"]
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]

    return {
        "total_trades": len(trades),
        "win_rate": round(len(winners) / len(pnl), 3) if len(pnl) > 0 else 0,
        "avg_win": round(winners.mean(), 4) if len(winners) > 0 else 0,
        "avg_loss": round(losers.mean(), 4) if len(losers) > 0 else 0,
        "best_trade": round(pnl.max(), 4),
        "worst_trade": round(pnl.min(), 4),
        "avg_holding_days": round(trades["holding_days"].mean(), 1),
        "profit_factor": round(
            winners.sum() / abs(losers.sum()), 3
        ) if len(losers) > 0 and losers.sum() != 0 else float("inf"),
    }


# ── Monthly Returns ───────────────────────────────────────────────────────────

def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """
    Returns a (years x 12) DataFrame of monthly returns.
    Used for the heatmap in the dashboard.
    Vectorized via resample and unstack.
    """
    monthly = (1 + returns).resample("ME").prod() - 1
    table = monthly.groupby([monthly.index.year, monthly.index.month]).first()
    table = table.unstack(level=1)
    table.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    return table.round(4)


# ── Master Summary ────────────────────────────────────────────────────────────

def compute_all_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    regime_series: pd.Series,
    trades: pd.DataFrame,
    initial_capital: float = 100_000,
) -> dict:
    """
    Compute all risk metrics in one call.
    Returns a flat dictionary suitable for JSON serialisation.
    """
    return {
        # Returns
        "total_return":       round(total_return(equity_curve), 4),
        "cagr":               round(cagr(equity_curve), 4),
        "final_equity":       round(equity_curve.iloc[-1], 2),
        "initial_capital":    initial_capital,

        # Risk-adjusted
        "sharpe_ratio":       round(sharpe_ratio(returns), 3),
        "sortino_ratio":      round(sortino_ratio(returns), 3),
        "calmar_ratio":       round(calmar_ratio(equity_curve), 3),
        "volatility":         round(volatility(returns), 4),

        # Drawdown
        "max_drawdown":       round(max_drawdown(equity_curve), 4),

        # Win/loss
        "win_rate":           round(win_rate(returns), 3),
        "profit_factor":      round(profit_factor(returns), 3),

        # Trade stats
        **trade_statistics(trades),

        # Period info
        "start_date":         str(equity_curve.index[0].date()),
        "end_date":           str(equity_curve.index[-1].date()),
        "total_days":         len(equity_curve),
        "total_years":        round(len(equity_curve) / TRADING_DAYS, 1),
    }
