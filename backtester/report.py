"""
backtester/report.py
--------------------
Generates a clean, human-readable backtest report.

Used for:
    - Printing to console during development
    - Saving as text summary alongside results
    - README/CV screenshots

Call generate_report() after a backtest completes.
"""

import pandas as pd
import numpy as np
from backtester.risk import (
    sharpe_ratio, sortino_ratio, max_drawdown,
    cagr, total_return, win_rate, calmar_ratio,
    drawdown_series, monthly_returns_table,
)

REGIME_NAMES = {0: "Low Vol", 1: "High Vol", 2: "Transitional"}
SEPARATOR = "=" * 60


def generate_report(
    result: dict,
    config,
    print_output: bool = True,
) -> str:
    """
    Generate a full text backtest report.

    Parameters
    ----------
    result  : dict from BacktestEngine.run()
    config  : BacktestConfig instance
    print_output : if True, prints to console

    Returns
    -------
    str — full report as a string
    """
    equity = result["equity_curve"]
    returns = result["returns"]
    trades = result["trades"]
    regime_series = result["regime_series"]

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(SEPARATOR)
    lines.append("  REGIMEROUTE — BACKTEST REPORT")
    lines.append(SEPARATOR)
    lines.append(f"  Period     : {equity.index[0].date()} → {equity.index[-1].date()}")
    lines.append(f"  Duration   : {len(equity) / 252:.1f} years ({len(equity)} trading days)")
    lines.append(f"  Capital    : ${config.capital:,.0f}")
    lines.append(f"  Commission : {config.commission:.2%} | Slippage: {config.slippage:.3%}")
    lines.append(SEPARATOR)

    # ── Performance ───────────────────────────────────────────────────────────
    lines.append("")
    lines.append("  PERFORMANCE")
    lines.append("-" * 60)
    lines.append(f"  Total Return    : {total_return(equity):.2%}")
    lines.append(f"  CAGR            : {cagr(equity):.2%}")
    lines.append(f"  Final Equity    : ${equity.iloc[-1]:,.2f}")
    lines.append("")

    # ── Risk ──────────────────────────────────────────────────────────────────
    lines.append("  RISK METRICS")
    lines.append("-" * 60)
    lines.append(f"  Sharpe Ratio    : {sharpe_ratio(returns):.3f}")
    lines.append(f"  Sortino Ratio   : {sortino_ratio(returns):.3f}")
    lines.append(f"  Calmar Ratio    : {calmar_ratio(equity):.3f}")
    lines.append(f"  Max Drawdown    : {max_drawdown(equity):.2%}")
    lines.append(f"  Volatility      : {returns.std() * np.sqrt(252):.2%}")
    lines.append(f"  Win Rate        : {win_rate(returns):.2%}")
    lines.append("")

    # ── Regime Attribution ────────────────────────────────────────────────────
    lines.append("  REGIME ATTRIBUTION")
    lines.append("-" * 60)

    for regime_id, name in REGIME_NAMES.items():
        mask = regime_series == regime_id
        regime_returns = returns[mask]

        if len(regime_returns) < 10:
            continue

        regime_equity = (1 + regime_returns).cumprod() * config.capital
        lines.append(f"  {name} ({mask.sum()} days, {mask.mean():.0%} of time)")
        lines.append(f"    Sharpe   : {sharpe_ratio(regime_returns):.3f}")
        lines.append(f"    Return   : {regime_returns.mean() * 252:.2%} annualized")
        lines.append(f"    Win Rate : {win_rate(regime_returns):.2%}")
        lines.append(f"    Max DD   : {max_drawdown(regime_equity):.2%}")
        lines.append("")

    # ── Trade Statistics ──────────────────────────────────────────────────────
    if not trades.empty:
        lines.append("  TRADE STATISTICS")
        lines.append("-" * 60)
        lines.append(f"  Total Trades    : {len(trades)}")
        lines.append(f"  Avg Holding     : {trades['holding_days'].mean():.1f} days")

        winners = trades[trades["pnl_pct"] > 0]
        losers = trades[trades["pnl_pct"] < 0]

        lines.append(f"  Win Rate        : {len(winners)/len(trades):.2%}")
        if len(winners) > 0:
            lines.append(f"  Avg Win         : {winners['pnl_pct'].mean():.2%}")
        if len(losers) > 0:
            lines.append(f"  Avg Loss        : {losers['pnl_pct'].mean():.2%}")
        lines.append(f"  Best Trade      : {trades['pnl_pct'].max():.2%}")
        lines.append(f"  Worst Trade     : {trades['pnl_pct'].min():.2%}")

        # Trades by regime
        lines.append("")
        lines.append("  Trades by Regime:")
        for regime_id, name in REGIME_NAMES.items():
            regime_trades = trades[trades["regime_at_entry"] == regime_id]
            if len(regime_trades) > 0:
                wr = (regime_trades["pnl_pct"] > 0).mean()
                lines.append(f"    {name:15} : {len(regime_trades):4d} trades | Win Rate: {wr:.2%}")

    # ── Monthly Returns ───────────────────────────────────────────────────────
    lines.append("")
    lines.append("  MONTHLY RETURNS")
    lines.append("-" * 60)
    try:
        monthly = monthly_returns_table(returns)
        for year, row in monthly.iterrows():
            row_str = "  " + str(year) + "  "
            for val in row:
                if pd.isna(val):
                    row_str += "      "
                elif val > 0:
                    row_str += f" +{val:.1%}"
                else:
                    row_str += f"  {val:.1%}"
            lines.append(row_str)
    except Exception:
        lines.append("  (Monthly table unavailable)")

    lines.append("")
    lines.append(SEPARATOR)

    report = "\n".join(lines)

    if print_output:
        print(report)

    return report


def save_report(report: str, path: str = "backtest_report.txt"):
    """Save report to a text file."""
    with open(path, "w") as f:
        f.write(report)
    print(f"Report saved to {path}")