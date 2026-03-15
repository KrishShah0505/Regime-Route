"""
backtester/controls.py
----------------------
Control strategies for benchmarking RegimeRoute.

Every control runs on identical data, date range, and capital
as the main backtest so comparisons are fair.

Controls:
    1. Static Momentum        - momentum every day, no regime switching
    2. Static Mean Reversion  - mean reversion every day, no regime switching
    3. Static Trend Following - trend filter every day, no regime switching
    4. SPY Buy and Hold       - passive market baseline
    5. Equal Weight Universe  - passive diversification baseline
    6. Random Regime          - proves HMM classifier adds real value
"""

import numpy as np
import pandas as pd
from backtester.engine import BacktestEngine, BacktestConfig
from backtester.risk import (
    sharpe_ratio, max_drawdown, cagr,
    total_return, win_rate, volatility
)


# ── Master Function ───────────────────────────────────────────────────────────

def run_all_controls(
    features: dict,
    regimes: np.ndarray,
    config: BacktestConfig,
) -> dict:
    """
    Run all 6 control strategies on the same data.

    Returns
    -------
    dict:
        key   = strategy name
        value = dict with equity_curve + metrics
    """
    controls = {}

    print("  Running control: Static Momentum...")
    controls["Static Momentum"] = _run_static(
        features, regimes, config, strategy_name="momentum"
    )

    print("  Running control: Static Mean Reversion...")
    controls["Static Mean Reversion"] = _run_static(
        features, regimes, config, strategy_name="mean_reversion"
    )

    print("  Running control: Static Trend Following...")
    controls["Static Trend Following"] = _run_static(
        features, regimes, config, strategy_name="trend_filter"
    )

    print("  Running control: SPY Buy and Hold...")
    controls["SPY Buy & Hold"] = _run_spy(features, config)

    print("  Running control: Equal Weight...")
    controls["Equal Weight"] = _run_equal_weight(features, config)

    print("  Running control: Random Regime...")
    controls["Random Regime"] = _run_random_regime(features, regimes, config)

    return controls


# ── Static Strategies ─────────────────────────────────────────────────────────

def _run_static(
    features: dict,
    regimes: np.ndarray,
    config: BacktestConfig,
    strategy_name: str,
) -> dict:
    """
    Force all three regimes to use the same strategy.
    This removes the regime switching entirely.

    If RegimeRoute beats this — regime switching adds value.
    If it doesn't — the HMM layer is pointless complexity.
    """
    from strategies.router import RegimeRouter

    # Override regime map so all regimes use the same strategy
    forced_map = {
        0: strategy_name,
        1: strategy_name,
        2: strategy_name,
    }

    router = RegimeRouter(regime_map=forced_map)
    weights = router.generate_signals(features, regimes)

    engine = BacktestEngine(config)
    result = engine.run(weights, features, regimes)

    return {
        "equity_curve": result["equity_curve"],
        "metrics": _compute_metrics(result["equity_curve"], config.capital),
    }


# ── Passive Baselines ─────────────────────────────────────────────────────────

def _run_spy(features: dict, config: BacktestConfig) -> dict:
    """
    Buy SPY on day 1 and hold forever.
    The universal benchmark — if you can't beat this nothing else matters.
    Uses equal weight as proxy if SPY not in the universe.
    """
    returns = features["returns"]

    if "SPY" in returns.columns:
        daily = returns["SPY"].fillna(0)
    else:
        # Equal weight proxy — close enough for benchmarking
        daily = returns.mean(axis=1).fillna(0)

    equity = config.capital * (1 + daily).cumprod()
    equity.name = "SPY Buy & Hold"

    return {
        "equity_curve": equity,
        "metrics": _compute_metrics(equity, config.capital),
    }


def _run_equal_weight(features: dict, config: BacktestConfig) -> dict:
    """
    Equal weight all tickers, rebalanced daily.
    Vectorized: mean(axis=1) gives equal weight return each day.
    No signal generation needed — pure passive diversification.
    """
    returns = features["returns"]

    # One vectorized operation across all tickers each day
    daily = returns.mean(axis=1).fillna(0)

    # Small daily rebalancing cost
    rebal_cost = (config.commission + config.slippage) * 0.1
    net = daily - rebal_cost

    equity = config.capital * (1 + net).cumprod()
    equity.name = "Equal Weight"

    return {
        "equity_curve": equity,
        "metrics": _compute_metrics(equity, config.capital),
    }


def _run_random_regime(
    features: dict,
    regimes: np.ndarray,
    config: BacktestConfig,
) -> dict:
    """
    Randomly shuffle regime labels then run the full pipeline.

    This destroys any real signal in the regime labels.
    If random regime performs similarly to HMM regime:
        → the classifier is not detecting anything real
    If HMM regime significantly outperforms random:
        → the classifier is genuinely detecting market states

    This is the most rigorous control in the project.
    Run 5 random seeds and average to reduce variance.
    """
    from strategies.router import RegimeRouter

    n_seeds = 5
    equity_curves = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        # Shuffle destroys temporal structure — any regime pattern is now noise
        random_regimes = np.random.permutation(regimes)

        router = RegimeRouter()
        weights = router.generate_signals(features, random_regimes)

        engine = BacktestEngine(config)
        result = engine.run(weights, features, random_regimes)
        equity_curves.append(result["equity_curve"])

    # Stack all curves and average — reduces luck from a single random seed
    # Vectorized: pd.concat then mean(axis=1)
    stacked = pd.concat(equity_curves, axis=1)
    avg_equity = stacked.mean(axis=1)
    avg_equity.name = "Random Regime"

    return {
        "equity_curve": avg_equity,
        "metrics": _compute_metrics(avg_equity, config.capital),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(equity: pd.Series, capital: float) -> dict:
    """Compute summary metrics for any equity curve."""
    returns = equity.pct_change().fillna(0)

    return {
        "total_return": round(total_return(equity), 4),
        "cagr":         round(cagr(equity), 4),
        "sharpe":       round(sharpe_ratio(returns), 3),
        "max_drawdown": round(max_drawdown(equity), 4),
        "volatility":   round(volatility(returns), 4),
        "win_rate":     round(win_rate(returns), 3),
        "final_equity": round(float(equity.iloc[-1]), 2),
    }


# ── Comparison Table ──────────────────────────────────────────────────────────

def build_comparison_table(
    main_equity: pd.Series,
    main_metrics: dict,
    controls: dict,
) -> list:
    """
    Build the full comparison table.
    RegimeRoute first, then all controls.
    Returns list of dicts ready for JSON serialisation.
    """
    rows = []

    # RegimeRoute — main strategy always first
    rows.append({
        "name":         "RegimeRoute ★",
        "total_return": main_metrics["total_return"],
        "cagr":         main_metrics["cagr"],
        "sharpe":       main_metrics["sharpe_ratio"],
        "max_drawdown": main_metrics["max_drawdown"],
        "volatility":   main_metrics["volatility"],
        "win_rate":     main_metrics["win_rate"],
        "final_equity": main_metrics["final_equity"],
        "is_main":      True,
    })

    # All controls
    for name, data in controls.items():
        m = data["metrics"]
        rows.append({
            "name":         name,
            "total_return": m["total_return"],
            "cagr":         m["cagr"],
            "sharpe":       m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "volatility":   m["volatility"],
            "win_rate":     m["win_rate"],
            "final_equity": m["final_equity"],
            "is_main":      False,
        })

    return rows


# ── Equity Curves For Chart ───────────────────────────────────────────────────

def build_chart_data(
    main_equity: pd.Series,
    controls: dict,
) -> list:
    """
    Combine all equity curves into a single list of dicts for the chart.
    Each dict = one trading day with values for all strategies.

    Format:
        [
            {
                "date": "2015-01-02",
                "RegimeRoute": 100000,
                "Static Momentum": 99800,
                "SPY Buy & Hold": 100200,
                ...
            },
            ...
        ]

    Recharts can render multiple lines from this format directly.
    """
    # Align all curves to the same index
    all_curves = {"RegimeRoute": main_equity}
    for name, data in controls.items():
        all_curves[name] = data["equity_curve"]

    # Combine into DataFrame — vectorized alignment via reindex
    df = pd.DataFrame(all_curves)
    df = df.ffill().bfill()

    # Convert to list of dicts for JSON
    result = []
    for date, row in df.iterrows():
        point = {"date": str(date.date())}
        for col in df.columns:
            point[col] = round(float(row[col]), 2)
        result.append(point)

    return result