"""
backtester/regime_audit.py
--------------------------
Per-regime strategy attribution matrix.

Answers: "Which strategy performs best in each specific regime?"

For each regime (Low Vol, High Vol, Transitional) we isolate only
the days belonging to that regime and compute how EVERY strategy
performed on those exact days.

Expected result (if regime assignments are correct):
    Low Vol    → Momentum should have highest Sharpe
    High Vol   → Mean Reversion should have highest Sharpe
    Transitional → Trend Filter should have highest Sharpe

If the diagonal is NOT the highest values, the regime assignments
need fixing. Right now we expect High Vol to be broken because
mean reversion is underperforming — this table will prove it.
"""

import numpy as np
import pandas as pd
from backtester.risk import sharpe_ratio, max_drawdown, win_rate

REGIME_NAMES = {
    0: "Low Vol",
    1: "High Vol",
    2: "Transitional",
}

STRATEGY_NAMES = [
    "momentum",
    "mean_reversion",
    "trend_filter",
]


def run_regime_audit(
    features: dict,
    regimes: np.ndarray,
    config,
) -> dict:
    """
    Run all three strategies across all data, then slice
    performance by regime to build the audit matrix.

    Returns
    -------
    dict with:
        'matrix'  : the full strategy x regime performance table
        'verdict' : which strategy is best per regime
        'diagonal_correct': bool — is the right strategy winning?
    """
    from strategies.router import RegimeRouter
    from backtester.engine import BacktestEngine

    engine = BacktestEngine(config)
    regime_series = pd.Series(regimes, index=features["close"].index)

    # ── Step 1: Run all three strategies and get daily returns ────────────
    strategy_returns = {}

    for strategy_name in STRATEGY_NAMES:
        # Force all regimes to use this one strategy
        forced_map = {0: strategy_name, 1: strategy_name, 2: strategy_name}
        router = RegimeRouter(regime_map=forced_map)
        weights = router.generate_signals(features, regimes)
        result = engine.run(weights, features, regimes)
        strategy_returns[strategy_name] = result["returns"]

    # ── Step 2: For each regime, slice returns and compute metrics ────────
    matrix_rows = []

    for strategy_name, returns in strategy_returns.items():
        row = {"strategy": strategy_name.replace("_", " ").title()}

        for regime_id, regime_name in REGIME_NAMES.items():
            mask = regime_series == regime_id
            regime_returns = returns[mask]

            if len(regime_returns) < 10:
                row[regime_name] = {
                    "sharpe":   None,
                    "win_rate": None,
                    "return":   None,
                    "days":     0,
                }
                continue

            row[regime_name] = {
                "sharpe":   round(sharpe_ratio(regime_returns), 3),
                "win_rate": round(win_rate(regime_returns), 3),
                "return":   round(regime_returns.mean() * 252, 4),
                "days":     int(mask.sum()),
            }

        matrix_rows.append(row)

    # ── Step 3: Determine verdict per regime ──────────────────────────────
    verdict = {}
    diagonal_correct = True

    expected_best = {
        "Low Vol":       "Momentum",
        "High Vol":      "Mean Reversion",
        "Transitional":  "Trend Filter",
    }

    for regime_name in REGIME_NAMES.values():
        best_strategy = None
        best_sharpe = -999

        for row in matrix_rows:
            regime_data = row.get(regime_name, {})
            sharpe = regime_data.get("sharpe")
            if sharpe is not None and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_strategy = row["strategy"]

        verdict[regime_name] = {
            "best_strategy": best_strategy,
            "best_sharpe":   round(best_sharpe, 3),
            "expected":      expected_best.get(regime_name),
            "correct":       best_strategy == expected_best.get(regime_name),
        }

        if not verdict[regime_name]["correct"]:
            diagonal_correct = False

    return {
        "matrix":           matrix_rows,
        "verdict":          verdict,
        "diagonal_correct": diagonal_correct,
    }


def format_audit_for_api(audit: dict) -> dict:
    """Clean the audit result for JSON serialisation."""
    return {
        "matrix":           audit["matrix"],
        "verdict":          audit["verdict"],
        "diagonal_correct": audit["diagonal_correct"],
        "interpretation":   _interpret(audit),
    }


def _interpret(audit: dict) -> str:
    """Generate a plain English interpretation of the audit results."""
    verdict = audit["verdict"]
    lines = []

    for regime, v in verdict.items():
        if v["correct"]:
            lines.append(
                f"{regime}: {v['best_strategy']} is correctly the best "
                f"strategy (Sharpe {v['best_sharpe']}) ✓"
            )
        else:
            lines.append(
                f"{regime}: Expected {v['expected']} to be best but "
                f"{v['best_strategy']} leads (Sharpe {v['best_sharpe']}) ✗ "
                f"— regime assignment may need review"
            )

    if audit["diagonal_correct"]:
        lines.append(
            "\nAll regime assignments are optimal — "
            "the HMM is correctly routing to the best strategy per regime."
        )
    else:
        lines.append(
            "\nSome regime assignments are suboptimal — "
            "consider reviewing strategy parameters or replacing "
            "underperforming strategies."
        )

    return " | ".join(lines)