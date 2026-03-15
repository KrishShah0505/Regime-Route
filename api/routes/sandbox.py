"""
api/routes/sandbox.py
---------------------
POST /api/sandbox — run a custom strategy + Monte Carlo validation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import numpy as np
import pandas as pd

from data.fetcher import fetch_universe_data, validate_data
from data.processor import build_all_features
from regime.features import build_regime_features, select_hmm_features
from regime.classifier import RegimeClassifier
from strategies.custom import CustomStrategy, get_available_indicators
from backtester.engine import BacktestEngine, BacktestConfig
from backtester.risk import compute_all_metrics
from backtester.monte_carlo import run_monte_carlo, extract_trade_returns

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Request Schema ────────────────────────────────────────────────────────────

class Rule(BaseModel):
    indicator: str
    operator:  str
    value:     float
    action:    str


class SandboxRequest(BaseModel):
    tickers:        List[str]
    start_date:     str
    end_date:       str
    capital:        float = 100_000
    rules:          List[Rule]
    regime_filter:  Optional[int] = None
    allow_short:    bool = True
    position_size:  float = 1.0
    regime_method:  str = "hmm"
    commission:     float = 0.001
    slippage:       float = 0.0005
    n_simulations:  int = 10_000


# ── Serialisation Helper ──────────────────────────────────────────────────────

def _sanitize(obj):
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("")
async def run_sandbox(request: SandboxRequest):
    logger.info(f"Sandbox request | rules={len(request.rules)} | tickers={request.tickers}")

    try:
        # ── Step 1: Fetch Data ─────────────────────────────────────────────
        raw = fetch_universe_data(
            request.tickers,
            request.start_date,
            request.end_date,
        )
        validate_data(raw)

        # ── Step 2: Build Features ─────────────────────────────────────────
        features = build_all_features(raw)

        # ── Step 3: Classify Regimes ───────────────────────────────────────
        regime_features = build_regime_features(
            features["vix"],
            features["returns"],
        )
        hmm_features = select_hmm_features(regime_features.dropna())

        classifier = RegimeClassifier(method=request.regime_method)
        labels = classifier.fit_predict(hmm_features)

        full_regimes = pd.Series(2, index=features["close"].index)
        full_regimes.iloc[len(full_regimes) - len(labels):] = labels
        regime_array = full_regimes.values

        # ── Step 4: Compile Custom Strategy ───────────────────────────────
        strategy = CustomStrategy(
            rules=[r.model_dump() for r in request.rules],
            allow_short=request.allow_short,
            regime_filter=request.regime_filter,
            position_size=request.position_size,
        )
        weights = strategy.generate_signals(features, regime_array)

        total_signal = weights.abs().sum().sum()
        if total_signal == 0:
            return _sanitize({
                "status":  "no_signals",
                "message": "Your rules generated no trading signals. "
                           "Try relaxing the conditions or checking indicator ranges.",
                "rules":   [r.model_dump() for r in request.rules],
            })

        # ── Step 5: Run Backtest ───────────────────────────────────────────
        config = BacktestConfig(
            capital=request.capital,
            commission=request.commission,
            slippage=request.slippage,
            allow_short=request.allow_short,
        )
        engine = BacktestEngine(config)
        result = engine.run(weights, features, regime_array)

        # ── Step 6: Compute Metrics ────────────────────────────────────────
        metrics = compute_all_metrics(
            result["equity_curve"],
            result["returns"],
            result["regime_series"],
            result["trades"],
            initial_capital=request.capital,
        )

        # ── Step 7: Monte Carlo Validation ────────────────────────────────
        trade_returns = extract_trade_returns(
            result["trades"].to_dict(orient="records")
            if not result["trades"].empty else []
        )

        monte_carlo = run_monte_carlo(
            trade_returns=trade_returns,
            capital=request.capital,
            n_simulations=request.n_simulations,
        )

        # Override real_final_equity with actual backtest result
        # (more accurate than recomputing from trade returns)
        if "error" not in monte_carlo:
            monte_carlo["real_final_equity"] = round(float(result["equity_curve"].iloc[-1]), 2)
            monte_carlo["real_total_return"] = round(
                float((result["equity_curve"].iloc[-1] - request.capital) / request.capital), 4
            )

        # ── Step 8: Serialise Equity Curve ────────────────────────────────
        equity_series = result["equity_curve"]
        regime_series = result["regime_series"]
        equity_points = [
            {
                "date":   str(d.date()),
                "value":  round(float(v), 2),
                "regime": int(regime_series.get(d, 2)),
            }
            for d, v in equity_series.items()
        ]

        trades_list = (
            result["trades"].to_dict(orient="records")
            if not result["trades"].empty else []
        )

        # ── Step 9: Return Sanitized Response ─────────────────────────────
        return _sanitize({
            "status":        "success",
            "rules":         [r.model_dump() for r in request.rules],
            "regime_filter": request.regime_filter,
            "total_signals": int(total_signal),
            **metrics,
            "equity_curve":  equity_points,
            "trades":        trades_list,
            "monte_carlo":   monte_carlo,
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Sandbox error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators")
async def get_indicators():
    return get_available_indicators()