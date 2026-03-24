"""
api/routes/backtest.py
----------------------
POST /api/backtest        — run a new backtest with controls
GET  /api/backtest/{id}   — retrieve saved result
GET  /api/backtest/list   — list all saved results
"""
from backtester.regime_audit import run_regime_audit, format_audit_for_api
from fastapi import APIRouter, HTTPException
from api.schemas import BacktestRequest
from data.fetcher import fetch_universe_data, validate_data
from data.processor import build_all_features
from data.storage import save_backtest_result, load_backtest_result, list_backtest_results
from regime.features import build_regime_features, select_hmm_features
from regime.classifier import RegimeClassifier
from strategies.router import RegimeRouter
from backtester.engine import BacktestEngine, BacktestConfig
from backtester.risk import compute_all_metrics, regime_attribution
from backtester.controls import run_all_controls, build_comparison_table, build_chart_data
import uuid
import numpy as np
import pandas as pd
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("")
async def run_backtest(request: BacktestRequest):
    """
    Run a full backtest + all 6 control strategies.

    Pipeline:
    1. Fetch data
    2. Build features
    3. Classify regimes (walk-forward HMM)
    4. Generate RegimeRoute signals
    5. Run RegimeRoute backtest
    6. Run all 6 control backtests
    7. Build comparison table
    8. Build combined chart data
    9. Return everything
    """
    run_id = str(uuid.uuid4())[:8]
    logger.info(f"Backtest {run_id} starting | tickers={request.tickers}")

    try:
        # ── Step 1: Fetch Data ─────────────────────────────────────────────
        logger.info("Step 1: Fetching data...")
        raw = fetch_universe_data(request.tickers, request.start_date, request.end_date)
        validate_data(raw)

        # ── Step 2: Build Features ─────────────────────────────────────────
        logger.info("Step 2: Building features...")
        features = build_all_features(raw)

        # ── Step 3: Classify Regimes ───────────────────────────────────────
        logger.info("Step 3: Classifying regimes...")
        regime_features = build_regime_features(features["vix"], features["returns"])
        hmm_features = select_hmm_features(regime_features.dropna())

        classifier = RegimeClassifier(method=request.regime_method)
        labels = classifier.fit_predict(hmm_features)

        # Align regime labels to full index
        full_regimes = pd.Series(2, index=features["close"].index)
        full_regimes.iloc[len(full_regimes) - len(labels):] = labels
        regime_array = full_regimes.values

        # ── Step 4: Generate RegimeRoute Signals ──────────────────────────
        logger.info("Step 4: Generating RegimeRoute signals...")
        regime_router = RegimeRouter(
            regime_map=request.regime_map,
            strategy_params=request.strategy_params,
        )
        weights = regime_router.generate_signals(features, regime_array)

        # ── Step 5: Run RegimeRoute Backtest ──────────────────────────────
        logger.info("Step 5: Running RegimeRoute backtest...")
        config = BacktestConfig(
            capital=request.capital,
            commission=request.commission,
            slippage=request.slippage,
            allow_short=request.allow_short,
            rebalance_frequency=request.rebalance_frequency,
        )
        engine = BacktestEngine(config)
        main_result = engine.run(weights, features, regime_array)

        # ── Step 6: Run All Controls ───────────────────────────────────────
        logger.info("Step 6: Running control strategies...")
        controls = run_all_controls(features, regime_array, config)

        # ── Step 7: Compute Metrics ────────────────────────────────────────
        logger.info("Step 7: Computing metrics...")
        main_metrics = compute_all_metrics(
            main_result["equity_curve"],
            main_result["returns"],
            main_result["regime_series"],
            main_result["trades"],
            initial_capital=request.capital,
        )

        regime_perf = regime_attribution(
            main_result["returns"],
            main_result["regime_series"],
        )

        # ── Step 8: Build Comparison Table ────────────────────────────────
        logger.info("Step 8: Building comparison table...")
        comparison_table = build_comparison_table(
            main_result["equity_curve"],
            main_metrics,
            controls,
        )
        # ── Step 9a: Run Regime Audit ──────────────────────────────────────
        logger.info("Step 9a: Running regime audit...")
        audit = run_regime_audit(features, regime_array, config)
        regime_audit = format_audit_for_api(audit)

        # ── Step 9: Build Combined Chart Data ─────────────────────────────
        logger.info("Step 9: Building chart data...")
        chart_data = build_chart_data(
            main_result["equity_curve"],
            controls,
        )

        # ── Serialise Equity Curve With Regime Overlay ────────────────────
        equity_series = main_result["equity_curve"]
        regime_series = main_result["regime_series"]
        equity_points = [
            {
                "date": str(d.date()),
                "value": round(v, 2),
                "regime": int(regime_series.get(d, 2)),
            }
            for d, v in equity_series.items()
        ]

        # ── Serialise Trades ───────────────────────────────────────────────
        trades_list = (
            main_result["trades"].to_dict(orient="records")
            if not main_result["trades"].empty
            else []
        )

        # ── Build Response ─────────────────────────────────────────────────
        response = {
            "run_id":        run_id,
            "status":        "success",
            **main_metrics,
            "regime_performance":  regime_perf.to_dict(orient="index"),
            "equity_curve":        equity_points,
            "trades":              trades_list,
            "comparison_table":    comparison_table,
            "comparison_chart":    chart_data,
            "regime_audit":        regime_audit,
            "tickers":       request.tickers,
        }

        # ── Save to DB ─────────────────────────────────────────────────────
        save_backtest_result(run_id, request.model_dump(), {
            **main_metrics,
            "equity_curve": main_result["equity_curve"],
            "trades":       main_result["trades"],
        })

        logger.info(f"Backtest {run_id} complete")
        return response

    except Exception as e:
        logger.error(f"Backtest {run_id} failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_results():
    return list_backtest_results()


@router.get("/{run_id}")
async def get_result(run_id: str):
    result = load_backtest_result(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Backtest {run_id} not found")
    return result