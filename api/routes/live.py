"""
api/routes/live.py
------------------
GET /api/regime/live — classify today's market regime in real time.

Fetches the last 2 years of VIX + price data, runs the HMM classifier,
and returns the current regime with supporting context.
"""

from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from data.fetcher import fetch_vix, fetch_universe_data, validate_data
from data.processor import build_all_features
from regime.features import build_regime_features, select_hmm_features
from regime.classifier import RegimeClassifier

router = APIRouter()
logger = logging.getLogger(__name__)

REGIME_NAMES  = {0: "Low Volatility", 1: "High Volatility", 2: "Transitional"}
REGIME_COLOR  = {0: "emerald", 1: "red", 2: "yellow"}
REGIME_STRATEGY = {
    0: {"name": "Time-Series Momentum", "action": "Long top 20% by 12M return, short bottom 20%"},
    1: {"name": "Circuit Breaker",      "action": "Flat — capital preservation mode"},
    2: {"name": "EMA Trend Filter",     "action": "Long above EMA200 with 50% position size"},
}

# Reference tickers for regime classification
REFERENCE_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "BAC", "XOM", "JNJ"]


@router.get("")
async def get_live_regime():
    """
    Classify today's market regime using the last 2 years of data.

    Pipeline:
        1. Fetch last 2 years VIX + price data
        2. Build regime features
        3. Run HMM classifier
        4. Return current regime + last 60 days history
    """
    try:
        end_date   = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

        logger.info(f"Live regime classification | {start_date} to {end_date}")

        # ── Fetch Data ─────────────────────────────────────────────────────
        raw      = fetch_universe_data(REFERENCE_TICKERS, start_date, end_date)
        features = build_all_features(raw)

        # ── Classify Regimes ───────────────────────────────────────────────
        regime_features = build_regime_features(features["vix"], features["returns"])
        hmm_features    = select_hmm_features(regime_features.dropna())

        classifier = RegimeClassifier(method="hmm")
        labels     = classifier.fit_predict(hmm_features)

        # Align to full index
        full_regimes = pd.Series(2, index=features["close"].index)
        full_regimes.iloc[len(full_regimes) - len(labels):] = labels

        # ── Current Regime ─────────────────────────────────────────────────
        current_regime    = int(full_regimes.iloc[-1])
        current_regime_name = REGIME_NAMES[current_regime]
        current_vix       = round(float(features["vix"].iloc[-1]), 2)
        current_date      = str(features["close"].index[-1].date())

        # ── Last 60 Days History ───────────────────────────────────────────
        last_60 = full_regimes.iloc[-60:]
        history = [
            {
                "date":   str(d.date()),
                "regime": int(r),
                "name":   REGIME_NAMES[int(r)],
            }
            for d, r in last_60.items()
        ]

        # ── Regime Distribution Last 60 Days ──────────────────────────────
        dist = last_60.value_counts().to_dict()
        distribution = {
            REGIME_NAMES[k]: int(v)
            for k, v in dist.items()
        }

        # ── VIX Context ────────────────────────────────────────────────────
        vix_series   = features["vix"].iloc[-60:]
        vix_zscore   = round(float(
            (current_vix - vix_series.mean()) / vix_series.std()
        ), 2)

        # ── Consecutive Days In Current Regime ────────────────────────────
        consecutive = 1
        for r in reversed(full_regimes.iloc[:-1].values):
            if int(r) == current_regime:
                consecutive += 1
            else:
                break

        return {
            "status":          "success",
            "as_of_date":      current_date,
            "current_regime":  current_regime,
            "regime_name":     current_regime_name,
            "regime_color":    REGIME_COLOR[current_regime],
            "vix":             current_vix,
            "vix_zscore":      vix_zscore,
            "consecutive_days": consecutive,
            "active_strategy": REGIME_STRATEGY[current_regime],
            "distribution":    distribution,
            "history":         history,
        }

    except Exception as e:
        logger.error(f"Live regime error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))