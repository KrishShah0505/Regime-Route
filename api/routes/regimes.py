from fastapi import APIRouter, HTTPException
from data.fetcher import fetch_universe_data
from data.processor import build_all_features
from regime.features import build_regime_features, select_hmm_features
from regime.classifier import RegimeClassifier, REGIME_NAMES

router = APIRouter()

@router.get("/{ticker}")
async def get_regimes(ticker: str, start: str = "2015-01-01", end: str = "2024-01-01", method: str = "hmm"):
    try:
        raw = fetch_universe_data([ticker], start, end)
        features = build_all_features(raw)
        regime_features = build_regime_features(features["vix"], features["returns"])
        hmm_features = select_hmm_features(regime_features.dropna())

        classifier = RegimeClassifier(method=method)
        labels = classifier.fit_predict(hmm_features)

        vix = features["vix"].values[-len(labels):]
        dates = features["close"].index[-len(labels):]

        return [
            {
                "date": str(d.date()),
                "regime": int(r),
                "regime_name": REGIME_NAMES.get(int(r), "Unknown"),
                "vix": round(float(v), 2),
            }
            for d, r, v in zip(dates, labels, vix)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
