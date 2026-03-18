"""
api/main.py
-----------
FastAPI application entrypoint.
Run with: uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import backtest, strategies, regimes, sandbox,live
from data.storage import initialize_db
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

app = FastAPI(
    title="QuantRegime API",
    description="Volatility Regime-Aware Equity Strategy Backtesting Engine",
    version="1.0.0",
)

# Allow React dashboard to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(backtest.router,   prefix="/api/backtest",    tags=["Backtest"])
app.include_router(strategies.router, prefix="/api/strategies",  tags=["Strategies"])
app.include_router(regimes.router,    prefix="/api/regimes",     tags=["Regimes"])
app.include_router(sandbox.router, prefix="/api/sandbox", tags=["Sandbox"])
app.include_router(live.router, prefix="/api/live", tags=["Live Regime"])


@app.on_event("startup")
async def startup():
    initialize_db()


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
