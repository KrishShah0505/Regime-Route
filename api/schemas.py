"""
api/schemas.py
--------------
Pydantic models for all API request and response bodies.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import IntEnum


class RegimeLabel(IntEnum):
    LOW_VOL = 0
    HIGH_VOL = 1
    TRANSITIONAL = 2


# ── Request Models ────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    tickers: list[str] = Field(
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        description="List of stock tickers to include in the backtest universe",
    )
    start_date: str = Field(default="2010-01-01", description="YYYY-MM-DD")
    end_date: str = Field(default="2024-01-01", description="YYYY-MM-DD")
    capital: float = Field(default=100_000.0, ge=1000)
    commission: float = Field(default=0.001, ge=0, le=0.05)
    slippage: float = Field(default=0.0005, ge=0, le=0.05)
    allow_short: bool = Field(default=True)
    rebalance_frequency: str = Field(default="daily")
    regime_method: str = Field(default="hmm", description="'hmm' or 'zscore'")

    # Optional regime → strategy override
    regime_map: Optional[dict[int, str]] = Field(
        default=None,
        description="Override default regime→strategy mapping. E.g. {0: 'momentum', 1: 'mean_reversion', 2: 'trend_filter'}"
    )

    # Optional strategy parameter overrides
    strategy_params: Optional[dict[str, dict]] = Field(
        default=None,
        description="Override strategy hyperparameters. E.g. {'momentum': {'lookback': 180}}"
    )


class RegimeQueryRequest(BaseModel):
    ticker: str = Field(default="SPY")
    start_date: str = Field(default="2015-01-01")
    end_date: str = Field(default="2024-01-01")
    method: str = Field(default="hmm")


# ── Response Models ───────────────────────────────────────────────────────────

class RegimePerformance(BaseModel):
    days: int
    pct_of_time: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    win_rate: float
    max_drawdown: float


class TradeRecord(BaseModel):
    ticker: str
    direction: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    holding_days: int
    regime_at_entry: int


class EquityPoint(BaseModel):
    date: str
    value: float
    regime: int


class BacktestResult(BaseModel):
    run_id: str
    status: str = "success"

    # Summary metrics
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    volatility: float
    final_equity: float
    initial_capital: float
    total_trades: int

    # Period
    start_date: str
    end_date: str
    total_years: float

    # Regime breakdown
    regime_performance: dict[str, RegimePerformance]

    # Time series (for charts)
    equity_curve: list[EquityPoint]

    # Trade log
    trades: list[TradeRecord]


class StrategyInfo(BaseModel):
    name: str
    description: str
    preferred_regimes: list[int]
    default_params: dict


class RegimePoint(BaseModel):
    date: str
    regime: int
    regime_name: str
    vix: float


class HealthResponse(BaseModel):
    status: str
    version: str
    strategies_available: list[str]
