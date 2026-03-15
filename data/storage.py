"""
data/storage.py
---------------
SQLite persistence layer. Caches fetched price data so we don't
hammer yFinance on every backtest run.

Simple key-value style storage:
    - Cache prices by (tickers_hash, start, end)
    - Store backtest results for the dashboard to retrieve
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = Path("quantregime.db")


# ── Connection Helper ─────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    """Create tables if they don't exist. Call once at startup."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS price_cache (
                cache_key   TEXT PRIMARY KEY,
                ticker      TEXT NOT NULL,
                field       TEXT NOT NULL,
                data_json   TEXT NOT NULL,
                fetched_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS backtest_results (
                id              TEXT PRIMARY KEY,
                run_at          TEXT NOT NULL,
                config_json     TEXT NOT NULL,
                metrics_json    TEXT NOT NULL,
                equity_json     TEXT NOT NULL,
                trades_json     TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_price_ticker ON price_cache(ticker);
            CREATE INDEX IF NOT EXISTS idx_backtest_run ON backtest_results(run_at);
        """)
    logger.info(f"Database initialized at {DB_PATH}")


# ── Price Cache ───────────────────────────────────────────────────────────────

def _make_cache_key(ticker: str, field: str, start: str, end: str) -> str:
    raw = f"{ticker}|{field}|{start}|{end}"
    return hashlib.md5(raw.encode()).hexdigest()


def cache_prices(close: pd.DataFrame, start: str, end: str):
    """Save Close prices DataFrame to SQLite cache."""
    with get_connection() as conn:
        for ticker in close.columns:
            key = _make_cache_key(ticker, "close", start, end)
            series = close[ticker].dropna()
            data = {str(k.date()): v for k, v in series.items()}
            conn.execute(
                "INSERT OR REPLACE INTO price_cache VALUES (?,?,?,?,?)",
                (key, ticker, "close", json.dumps(data), datetime.utcnow().isoformat())
            )
    logger.info(f"Cached prices for {len(close.columns)} tickers")


def load_cached_prices(tickers: list, start: str, end: str) -> pd.DataFrame | None:
    """
    Load cached Close prices. Returns None if any ticker is missing from cache.
    """
    frames = {}
    with get_connection() as conn:
        for ticker in tickers:
            key = _make_cache_key(ticker, "close", start, end)
            row = conn.execute(
                "SELECT data_json FROM price_cache WHERE cache_key=?", (key,)
            ).fetchone()
            if row is None:
                logger.debug(f"Cache miss for {ticker}")
                return None
            data = json.loads(row["data_json"])
            frames[ticker] = pd.Series(data, dtype=float)

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    logger.info(f"Loaded {len(tickers)} tickers from cache")
    return df


# ── Backtest Results ──────────────────────────────────────────────────────────

def save_backtest_result(run_id: str, config: dict, result: dict):
    """Persist a backtest result for dashboard retrieval."""
    equity = result.get("equity_curve", pd.Series())
    trades = result.get("trades", pd.DataFrame())

    # Serialise equity curve
    equity_json = json.dumps({
        str(k.date()): round(v, 4) for k, v in equity.items()
    }) if isinstance(equity, pd.Series) else json.dumps({})

    # Serialise trades
    if isinstance(trades, pd.DataFrame) and not trades.empty:
        trades_serialisable = trades.copy()
        for col in trades_serialisable.select_dtypes(include=["datetime64"]).columns:
            trades_serialisable[col] = trades_serialisable[col].astype(str)
        trades_json = trades_serialisable.to_json(orient="records")
    else:
        trades_json = "[]"

    metrics = {k: v for k, v in result.items()
               if k not in ("equity_curve", "trades", "returns")}

    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO backtest_results VALUES (?,?,?,?,?,?)",
            (
                run_id,
                datetime.utcnow().isoformat(),
                json.dumps(config),
                json.dumps(metrics, default=str),
                equity_json,
                trades_json,
            )
        )
    logger.info(f"Backtest result saved: {run_id}")


def load_backtest_result(run_id: str) -> dict | None:
    """Retrieve a saved backtest result by run ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM backtest_results WHERE id=?", (run_id,)
        ).fetchone()

    if row is None:
        return None

    equity_raw = json.loads(row["equity_json"])
    equity = pd.Series(equity_raw)
    equity.index = pd.to_datetime(equity.index)

    return {
        "config": json.loads(row["config_json"]),
        "metrics": json.loads(row["metrics_json"]),
        "equity_curve": equity,
        "trades": pd.read_json(row["trades_json"], orient="records"),
        "run_at": row["run_at"],
    }


def list_backtest_results() -> list[dict]:
    """Return summary list of all saved backtests."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, run_at, config_json, metrics_json FROM backtest_results ORDER BY run_at DESC"
        ).fetchall()

    return [
        {
            "id": row["id"],
            "run_at": row["run_at"],
            "config": json.loads(row["config_json"]),
            "metrics": json.loads(row["metrics_json"]),
        }
        for row in rows
    ]
