"""
data/fetcher.py
---------------
Responsible for all external data ingestion.
Uses yFinance (free, no API key) for OHLCV and VIX data.

Design notes:
- All functions return DataFrames with a clean DatetimeIndex
- Vectorized operations only — no row-by-row loops
- Results are cached to SQLite to avoid redundant network calls
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ── Universe Definitions ──────────────────────────────────────────────────────

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "AMD", "INTC", "CRM", "ORCL",
    "JPM", "BAC", "GS", "MS", "WFC",
    "XOM", "CVX", "COP",
    "JNJ", "UNH", "PFE", "MRK",
    "PG", "KO", "PEP", "WMT", "COST",
    "TSLA", "F", "GM",
    "V", "MA", "PYPL",
]

VIX_TICKER = "^VIX"


# ── Core Fetch Functions ──────────────────────────────────────────────────────

def fetch_ohlcv(
    tickers: list,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a list of tickers.
    Always returns a MultiIndex DataFrame regardless of universe size.
    """
    logger.info(f"Fetching OHLCV for {len(tickers)} tickers: {start} to {end}")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError(f"No data returned for tickers: {tickers}")

    # Normalise to MultiIndex only if not already MultiIndex
    # Newer yFinance versions return MultiIndex even for single tickers
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers])

    logger.info(f"Fetched {len(raw)} trading days")
    return raw


def fetch_close_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Convenience wrapper — returns just Close prices."""
    raw = fetch_ohlcv(tickers, start, end)
    close = raw["Close"].copy()
    close = close.dropna(how="all")
    close = close.ffill()
    return close


def fetch_vix(start: str, end: str) -> pd.Series:
    """Fetch VIX index — primary volatility regime input."""
    logger.info(f"Fetching VIX: {start} to {end}")

    raw = yf.download(
        tickers=VIX_TICKER,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError("Failed to fetch VIX data")

    vix = raw["Close"].squeeze()
    vix.name = "VIX"
    return vix.ffill()


def fetch_universe_data(tickers: list, start: str, end: str) -> dict:
    """
    Master fetch function — returns everything the pipeline needs.

    Always uses fetch_ohlcv() so single-ticker normalisation applies
    consistently regardless of universe size.

    Returns dict:
        'close'   : pd.DataFrame  (days x tickers)
        'volume'  : pd.DataFrame  (days x tickers)
        'vix'     : pd.Series
        'returns' : pd.DataFrame  daily pct returns (days x tickers)
    """
    logger.info("Starting full universe data fetch...")

    raw = fetch_ohlcv(tickers, start, end)

    close  = raw["Close"].ffill()
    volume = raw["Volume"].ffill()

    vix = fetch_vix(start, end)

    # Align VIX to same trading day index as prices
    vix = vix.reindex(close.index).ffill()

    # Vectorized daily returns
    returns = close.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)

    logger.info(
        f"Universe ready | Tickers: {len(tickers)} | Days: {len(close)} | "
        f"Range: {close.index[0].date()} to {close.index[-1].date()}"
    )

    return {
        "close":   close,
        "volume":  volume,
        "vix":     vix,
        "returns": returns,
    }


# ── Data Validation ───────────────────────────────────────────────────────────

def validate_data(data: dict) -> dict:
    """Sanity checks on fetched data. Returns same dict (chainable)."""
    close   = data["close"]
    returns = data["returns"]

    missing_pct = close.isna().mean()
    bad_tickers = missing_pct[missing_pct > 0.10].index.tolist()
    if bad_tickers:
        logger.warning(f"Tickers >10% missing data: {bad_tickers}")

    extreme = (returns.abs() > 0.5).any()
    extreme_tickers = extreme[extreme].index.tolist()
    if extreme_tickers:
        logger.warning(f"Tickers with >50% single-day moves: {extreme_tickers}")

    vix_missing = data["vix"].isna().sum()
    if vix_missing > 0:
        logger.warning(f"VIX has {vix_missing} missing values — forward-filled")

    logger.info("Data validation complete")
    return data


# ── Utility ───────────────────────────────────────────────────────────────────

def describe_universe(close: pd.DataFrame) -> pd.DataFrame:
    """Summary DataFrame for the loaded universe."""
    return pd.DataFrame({
        "start_date":  close.apply(lambda col: col.first_valid_index()),
        "end_date":    close.apply(lambda col: col.last_valid_index()),
        "missing_pct": close.isna().mean().round(4),
        "total_days":  close.notna().sum(),
        "start_price": close.iloc[0],
        "end_price":   close.iloc[-1],
    })