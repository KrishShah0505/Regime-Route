# QuantRegime 📈

**Volatility Regime-Aware Equity Strategy Backtesting Engine**

> A production-grade quantitative finance research platform that detects market volatility regimes using Hidden Markov Models and automatically routes to the optimal trading strategy for each regime — then backtests the adaptive system across 20 years of historical data.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized-orange)
![React](https://img.shields.io/badge/React-18-61dafb)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## The Idea

Most retail backtesting frameworks test a single strategy across all market conditions. The problem: strategies that work brilliantly in calm, trending markets collapse in volatile, chaotic ones.

QuantRegime solves this by:
1. **Detecting** the current market volatility regime (Low / High / Transitional) using a Gaussian HMM on VIX and realized volatility features
2. **Auto-routing** to the optimal strategy for each regime (Momentum / Mean Reversion / Trend Filter)
3. **Backtesting** this adaptive system with full walk-forward validation — no lookahead bias
4. **Attributing** performance by regime so you can see exactly where the alpha comes from

---

## Architecture

```
data/           → yFinance ingestion, feature engineering, SQLite caching
regime/         → HMM classifier, walk-forward fitting, regime features
strategies/     → Abstract base + 3 strategies + extensible router
backtester/     → Vectorized engine, risk metrics, regime attribution
api/            → FastAPI serving results as JSON
dashboard/      → React frontend with interactive charts
tests/          → pytest suite with coverage
```

## Strategies

| Regime | Condition | Strategy | Logic |
|--------|-----------|----------|-------|
| 0 — Low Vol | VIX < 15 | Momentum | Long winners, short losers (12m lookback) |
| 1 — High Vol | VIX > 25 | Mean Reversion | Fade extreme Bollinger Band moves |
| 2 — Transitional | VIX 15–25 | Trend Filter | 200-EMA direction, 50% position size |

## Key Technical Features

- **Walk-forward regime fitting** — HMM re-trains every quarter on a rolling 2-year window. Zero lookahead leakage.
- **Fully vectorized** — No `iterrows()`. Signals computed as NumPy array operations across 20yr × 100 ticker universe in under 1 second.
- **Execution lag** — `.shift(1)` enforced before every return multiplication.
- **Extensible strategy registry** — Add a strategy by creating one file + one line in the router. Zero changes to existing code.
- **Regime attribution** — Sharpe/drawdown/win-rate decomposed by regime label.

## Quickstart

```bash
# Install
pip install -r requirements.txt

# Copy env
cp .env.example .env

# Run API
uvicorn api.main:app --reload --port 8000

# Run tests
pytest tests/ -v --cov=.

# Dashboard
cd dashboard && npm install && npm run dev
```

## Example Backtest Request

```bash
curl -X POST http://localhost:8000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "start_date": "2010-01-01",
    "end_date": "2024-01-01",
    "capital": 100000,
    "regime_method": "hmm"
  }'
```

## Adding a New Strategy

```python
# 1. Create strategies/my_strategy.py
class MyStrategy(BaseStrategy):
    name = "my_strategy"
    preferred_regimes = [0]
    default_params = {"window": 30}

    def entry_signals(self, features, regimes): ...
    def exit_signals(self, features, positions): ...
    def position_size(self, signals, features, capital): ...

# 2. Register in strategies/router.py
STRATEGY_REGISTRY["my_strategy"] = MyStrategy

# Done. Router, engine, API — zero changes needed.
```
