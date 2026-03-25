# RegimeRoute — Volatility Regime-Aware Equity Backtesting Engine

> A full-stack quantitative finance research platform that uses Hidden Markov Models to classify market volatility regimes and dynamically routes to optimal trading strategies. Built from scratch in Python and React.

**Live Demo:** [regime-route.vercel.app](https://regime-route.vercel.app) &nbsp;|&nbsp; **API:** [regime-route-api.onrender.com/docs](https://regime-route-api.onrender.com/docs) &nbsp;|&nbsp; **GitHub:** [KrishShah0505/Regime-Route](https://github.com/KrishShah0505/Regime-Route)

---

## The Core Idea

Most backtesting frameworks apply a single strategy uniformly across all market conditions. RegimeRoute takes a different approach: it first classifies *what kind of market environment we're in*, then routes to the strategy that has historically performed best in that environment.

```
Market Data → HMM Classifier → Regime Label → Strategy Router → Backtest Engine
                                    │
                     ┌──────────────┼──────────────┐
                     ▼              ▼               ▼
               Low Vol (0)    High Vol (1)   Transitional (2)
               Momentum       Circuit        EMA Trend
               Strategy       Breaker        Filter
```

The key research finding: **going flat during High Volatility outperforms all active strategies** in that regime. Regime attribution showed Sharpe of +0.913 in Low Vol vs −0.908 in High Vol — confirming that capital preservation during systemic stress is more valuable than any active signal.

---

## Research Findings

All results on 10-stock universe (AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, BAC, XOM, JNJ), 2015–2024, $100k starting capital.

| Strategy | Total Return | Sharpe | Max DD | Notes |
|---|---|---|---|---|
| **RegimeRoute** | **+10.0%** | **−0.064** | **−23.2%** | With circuit breaker |
| Static Momentum | −19.2% | −0.381 | −43.9% | No regime switching |
| Static Mean Reversion | −50.4% | −1.648 | −51.2% | Bleeds in all regimes |
| Static Trend Following | +21.9% | +0.066 | −19.3% | |
| SPY Buy & Hold | +632.1% | +1.016 | −33.2% | Bull market baseline |
| Random Regime | −52.1% | −1.229 | −52.9% | **HMM validation** |

**The most important comparison:** RegimeRoute (+10%) vs Random Regime (−52%) = **62% gap**. This proves the HMM classifier is detecting genuine market structure, not noise.

**Regime Attribution:**

| Regime | Days | Sharpe | Ann. Return | Key Finding |
|---|---|---|---|---|
| Low Vol | 580 (26%) | **+0.913** | +11.6% | Momentum works correctly |
| High Vol | 634 (28%) | −0.908 | −7.3% | All strategies bleed → circuit breaker |
| Transitional | 1050 (46%) | −0.023 | +1.5% | Trend filter is defensive |

---

## Architecture

```
RegimeRoute/
├── data/
│   ├── fetcher.py          # yFinance ingestion, single/multi-ticker normalisation
│   ├── processor.py        # Feature engineering (19 features, fully vectorized)
│   └── storage.py          # SQLite persistence layer
│
├── regime/
│   ├── classifier.py       # Walk-forward HMM (504-day train, 63-day refit)
│   └── features.py         # VIX z-score, realized vol, VIX/RV ratio, VIX change
│
├── strategies/
│   ├── base.py             # Abstract BaseStrategy (Open/Closed principle)
│   ├── momentum.py         # Time-series momentum — 12M formation, 1M skip
│   ├── mean_reversion.py   # Bollinger Band mean reversion (±2σ)
│   ├── trend_filter.py     # EMA 200/50 trend confirmation, 50% size
│   ├── rsi_divergence.py   # RSI divergence — price/momentum structural confirmation
│   ├── breakout.py         # Volatility breakout with volume confirmation
│   ├── pairs_trading.py    # Statistical pairs trading on spread z-score
│   ├── circuit_breaker.py  # Go flat — capital preservation in High Vol
│   └── router.py           # Vectorized regime dispatch
│
├── backtester/
│   ├── engine.py           # Fully vectorized backtester, 1-day execution lag
│   ├── risk.py             # Sharpe, Sortino, Calmar, max drawdown, regime attribution
│   ├── controls.py         # 6 control strategies for benchmarking
│   ├── regime_audit.py     # Strategy × regime performance matrix
│   ├── monte_carlo.py      # 10,000-path vectorized bootstrap validation
│   ├── portfolio.py        # Position tracking, turnover, HHI concentration
│   └── report.py           # Text report generation with monthly return table
│
├── api/
│   ├── main.py             # FastAPI app, CORS, startup
│   ├── schemas.py          # Pydantic request/response models
│   └── routes/
│       ├── backtest.py     # POST /api/backtest — full pipeline + controls
│       ├── sandbox.py      # POST /api/sandbox — custom rules + Monte Carlo
│       ├── live.py         # GET /api/live — real-time regime classification
│       ├── strategies.py   # GET /api/strategies — strategy metadata
│       └── regimes.py      # GET /api/regimes/{ticker} — historical regimes
│
└── dashboard/              # React 18 + Vite + Tailwind + Recharts
    └── src/
        ├── pages/
        │   ├── Backtest.jsx    # Configurable backtest form + regime map override
        │   ├── Dashboard.jsx   # Results: equity curve, comparison, audit, pairs
        │   └── Sandbox.jsx     # Custom strategy builder + Monte Carlo
        └── components/
            ├── LiveRegime.jsx      # Real-time regime bar (always visible)
            ├── PnLCurve.jsx        # Multi-strategy overlay chart
            ├── ComparisonTable.jsx # Head-to-head performance table
            ├── RegimeAudit.jsx     # Strategy × regime matrix with verdicts
            ├── ActivePairs.jsx     # Live pairs display with sector color coding
            └── MonteCarloChart.jsx # Distribution histogram + percentile bands
```

---

## Strategy Library

RegimeRoute includes 7 production-ready strategies, all selectable for any regime via the UI.

| Strategy | Best Regime | Logic |
|---|---|---|
| Time-Series Momentum | Low Vol | Long 12M winners, short losers. Cross-sectional ranking. |
| Mean Reversion | High Vol | Fade ±2σ Bollinger Band moves. |
| EMA Trend Filter | Transitional | Long above EMA200 with 50% position size. |
| RSI Divergence | High Vol | Price/RSI structural divergence — momentum turning before price confirms. |
| Volatility Breakout | Transitional | Price breaks N-day high/low with volume confirmation. Trend-following entry. |
| Pairs Trading | High Vol | Long/short correlated pairs on spread z-score divergence. Market neutral. |
| Circuit Breaker | High Vol | Go completely flat. Best High Vol result found in research. |

### Pairs Trading — Predefined Pairs (15 pairs across 7 sectors)

| Sector | Pairs |
|---|---|
| Banks | JPM/BAC, GS/MS, WFC/BAC |
| Tech | AAPL/MSFT, AMZN/GOOGL, META/GOOGL |
| Semiconductors | NVDA/AMD, INTC/AMD |
| Energy | XOM/CVX, XOM/COP |
| Consumer | KO/PEP, WMT/COST |
| Payments | V/MA |
| Healthcare | JNJ/PFE, UNH/CVS |

A pair is only active if both tickers are present in your universe. For best results, use 20+ tickers across multiple sectors.

---
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
## Technical Highlights

### Walk-Forward HMM Classification
Gaussian HMM with 3 hidden states fit on four features: VIX z-score, 21-day realized volatility, VIX/RV ratio, and 5-day VIX change. Walk-forward fitting (504-day training window, refit every 63 trading days) ensures no future data leakage. The classifier is regime-relative — a VIX of 27 may be Transitional if recent history included VIX at 80, which is more contextually accurate than a fixed threshold rule.

### Fully Vectorized Backtester
Zero Python loops over rows. All operations use NumPy broadcasting and pandas vectorized methods. Signal generation, position sizing, cost deduction, and equity curve construction are all single matrix operations. A `.shift(1)` enforces the execution lag — signals on day T execute on day T+1.

### Monte Carlo Validation
10,000 bootstrap simulations in a single matrix operation:
```python
simulated    = np.random.choice(trade_returns, size=(10_000, n_trades), replace=True)
equity_paths = capital * np.cumprod(1 + simulated, axis=1)  # all 10k paths at once
```
Runs in ~28ms. Produces a percentile rank verdict: if the real strategy beats 95%+ of random simulations, it has statistically significant edge.

### Drawdown Circuit Breaker
When the HMM classifies High Volatility, all position weights are set to zero via a dedicated strategy class — no special casing in the router:
```python
# CircuitBreakerStrategy.entry_signals()
return pd.DataFrame(0.0, index=close.index, columns=close.columns)
```
Result: Total Return improved from +6.3% to +10.0%, max drawdown reduced from −24.3% to −23.2%.

### Statistical Pairs Trading
Spread computed as log price ratio, z-scored over a rolling 60-day window:
```python
spread  = np.log(close[ticker_a]) - np.log(close[ticker_b])
z_score = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
# Entry: |z| > 2.0  |  Exit: |z| < 0.5
```
Each pair is dollar-neutral — long one leg, short the other — so market direction is irrelevant.

### Regime Audit Matrix
Every backtest produces a 3×3 strategy × regime performance matrix. The diagonal should be highest if regime assignments are optimal:

```
              Low Vol   High Vol   Transitional
Momentum       ★1.681    −0.466      −0.035     ✓
Mean Reversion −1.980    ★−0.093    −1.834      ✓
Trend Filter    1.429    −0.534      ★0.235      ✓
```

---

## Features

**Backtest Engine**
- Configurable universe, date range, capital, commission, slippage
- User-definable regime → strategy mapping (any of 7 strategies for any regime)
- HMM or Z-Score regime detection
- 6 automatic control strategies run alongside every backtest
- Full trade log with regime label at entry

**Strategy Comparison Dashboard**
- Multi-line equity curve overlaying all 7 strategies
- Head-to-head performance table with best-in-column highlighting
- Regime performance breakdown cards
- Strategy × Regime audit matrix with pass/fail verdicts
- Active pairs display with sector color coding

**Custom Strategy Sandbox**
- Rule builder: IF [indicator] [operator] [value] → BUY/SELL
- 10 supported indicators with auto-populated defaults and valid range hints
- Regime filter: test only in specific market conditions
- Monte Carlo validation with distribution histogram and edge verdict

**Live Regime Indicator**
- Real-time VIX data + HMM classification of today's market state
- Shows active strategy, VIX level, z-score, consecutive days in regime
- 60-day history bar, always visible across all pages

---

## Setup

```bash
# Clone
git clone https://github.com/KrishShah0505/Regime-Route.git
cd Regime-Route

# Backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000

# Frontend (new terminal)
cd dashboard
npm install
npm run dev
```

**Requirements:** Python 3.11+, Node 18+

On Windows, `hmmlearn` requires [VS C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/backtest` | Run full backtest + 6 controls |
| GET | `/api/backtest/list` | List saved backtest runs |
| GET | `/api/backtest/{run_id}` | Retrieve saved result |
| POST | `/api/sandbox` | Custom strategy + Monte Carlo |
| GET | `/api/sandbox/indicators` | Available rule builder indicators |
| GET | `/api/live` | Real-time regime classification |
| GET | `/api/strategies` | Strategy metadata |
| GET | `/api/strategies/regime-map` | Current regime → strategy mapping |
| GET | `/api/regimes/{ticker}` | Historical regime labels for ticker |

Full interactive docs: [regime-route-api.onrender.com/docs](https://regime-route-api.onrender.com/docs)

---

## Testing

```bash
python -m pytest tests/ -v
```

16 tests covering the backtester engine, risk metrics, and all three core strategies.

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11, JavaScript (React 18) |
| Regime Detection | hmmlearn (Gaussian HMM), scikit-learn |
| Data | yFinance (free, no API key) |
| Numerics | NumPy, Pandas (fully vectorized) |
| Backend | FastAPI, Pydantic v2, SQLite |
| Frontend | React 18, Vite 7, Tailwind CSS, Recharts |
| Deployment | Render (API), Vercel (dashboard) |
| Testing | pytest, pytest-cov |

---

## Known Limitations & Future Work

- **High Vol strategy:** All tested active strategies (Bollinger Bands, RSI Divergence, Volatility Breakout, Pairs Trading) underperform the circuit breaker in High Vol on this universe. An effective active High Vol strategy remains an open research question.
- **Pairs trading universe:** Works best with 20+ tickers across multiple sectors. With the default 10-stock universe, only 2-3 pairs are active.
- **SPY benchmark:** Buy-and-hold SPY returned +632% in the same period due to a decade-long bull market. RegimeRoute targets better risk-adjusted returns across full market cycles, not maximum returns in persistent uptrends.
- **Walk-forward optimization:** Strategy parameters are currently fixed. Walk-forward parameter optimization is a planned extension.
- **Factor decomposition:** Fama-French factor exposure analysis would quantify genuine alpha vs known risk premia.

---

*Built by Krish Shah*