import sys
print(f"Python: {sys.version}")
from data.fetcher import fetch_universe_data
from data.processor import build_all_features
from regime.features import build_regime_features, select_hmm_features
from regime.classifier import RegimeClassifier
from strategies.router import RegimeRouter
from backtester.engine import BacktestEngine, BacktestConfig
from backtester.risk import compute_all_metrics
import numpy as np
import pandas as pd

print("Step 1: Fetching data...")
data = fetch_universe_data(["AAPL", "MSFT", "GOOGL"], "2018-01-01", "2023-01-01")
print(f"  Got {len(data['close'])} trading days for {len(data['close'].columns)} tickers")

print("Step 2: Building features...")
features = build_all_features(data)
print(f"  Features built: {list(features.keys())}")

print("Step 3: Classifying regimes...")
regime_features = build_regime_features(features["vix"], features["returns"])
hmm_features = select_hmm_features(regime_features.dropna())
classifier = RegimeClassifier(method="hmm")
labels = classifier.fit_predict(hmm_features)
unique, counts = np.unique(labels, return_counts=True)
print(f"  Regime distribution: { {int(u): int(c) for u, c in zip(unique, counts)} }")

print("Step 4: Generating signals...")
full_regimes = pd.Series(2, index=features["close"].index)
full_regimes.iloc[len(full_regimes) - len(labels):] = labels
router = RegimeRouter()
weights = router.generate_signals(features, full_regimes.values)
print(f"  Weights shape: {weights.shape}")

print("Step 5: Running backtest...")
config = BacktestConfig(capital=100_000)
engine = BacktestEngine(config)
result = engine.run(weights, features, full_regimes.values)

print("Step 6: Computing metrics...")
metrics = compute_all_metrics(
    result["equity_curve"],
    result["returns"],
    result["regime_series"],
    result["trades"],
)

print("\n========== RESULTS ==========")
print(f"  Total Return : {metrics['total_return']:.1%}")
print(f"  CAGR         : {metrics['cagr']:.1%}")
print(f"  Sharpe Ratio : {metrics['sharpe_ratio']:.2f}")
print(f"  Max Drawdown : {metrics['max_drawdown']:.1%}")
print(f"  Win Rate     : {metrics['win_rate']:.1%}")
print(f"  Total Trades : {metrics['total_trades']}")
print(f"  Final Equity : ${metrics['final_equity']:,.0f}")
print("=============================")
print("\nPhase 1 complete. Pipeline works end to end.")