import { useEffect, useState } from "react"
import PnLCurve from "../components/PnLCurve"
import RiskMetrics from "../components/RiskMetrics"
import RegimeChart from "../components/RegimeChart"
import TradeTable from "../components/TradeTable"
import ComparisonTable from "../components/ComparisonTable"
import RegimeAudit from "../components/RegimeAudit"
import ActivePairs from "../components/ActivePairs"

export default function Dashboard() {
  const [result, setResult] = useState(null)

  useEffect(() => {
    const saved = localStorage.getItem("lastResult")
    if (saved) setResult(JSON.parse(saved))
  }, [])

  if (!result) {
    return (
      <div className="text-center mt-20">
        <p className="text-gray-400">No results yet. Run a backtest first.</p>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto flex flex-col gap-6">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Backtest Results</h1>
          <p className="text-gray-400 text-sm mt-1">
            Run ID: {result.run_id} — {result.start_date} to {result.end_date} — {result.total_years} years
          </p>
        </div>
        <div className={`px-3 py-1 rounded-full text-xs font-semibold ${
          result.sharpe_ratio > 1 ? "bg-emerald-900 text-emerald-300" :
          result.sharpe_ratio > 0 ? "bg-yellow-900 text-yellow-300" :
          "bg-red-900 text-red-300"
        }`}>
          Sharpe {result.sharpe_ratio.toFixed(2)}
        </div>
      </div>

      {/* Risk Metrics */}
      <RiskMetrics result={result} />

      {/* Strategy Comparison Chart */}
      <div className="bg-gray-900 rounded-xl p-6">
        <h2 className="text-sm font-semibold text-gray-400 mb-1">
          STRATEGY COMPARISON
        </h2>
        <p className="text-xs text-gray-500 mb-4">
          RegimeRoute vs all control strategies — same universe, same period, same capital
        </p>
        <PnLCurve
          data={result.equity_curve}
          comparisonChart={result.comparison_chart}
          capital={result.initial_capital}
        />
      </div>

      {/* Comparison Table */}
      {result.comparison_table && (
        <div className="bg-gray-900 rounded-xl p-6">
          <h2 className="text-sm font-semibold text-gray-400 mb-1">
            PERFORMANCE COMPARISON TABLE
          </h2>
          <p className="text-xs text-gray-500 mb-4">
            Head-to-head metrics across all strategies
          </p>
          <ComparisonTable data={result.comparison_table} />
        </div>
      )}

      {/* Equity Curve — RegimeRoute only */}
      <div className="bg-gray-900 rounded-xl p-6">
        <h2 className="text-sm font-semibold text-gray-400 mb-4">
          REGIMEROUTE EQUITY CURVE
        </h2>
        <PnLCurve
          data={result.equity_curve}
          capital={result.initial_capital}
        />
      </div>

      {/* Regime Chart */}
      <div className="bg-gray-900 rounded-xl p-6">
        <h2 className="text-sm font-semibold text-gray-400 mb-4">REGIME OVERLAY</h2>
        <RegimeChart data={result.equity_curve} />
      </div>

      {/* Regime Performance */}
      <div className="bg-gray-900 rounded-xl p-6">
        <h2 className="text-sm font-semibold text-gray-400 mb-4">
          PERFORMANCE BY REGIME
        </h2>
        <div className="grid grid-cols-3 gap-4">
          {Object.entries(result.regime_performance).map(([name, stats]) => (
            <div key={name} className={`rounded-lg p-4 ${
              name === "Low Vol"      ? "bg-emerald-900/30 border border-emerald-800" :
              name === "High Vol"     ? "bg-red-900/30 border border-red-800" :
              "bg-yellow-900/30 border border-yellow-800"
            }`}>
              <p className="text-xs font-semibold text-gray-400 mb-3">
                {name.toUpperCase()}
              </p>
              <div className="flex flex-col gap-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Sharpe</span>
                  <span className={stats.sharpe > 0 ? "text-emerald-400" : "text-red-400"}>
                    {stats.sharpe.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Return</span>
                  <span className={stats.annualized_return > 0 ? "text-emerald-400" : "text-red-400"}>
                    {(stats.annualized_return * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Win Rate</span>
                  <span className="text-white">{(stats.win_rate * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Max DD</span>
                  <span className="text-red-400">{(stats.max_drawdown * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Days</span>
                  <span className="text-white">
                    {stats.days} ({(stats.pct_of_time * 100).toFixed(0)}%)
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Regime Audit */}
      {result.regime_audit && (
        <div className="bg-gray-900 rounded-xl p-6">
          <h2 className="text-sm font-semibold text-gray-400 mb-1">
            STRATEGY × REGIME AUDIT
          </h2>
          <p className="text-xs text-gray-500 mb-4">
            Which strategy performs best in each regime?
            The diagonal should be the highest values if regime assignments are correct.
          </p>
          <RegimeAudit data={result.regime_audit} />
        </div>
      )}

      {/* Active Pairs */}
      <div className="bg-gray-900 rounded-xl p-6">
        <h2 className="text-sm font-semibold text-gray-400 mb-1">
          PAIRS TRADING — ACTIVE PAIRS
        </h2>
        <p className="text-xs text-gray-500 mb-4">
          Pairs active in your universe. Both tickers must be present for a pair to trade.
          More tickers = more active pairs = better diversification.
        </p>
        <ActivePairs tickers={result.tickers || []} />
      </div>

      {/* Trade Table */}
      <div className="bg-gray-900 rounded-xl p-6">
        <h2 className="text-sm font-semibold text-gray-400 mb-4">TRADE LOG</h2>
        <TradeTable trades={result.trades} />
      </div>

    </div>
  )
}