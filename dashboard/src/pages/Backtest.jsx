import { useState } from "react"
import axios from "axios"
import API_BASE from "../api/client"
const DEFAULT_TICKERS = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,JPM,BAC,XOM,JNJ"

export default function Backtest({ onResult }) {
  const [tickers, setTickers] = useState(DEFAULT_TICKERS)
  const [startDate, setStartDate] = useState("2015-01-01")
  const [endDate, setEndDate] = useState("2024-01-01")
  const [capital, setCapital] = useState(100000)
  const [method, setMethod] = useState("hmm")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const STRATEGIES = [
  { value: "momentum",       label: "Time-Series Momentum" },
  { value: "mean_reversion", label: "Mean Reversion (Bollinger)" },
  { value: "trend_filter",   label: "EMA Trend Filter" },
  { value: "rsi_divergence", label: "RSI Divergence" },
  {value: "circuit_breaker",label: "Circuit Breaker(Go Flat)"},
  { value: "breakout",        label: "Volatility Breakout" },
  { value: "pairs_trading",   label: "Pairs Trading" },
]

const REGIME_LABELS = {
  0: "🟢 Low Vol",
  1: "🔴 High Vol",
  2: "🟡 Transitional",
}
  const [regimeMap, setRegimeMap] = useState({
  0: "momentum",
  1: "circuit_breaker", 
  2: "trend_filter",
})
  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await axios.post(`${API_BASE}/api/backtest`, {
        tickers: tickers.split(",").map(t => t.trim()),
        start_date: startDate,
        end_date: endDate,
        capital: parseFloat(capital),
        commission: 0.001,
        slippage: 0.0005,
        allow_short: true,
        rebalance_frequency: "daily",
        regime_method: method,
        regime_map:regimeMap,
      })
      localStorage.setItem("lastResult", JSON.stringify(res.data))
      onResult()
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto mt-10">
      <h1 className="text-2xl font-bold mb-2">Run Backtest</h1>
      <p className="text-gray-400 text-sm mb-8">
        Configure your universe and parameters then fire the backtest.
      </p>

      <div className="bg-gray-900 rounded-xl p-6 flex flex-col gap-5">

        {/* Tickers */}
        <div>
          <label className="text-xs text-gray-400 mb-1 block">Tickers (comma separated)</label>
          <input
            className="w-full bg-gray-800 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
            value={tickers}
            onChange={e => setTickers(e.target.value)}
          />
        </div>

        {/* Dates */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-xs text-gray-400 mb-1 block">Start Date</label>
            <input
              type="date"
              className="w-full bg-gray-800 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
              value={startDate}
              onChange={e => setStartDate(e.target.value)}
            />
          </div>
          <div>
            <label className="text-xs text-gray-400 mb-1 block">End Date</label>
            <input
              type="date"
              className="w-full bg-gray-800 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
              value={endDate}
              onChange={e => setEndDate(e.target.value)}
            />
          </div>
        </div>

        {/* Capital */}
        <div>
          <label className="text-xs text-gray-400 mb-1 block">Starting Capital ($)</label>
          <input
            type="number"
            className="w-full bg-gray-800 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
            value={capital}
            onChange={e => setCapital(e.target.value)}
          />
        </div>

        {/* Regime Method */}
        <div>
          <label className="text-xs text-gray-400 mb-1 block">Regime Detection Method</label>
          <select
            className="w-full bg-gray-800 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
            value={method}
            onChange={e => setMethod(e.target.value)}
          >
            <option value="hmm">HMM (Hidden Markov Model)</option>
            <option value="zscore">Z-Score (Simpler)</option>
          </select>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}
        {/* Regime Strategy Map */}
<div>
  <label className="text-xs text-gray-400 mb-2 block">
    Regime → Strategy Map
  </label>
  <div className="flex flex-col gap-2">
    {[0, 1, 2].map(regime => (
      <div key={regime} className="flex items-center gap-3">
        <span className="text-xs text-gray-400 w-28 shrink-0">
          {REGIME_LABELS[regime]}
        </span>
        <select
          className="flex-1 bg-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
          value={regimeMap[regime]}
          onChange={e => setRegimeMap(m => ({ ...m, [regime]: e.target.value }))}
        >
          {STRATEGIES.map(s => (
            <option key={s.value} value={s.value}>{s.label}</option>
          ))}
        </select>
      </div>
    ))}
  </div>
</div>
        {/* Submit */}
        <button
          onClick={run}
          disabled={loading}
          className="bg-emerald-500 hover:bg-emerald-400 disabled:bg-gray-700 disabled:text-gray-500 text-black font-semibold rounded-lg py-3 text-sm transition-colors"
        >
          {loading ? "Running backtest..." : "Run Backtest →"}
        </button>

      </div>
    </div>
  )
}