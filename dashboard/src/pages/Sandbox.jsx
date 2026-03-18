import { useState, useEffect } from "react"
import axios from "axios"
import MonteCarloChart from "../components/MonteCarloChart"
import PnLCurve from "../components/PnLCurve"
import RiskMetrics from "../components/RiskMetrics"
import API_BASE from "../api/client"

const OPERATORS = ["<", ">", "<=", ">=", "=="]
const ACTIONS = ["buy", "sell"]
const REGIME_OPTIONS = [
  { value: "",  label: "All Regimes" },
  { value: "0", label: "Low Vol Only" },
  { value: "1", label: "High Vol Only" },
  { value: "2", label: "Transitional Only" },
]

const DEFAULT_TICKERS = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,JPM,BAC,XOM,JNJ"

const DEFAULT_RULES = [
  { indicator: "rsi", operator: "<", value: 30,  action: "buy"  },
  { indicator: "rsi", operator: ">", value: 70,  action: "sell" },
]

export default function Sandbox() {
  const [indicators, setIndicators] = useState([])
  const [rules, setRules]           = useState(DEFAULT_RULES)
  const [tickers, setTickers]       = useState(DEFAULT_TICKERS)
  const [startDate, setStartDate]   = useState("2015-01-01")
  const [endDate, setEndDate]       = useState("2024-01-01")
  const [capital, setCapital]       = useState(100000)
  const [regimeFilter, setRegimeFilter] = useState("")
  const [allowShort, setAllowShort] = useState(true)
  const [nSims, setNSims]           = useState(10000)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)
  const [result, setResult]         = useState(null)

  useEffect(() => {
    axios.get(`${API_BASE}/api/sandbox/indicators`)
      .then(res => setIndicators(res.data))
      .catch(() => {})
  }, [])

  // ── Rule Management ──────────────────────────────────────────────────────

  const addRule = () => {
    setRules(r => [...r, { indicator: "rsi", operator: "<", value: 30, action: "buy" }])
  }

  const removeRule = (i) => {
    setRules(r => r.filter((_, idx) => idx !== i))
  }

  const updateRule = (i, field, value) => {
    setRules(r => r.map((rule, idx) => {
      if (idx !== i) return rule
      // When indicator changes, auto-populate value with its default
      if (field === "indicator") {
        const ind = indicators.find(ind => ind.id === value)
        return { ...rule, indicator: value, value: ind?.default_value ?? 0 }
      }
      return { ...rule, [field]: value }
    }))
  }

  // ── Run ──────────────────────────────────────────────────────────────────

  const run = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await axios.post(`${API_BASE}/api/sandbox`, {
        tickers:       tickers.split(",").map(t => t.trim()),
        start_date:    startDate,
        end_date:      endDate,
        capital:       parseFloat(capital),
        rules:         rules.map(r => ({ ...r, value: parseFloat(r.value) })),
        regime_filter: regimeFilter === "" ? null : parseInt(regimeFilter),
        allow_short:   allowShort,
        n_simulations: parseInt(nSims),
        commission:    0.001,
        slippage:      0.0005,
      })
      setResult(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="max-w-7xl mx-auto flex flex-col gap-6">

      <div>
        <h1 className="text-2xl font-bold">Strategy Sandbox</h1>
        <p className="text-gray-400 text-sm mt-1">
          Define your own rules, backtest them, and validate with Monte Carlo simulation.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* ── Left Panel: Config ─────────────────────────────────────────── */}
        <div className="lg:col-span-1 flex flex-col gap-4">

          {/* Universe */}
          <div className="bg-gray-900 rounded-xl p-5">
            <h2 className="text-xs font-semibold text-gray-400 mb-4">UNIVERSE</h2>
            <div className="flex flex-col gap-3">
              <div>
                <label className="text-xs text-gray-400 mb-1 block">Tickers</label>
                <input
                  className="w-full bg-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
                  value={tickers}
                  onChange={e => setTickers(e.target.value)}
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs text-gray-400 mb-1 block">Start</label>
                  <input
                    type="date"
                    className="w-full bg-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
                    value={startDate}
                    onChange={e => setStartDate(e.target.value)}
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-400 mb-1 block">End</label>
                  <input
                    type="date"
                    className="w-full bg-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
                    value={endDate}
                    onChange={e => setEndDate(e.target.value)}
                  />
                </div>
              </div>
              <div>
                <label className="text-xs text-gray-400 mb-1 block">Capital ($)</label>
                <input
                  type="number"
                  className="w-full bg-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
                  value={capital}
                  onChange={e => setCapital(e.target.value)}
                />
              </div>
            </div>
          </div>

          {/* Options */}
          <div className="bg-gray-900 rounded-xl p-5">
            <h2 className="text-xs font-semibold text-gray-400 mb-4">OPTIONS</h2>
            <div className="flex flex-col gap-3">
              <div>
                <label className="text-xs text-gray-400 mb-1 block">Regime Filter</label>
                <select
                  className="w-full bg-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
                  value={regimeFilter}
                  onChange={e => setRegimeFilter(e.target.value)}
                >
                  {REGIME_OPTIONS.map(o => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 mb-1 block">Monte Carlo Simulations</label>
                <select
                  className="w-full bg-gray-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
                  value={nSims}
                  onChange={e => setNSims(e.target.value)}
                >
                  <option value="1000">1,000 (fast)</option>
                  <option value="10000">10,000 (default)</option>
                  <option value="50000">50,000 (slow)</option>
                </select>
              </div>
              <div className="flex items-center justify-between">
                <label className="text-xs text-gray-400">Allow Short Selling</label>
                <button
                  onClick={() => setAllowShort(s => !s)}
                  className={`w-10 h-5 rounded-full transition-colors ${
                    allowShort ? "bg-emerald-500" : "bg-gray-600"
                  }`}
                >
                  <div className={`w-4 h-4 bg-white rounded-full mx-0.5 transition-transform ${
                    allowShort ? "translate-x-5" : "translate-x-0"
                  }`} />
                </button>
              </div>
            </div>
          </div>

        </div>

        {/* ── Right Panel: Rule Builder ──────────────────────────────────── */}
        <div className="lg:col-span-2 flex flex-col gap-4">

          <div className="bg-gray-900 rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xs font-semibold text-gray-400">RULE BUILDER</h2>
              <button
                onClick={addRule}
                className="text-xs text-emerald-400 hover:text-emerald-300 border border-emerald-800 rounded-lg px-3 py-1"
              >
                + Add Rule
              </button>
            </div>

            {/* Rules */}
            <div className="flex flex-col gap-3">
              {rules.map((rule, i) => {
                const indMeta = indicators.find(ind => ind.id === rule.indicator)
                return (
                  <div key={i} className="flex items-start gap-2 bg-gray-800 rounded-lg p-3">

                    {/* IF label */}
                    <span className="text-xs text-gray-500 w-6 shrink-0 pt-2">
                      {i === 0 ? "IF" : "AND"}
                    </span>

                    {/* Indicator */}
                    <select
                      className="bg-gray-700 rounded px-2 py-1.5 text-sm flex-1 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                      value={rule.indicator}
                      onChange={e => updateRule(i, "indicator", e.target.value)}
                    >
                      {indicators.map(ind => (
                        <option key={ind.id} value={ind.id}>{ind.label}</option>
                      ))}
                    </select>

                    {/* Operator */}
                    <select
                      className="bg-gray-700 rounded px-2 py-1.5 text-sm w-16 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                      value={rule.operator}
                      onChange={e => updateRule(i, "operator", e.target.value)}
                    >
                      {OPERATORS.map(op => (
                        <option key={op} value={op}>{op}</option>
                      ))}
                    </select>

                    {/* Value + Range hint */}
                    <div className="flex flex-col w-20">
                      <input
                        type="number"
                        className="bg-gray-700 rounded px-2 py-1.5 text-sm w-full focus:outline-none focus:ring-1 focus:ring-emerald-500"
                        value={rule.value}
                        onChange={e => updateRule(i, "value", e.target.value)}
                      />
                      {indMeta?.range && (
                        <span className="text-xs text-gray-600 mt-0.5 text-center">
                          {indMeta.range[0]}–{indMeta.range[1]}
                        </span>
                      )}
                    </div>

                    {/* Action */}
                    <select
                      className={`rounded px-2 py-1.5 text-sm w-16 focus:outline-none focus:ring-1 ${
                        rule.action === "buy"
                          ? "bg-emerald-900 text-emerald-300 focus:ring-emerald-500"
                          : "bg-red-900 text-red-300 focus:ring-red-500"
                      }`}
                      value={rule.action}
                      onChange={e => updateRule(i, "action", e.target.value)}
                    >
                      {ACTIONS.map(a => (
                        <option key={a} value={a}>{a.toUpperCase()}</option>
                      ))}
                    </select>

                    {/* Remove */}
                    <button
                      onClick={() => removeRule(i)}
                      className="text-gray-600 hover:text-red-400 text-lg leading-none ml-1 pt-1"
                    >
                      ×
                    </button>

                  </div>
                )
              })}
            </div>

            {/* Rule preview */}
            <div className="mt-4 bg-gray-800/50 rounded-lg p-3">
              <p className="text-xs text-gray-500 font-mono">
                {rules.length === 0
                  ? "No rules defined"
                  : rules.map((r, i) =>
                      `${i === 0 ? "IF" : "AND"} ${r.indicator} ${r.operator} ${r.value} → ${r.action.toUpperCase()}`
                    ).join("  |  ")
                }
              </p>
            </div>

            {/* Error */}
            {error && (
              <div className="mt-3 bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 text-sm text-red-300">
                {error}
              </div>
            )}

            {/* No signals warning */}
            {result?.status === "no_signals" && (
              <div className="mt-3 bg-yellow-900/40 border border-yellow-700 rounded-lg px-4 py-3 text-sm text-yellow-300">
                {result.message}
              </div>
            )}

            {/* Run Button */}
            <button
              onClick={run}
              disabled={loading || rules.length === 0}
              className="mt-4 w-full bg-emerald-500 hover:bg-emerald-400 disabled:bg-gray-700 disabled:text-gray-500 text-black font-semibold rounded-lg py-3 text-sm transition-colors"
            >
              {loading ? "Running..." : "Run Strategy →"}
            </button>
          </div>

        </div>
      </div>

      {/* ── Results ─────────────────────────────────────────────────────────── */}
      {result?.status === "success" && (
        <div className="flex flex-col gap-6">

          <RiskMetrics result={result} />

          <div className="bg-gray-900 rounded-xl p-6">
            <h2 className="text-sm font-semibold text-gray-400 mb-4">EQUITY CURVE</h2>
            <PnLCurve data={result.equity_curve} capital={result.initial_capital} />
          </div>

          {result.monte_carlo && (
            result.monte_carlo.error ? (
              <div className="bg-gray-900 rounded-xl p-6">
                <h2 className="text-sm font-semibold text-gray-400 mb-2">MONTE CARLO VALIDATION</h2>
                <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg px-4 py-3 text-sm text-yellow-300">
                  {result.monte_carlo.error} — {result.monte_carlo.n_trades} trades extracted.
                  Try a larger universe or longer date range for more trades.
                </div>
              </div>
            ) : (
              <div className="bg-gray-900 rounded-xl p-6">
                <h2 className="text-sm font-semibold text-gray-400 mb-1">MONTE CARLO VALIDATION</h2>
                <p className="text-xs text-gray-500 mb-4">
                  Your strategy vs {result.monte_carlo.n_simulations.toLocaleString()} random simulations of the same trades
                </p>
                <MonteCarloChart data={result.monte_carlo} />
              </div>
            )
          )}

        </div>
      )}

    </div>
  )
}