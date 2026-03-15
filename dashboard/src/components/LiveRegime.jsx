import { useEffect, useState } from "react"
import axios from "axios"

const REGIME_COLORS = {
  emerald: {
    bg:     "bg-emerald-900/20",
    border: "border-emerald-700",
    text:   "text-emerald-400",
    dot:    "bg-emerald-400",
    bar:    "#10b981",
  },
  red: {
    bg:     "bg-red-900/20",
    border: "border-red-700",
    text:   "text-red-400",
    dot:    "bg-red-400",
    bar:    "#ef4444",
  },
  yellow: {
    bg:     "bg-yellow-900/20",
    border: "border-yellow-700",
    text:   "text-yellow-400",
    dot:    "bg-yellow-400",
    bar:    "#f59e0b",
  },
}

const REGIME_BAR_COLORS = {
  0: "#10b981",
  1: "#ef4444",
  2: "#f59e0b",
}

export default function LiveRegime() {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/api/live")
      .then(res => {
        setData(res.data)
        setLoading(false)
      })
      .catch(err => {
        setError("Failed to fetch live regime")
        setLoading(false)
      })
  }, [])

  if (loading) {
    return (
      <div className="border-b border-gray-800 px-6 py-3 flex items-center gap-3">
        <div className="w-2 h-2 rounded-full bg-gray-600 animate-pulse" />
        <span className="text-xs text-gray-500">Loading live regime...</span>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="border-b border-gray-800 px-6 py-3 flex items-center gap-3">
        <div className="w-2 h-2 rounded-full bg-gray-600" />
        <span className="text-xs text-gray-500">Live regime unavailable</span>
      </div>
    )
  }

  const colors = REGIME_COLORS[data.regime_color] || REGIME_COLORS.yellow

  return (
    <div className={`border-b border-gray-800 px-6 py-3 ${colors.bg}`}>
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-6 flex-wrap">

        {/* Left — Current Regime */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className={`w-2.5 h-2.5 rounded-full ${colors.dot} animate-pulse`} />
            <span className="text-xs text-gray-400 uppercase tracking-wider">
              Live Regime
            </span>
          </div>
          <span className={`text-sm font-bold ${colors.text}`}>
            {data.regime_name}
          </span>
          <span className="text-xs text-gray-500">
            {data.consecutive_days} days
          </span>
        </div>

        {/* Center — VIX + Strategy */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">VIX</span>
            <span className={`text-sm font-semibold ${
              data.vix > 25 ? "text-red-400" :
              data.vix < 15 ? "text-emerald-400" :
              "text-yellow-400"
            }`}>
              {data.vix}
            </span>
            <span className="text-xs text-gray-600">
              ({data.vix_zscore > 0 ? "+" : ""}{data.vix_zscore}σ)
            </span>
          </div>

          <div className="hidden md:flex items-center gap-2">
            <span className="text-xs text-gray-500">Strategy</span>
            <span className="text-xs text-white font-medium">
              {data.active_strategy.name}
            </span>
          </div>
        </div>

        {/* Right — 60 Day History Bar */}
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500 hidden lg:block">
            60d history
          </span>
          <div className="flex gap-px items-end h-5">
            {data.history.map((d, i) => (
              <div
                key={i}
                className="w-1 rounded-sm opacity-80 hover:opacity-100 transition-opacity"
                style={{
                  height:           d.regime === 1 ? "100%" : d.regime === 0 ? "60%" : "80%",
                  backgroundColor:  REGIME_BAR_COLORS[d.regime],
                }}
                title={`${d.date}: ${d.name}`}
              />
            ))}
          </div>

          <span className="text-xs text-gray-600">
            {data.as_of_date}
          </span>
        </div>

      </div>
    </div>
  )
}