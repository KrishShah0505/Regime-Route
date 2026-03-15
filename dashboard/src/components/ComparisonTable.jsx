const METRICS = [
  { key: "total_return", label: "Total Return", format: v => `${(v * 100).toFixed(1)}%`, higher: true },
  { key: "cagr",         label: "CAGR",         format: v => `${(v * 100).toFixed(1)}%`, higher: true },
  { key: "sharpe",       label: "Sharpe",        format: v => v.toFixed(3),               higher: true },
  { key: "max_drawdown", label: "Max DD",        format: v => `${(v * 100).toFixed(1)}%`, higher: false },
  { key: "volatility",   label: "Volatility",    format: v => `${(v * 100).toFixed(1)}%`, higher: false },
  { key: "win_rate",     label: "Win Rate",      format: v => `${(v * 100).toFixed(1)}%`, higher: true },
  { key: "final_equity", label: "Final Equity",  format: v => `$${v.toLocaleString()}`,   higher: true },
]

const STRATEGY_COLORS = {
  "RegimeRoute ★":          "text-emerald-400",
  "Static Momentum":         "text-blue-400",
  "Static Mean Reversion":   "text-yellow-400",
  "Static Trend Following":  "text-purple-400",
  "SPY Buy & Hold":          "text-gray-300",
  "Equal Weight":            "text-cyan-400",
  "Random Regime":           "text-gray-500",
}

export default function ComparisonTable({ data }) {
  if (!data || data.length === 0) return null

  // Find best value per metric for highlighting
  const best = {}
  METRICS.forEach(m => {
    const vals = data.map(row => row[m.key]).filter(v => v != null)
    best[m.key] = m.higher ? Math.max(...vals) : Math.min(...vals)
  })

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-xs text-gray-400 border-b border-gray-800">
            <th className="text-left py-3 pr-6">Strategy</th>
            {METRICS.map(m => (
              <th key={m.key} className="text-right py-3 px-3">{m.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr
              key={i}
              className={`border-b border-gray-800/50 ${
                row.is_main ? "bg-emerald-900/10" : "hover:bg-gray-800/20"
              }`}
            >
              {/* Strategy Name */}
              <td className={`py-3 pr-6 font-medium ${
                STRATEGY_COLORS[row.name] || "text-white"
              }`}>
                {row.name}
              </td>

              {/* Metrics */}
              {METRICS.map(m => {
                const val = row[m.key]
                const isBest = val === best[m.key]
                const isPositive = m.higher ? val > 0 : val > -0.05

                return (
                  <td key={m.key} className="py-3 px-3 text-right">
                    <span className={`
                      ${isBest ? "font-bold" : ""}
                      ${isBest ? "text-white" : isPositive ? "text-gray-300" : "text-red-400"}
                    `}>
                      {isBest && (
                        <span className="text-yellow-400 mr-1">★</span>
                      )}
                      {m.format(val)}
                    </span>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>

      {/* Legend */}
      <p className="text-xs text-gray-500 mt-3">
        ★ = best value in column
      </p>
    </div>
  )
}