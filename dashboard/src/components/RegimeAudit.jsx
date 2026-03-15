const STRATEGY_COLORS = {
  "Momentum":       "text-blue-400",
  "Mean Reversion": "text-yellow-400",
  "Trend Filter":   "text-purple-400",
}

const REGIME_COLORS = {
  "Low Vol":       "text-emerald-400",
  "High Vol":      "text-red-400",
  "Transitional":  "text-yellow-400",
}

const EXPECTED = {
  "Low Vol":      "Momentum",
  "High Vol":     "Mean Reversion",
  "Transitional": "Trend Filter",
}

export default function RegimeAudit({ data }) {
  if (!data || !data.matrix) return null

  const regimes = ["Low Vol", "High Vol", "Transitional"]

  // Find best sharpe per regime for highlighting
  const bestSharpe = {}
  regimes.forEach(regime => {
    const sharpes = data.matrix
      .map(row => row[regime]?.sharpe)
      .filter(s => s != null)
    bestSharpe[regime] = Math.max(...sharpes)
  })

  return (
    <div className="flex flex-col gap-4">

      {/* Matrix Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-gray-400 border-b border-gray-800">
              <th className="text-left py-3 pr-6">Strategy</th>
              {regimes.map(r => (
                <th key={r} className={`text-center py-3 px-6 ${REGIME_COLORS[r]}`}>
                  {r.toUpperCase()}
                  <div className="text-gray-500 font-normal mt-0.5">
                    ({data.matrix[0]?.[r]?.days || 0} days)
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.matrix.map((row, i) => (
              <tr key={i} className="border-b border-gray-800/50">
                <td className={`py-3 pr-6 font-medium ${STRATEGY_COLORS[row.strategy]}`}>
                  {row.strategy}
                </td>
                {regimes.map(regime => {
                  const stats = row[regime]
                  const isBest = stats?.sharpe === bestSharpe[regime]
                  const isExpected = EXPECTED[regime] === row.strategy

                  return (
                    <td key={regime} className={`py-3 px-6 text-center ${
                      isBest ? "bg-white/5 rounded" : ""
                    }`}>
                      {stats ? (
                        <div className="flex flex-col gap-1">
                          <span className={`font-bold text-base ${
                            isBest ? "text-white" :
                            stats.sharpe > 0 ? "text-gray-300" : "text-red-400"
                          }`}>
                            {isBest && "★ "}
                            {stats.sharpe?.toFixed(3)}
                          </span>
                          <span className="text-xs text-gray-500">
                            {(stats.win_rate * 100).toFixed(0)}% win
                          </span>
                          <span className={`text-xs ${
                            stats.return > 0 ? "text-emerald-400" : "text-red-400"
                          }`}>
                            {(stats.return * 100).toFixed(1)}% ann.
                          </span>
                          {isExpected && (
                            <span className="text-xs text-gray-500 italic">
                              (expected)
                            </span>
                          )}
                        </div>
                      ) : (
                        <span className="text-gray-600">—</span>
                      )}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Verdict Cards */}
      <div className="grid grid-cols-3 gap-4 mt-2">
        {Object.entries(data.verdict).map(([regime, v]) => (
          <div key={regime} className={`rounded-lg p-4 border ${
            v.correct
              ? "bg-emerald-900/20 border-emerald-800"
              : "bg-red-900/20 border-red-800"
          }`}>
            <div className="flex items-center justify-between mb-2">
              <span className={`text-xs font-semibold ${REGIME_COLORS[regime]}`}>
                {regime.toUpperCase()}
              </span>
              <span className="text-lg">{v.correct ? "✓" : "✗"}</span>
            </div>
            <p className="text-sm font-medium text-white">
              Best: {v.best_strategy}
            </p>
            <p className="text-xs text-gray-400 mt-1">
              Sharpe {v.best_sharpe}
            </p>
            {!v.correct && (
              <p className="text-xs text-red-400 mt-2">
                Expected: {v.expected}
              </p>
            )}
          </div>
        ))}
      </div>

      {/* Interpretation */}
      <p className="text-xs text-gray-500 leading-relaxed">
        {data.interpretation}
      </p>

    </div>
  )
}