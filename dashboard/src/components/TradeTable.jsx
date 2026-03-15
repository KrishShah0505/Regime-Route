import { useState } from "react"

const REGIME_NAMES = { 0: "Low Vol", 1: "High Vol", 2: "Transitional" }
const REGIME_COLORS = {
  0: "text-emerald-400",
  1: "text-red-400",
  2: "text-yellow-400",
}

export default function TradeTable({ trades }) {
  const [page, setPage] = useState(0)
  const pageSize = 10
  const total = trades.length
  const paged = trades.slice(page * pageSize, (page + 1) * pageSize)

  if (!trades || trades.length === 0) {
    return <p className="text-gray-400 text-sm">No trades recorded.</p>
  }

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-gray-400 border-b border-gray-800">
              <th className="text-left py-2 pr-4">Ticker</th>
              <th className="text-left py-2 pr-4">Direction</th>
              <th className="text-left py-2 pr-4">Entry</th>
              <th className="text-left py-2 pr-4">Exit</th>
              <th className="text-left py-2 pr-4">P&L</th>
              <th className="text-left py-2 pr-4">Days Held</th>
              <th className="text-left py-2 pr-4">Regime</th>
            </tr>
          </thead>
          <tbody>
            {paged.map((t, i) => (
              <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                <td className="py-2 pr-4 font-medium">{t.ticker}</td>
                <td className="py-2 pr-4">
                  <span className={`px-2 py-0.5 rounded text-xs ${
                    t.direction === "long" ? "bg-emerald-900/50 text-emerald-400" : "bg-red-900/50 text-red-400"
                  }`}>
                    {t.direction}
                  </span>
                </td>
                <td className="py-2 pr-4 text-gray-400">{t.entry_date?.split("T")[0]}</td>
                <td className="py-2 pr-4 text-gray-400">{t.exit_date?.split("T")[0]}</td>
                <td className={`py-2 pr-4 font-medium ${t.pnl_pct > 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {t.pnl_pct > 0 ? "+" : ""}{(t.pnl_pct * 100).toFixed(2)}%
                </td>
                <td className="py-2 pr-4 text-gray-400">{t.holding_days}d</td>
                <td className={`py-2 pr-4 text-xs ${REGIME_COLORS[t.regime_at_entry]}`}>
                  {REGIME_NAMES[t.regime_at_entry]}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between mt-4 text-sm text-gray-400">
        <span>{total} total trades</span>
        <div className="flex gap-2">
          <button
            onClick={() => setPage(p => Math.max(0, p - 1))}
            disabled={page === 0}
            className="px-3 py-1 rounded bg-gray-800 disabled:opacity-30 hover:bg-gray-700"
          >
            Prev
          </button>
          <span className="px-3 py-1">
            {page + 1} / {Math.ceil(total / pageSize)}
          </span>
          <button
            onClick={() => setPage(p => Math.min(Math.ceil(total / pageSize) - 1, p + 1))}
            disabled={page >= Math.ceil(total / pageSize) - 1}
            className="px-3 py-1 rounded bg-gray-800 disabled:opacity-30 hover:bg-gray-700"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  )
}   