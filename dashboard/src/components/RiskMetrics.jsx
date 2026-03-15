export default function RiskMetrics({ result }) {
  const metrics = [
    { label: "Total Return", value: `${(result.total_return * 100).toFixed(1)}%`, positive: result.total_return > 0 },
    { label: "CAGR", value: `${(result.cagr * 100).toFixed(1)}%`, positive: result.cagr > 0 },
    { label: "Sharpe Ratio", value: result.sharpe_ratio.toFixed(3), positive: result.sharpe_ratio > 0 },
    { label: "Sortino Ratio", value: result.sortino_ratio.toFixed(3), positive: result.sortino_ratio > 0 },
    { label: "Max Drawdown", value: `${(result.max_drawdown * 100).toFixed(1)}%`, positive: false },
    { label: "Calmar Ratio", value: result.calmar_ratio.toFixed(3), positive: result.calmar_ratio > 1 },
    { label: "Win Rate", value: `${(result.win_rate * 100).toFixed(1)}%`, positive: result.win_rate > 0.5 },
    { label: "Profit Factor", value: result.profit_factor.toFixed(3), positive: result.profit_factor > 1 },
    { label: "Volatility", value: `${(result.volatility * 100).toFixed(1)}%`, positive: false },
    { label: "Total Trades", value: result.total_trades, positive: true },
    { label: "Avg Holding Days", value: result.avg_holding_days, positive: true },
    { label: "Final Equity", value: `$${result.final_equity.toLocaleString()}`, positive: result.final_equity > result.initial_capital },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {metrics.map(m => (
        <div key={m.label} className="bg-gray-900 rounded-xl p-4">
          <p className="text-xs text-gray-400 mb-1">{m.label}</p>
          <p className={`text-xl font-bold ${m.positive ? "text-emerald-400" : "text-red-400"}`}>
            {m.value}
          </p>
        </div>
      ))}
    </div>
  )
}