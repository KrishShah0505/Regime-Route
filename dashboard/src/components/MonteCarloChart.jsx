import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell
} from "recharts"

const VERDICT_COLORS = {
  emerald: { bg: "bg-emerald-900/30", border: "border-emerald-700", text: "text-emerald-400" },
  blue:    { bg: "bg-blue-900/30",    border: "border-blue-700",    text: "text-blue-400"    },
  yellow:  { bg: "bg-yellow-900/30",  border: "border-yellow-700",  text: "text-yellow-400"  },
  red:     { bg: "bg-red-900/30",     border: "border-red-700",     text: "text-red-400"     },
}

export default function MonteCarloChart({ data }) {
  if (!data) return null

  const colors = VERDICT_COLORS[data.verdict_color] || VERDICT_COLORS.yellow

  return (
    <div className="flex flex-col gap-6">

      {/* Verdict Banner */}
      <div className={`rounded-xl p-5 border ${colors.bg} ${colors.border}`}>
        <div className="flex items-center justify-between mb-2">
          <span className={`text-lg font-bold ${colors.text}`}>
            {data.verdict}
          </span>
          <span className={`text-2xl font-bold ${colors.text}`}>
            {data.percentile_rank}th percentile
          </span>
        </div>
        <p className="text-sm text-gray-300 leading-relaxed">
          {data.verdict_detail}
        </p>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg p-4">
          <p className="text-xs text-gray-400 mb-1">Your Final Equity</p>
          <p className="text-lg font-bold text-white">
  ${Math.max(0, data.real_final_equity).toLocaleString()}
    </p>
        </div>
        <div className="bg-gray-800 rounded-lg p-4">
          <p className="text-xs text-gray-400 mb-1">Median Simulation</p>
          <p className="text-lg font-bold text-gray-300">
            ${data.summary.median.toLocaleString()}
          </p>
        </div>
        <div className="bg-gray-800 rounded-lg p-4">
          <p className="text-xs text-gray-400 mb-1">Simulations Run</p>
          <p className="text-lg font-bold text-white">
            {data.n_simulations.toLocaleString()}
          </p>
        </div>
        <div className="bg-gray-800 rounded-lg p-4">
          <p className="text-xs text-gray-400 mb-1">Trades Sampled</p>
          <p className="text-lg font-bold text-white">
            {data.n_trades}
          </p>
        </div>
      </div>

      {/* Distribution Histogram */}
      <div>
        <p className="text-xs text-gray-400 mb-3">
          DISTRIBUTION OF FINAL EQUITY — {data.n_simulations.toLocaleString()} SIMULATIONS
        </p>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.histogram} barCategoryGap={0}>
            <XAxis
              dataKey="bin_start"
              tick={{ fill: "#6b7280", fontSize: 10 }}
              tickLine={false}
              tickFormatter={v => `$${(v / 1000).toFixed(0)}k`}
              interval={9}
            />
            <YAxis
              tick={{ fill: "#6b7280", fontSize: 10 }}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#111827",
                border: "1px solid #374151",
                borderRadius: 8,
                fontSize: 11,
              }}
              formatter={(v, n, props) => [
                `${v} simulations`,
                props.payload.is_real ? "← Your result" : "Count",
              ]}
              labelFormatter={v => `$${parseFloat(v).toLocaleString()}`}
            />
            <ReferenceLine
              x={data.real_final_equity}
              stroke="#10b981"
              strokeWidth={2}
              strokeDasharray="4 4"
            />
            <Bar dataKey="count">
              {data.histogram.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.is_real ? "#10b981" : "#3b82f6"}
                  opacity={entry.is_real ? 1 : 0.6}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-500 mt-2 text-center">
          Green bar = your result | Blue bars = random simulations
        </p>
      </div>

      {/* Percentile Summary */}
      <div>
        <p className="text-xs text-gray-400 mb-3">SIMULATION PERCENTILE BANDS</p>
        <div className="grid grid-cols-5 gap-2">
          {[
            { label: "5th",  key: "p5"  },
            { label: "25th", key: "p25" },
            { label: "50th", key: "p50" },
            { label: "75th", key: "p75" },
            { label: "95th", key: "p95" },
          ].map(p => (
            <div key={p.key} className="bg-gray-800 rounded-lg p-3 text-center">
              <p className="text-xs text-gray-500 mb-1">{p.label} pct</p>
              <p className="text-sm font-medium text-white">
                {isNaN(data.summary[p.key]) ? "N/A" : `$${(data.summary[p.key] / 1000).toFixed(1)}k`}
              </p>
            </div>
          ))}
        </div>
      </div>

    </div>
  )
}