import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend
} from "recharts"

const STRATEGY_COLORS = {
  "RegimeRoute":             "#10b981",
  "Static Momentum":         "#3b82f6",
  "Static Mean Reversion":   "#f59e0b",
  "Static Trend Following":  "#8b5cf6",
  "SPY Buy & Hold":          "#e5e7eb",
  "Equal Weight":            "#06b6d4",
  "Random Regime":           "#6b7280",
}

export default function PnLCurve({ data, comparisonChart, capital }) {

  // If we have comparison chart data use that, otherwise fall back to single curve
  const chartData = comparisonChart || data.map(d => ({
    date: d.date,
    "RegimeRoute": d.value,
  }))

  const strategies = comparisonChart
    ? Object.keys(chartData[0]).filter(k => k !== "date")
    : ["RegimeRoute"]

  return (
    <ResponsiveContainer width="100%" height={350}>
      <LineChart data={chartData}>
        <XAxis
          dataKey="date"
          tick={{ fill: "#6b7280", fontSize: 11 }}
          tickLine={false}
          interval={200}
        />
        <YAxis
          tick={{ fill: "#6b7280", fontSize: 11 }}
          tickLine={false}
          tickFormatter={v => `$${(v / 1000).toFixed(0)}k`}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#111827",
            border: "1px solid #374151",
            borderRadius: 8,
            fontSize: 12,
          }}
          formatter={(v, name) => [`$${parseFloat(v).toLocaleString()}`, name]}
        />
        <Legend
          wrapperStyle={{ fontSize: 12, paddingTop: 16 }}
        />
        <ReferenceLine
          y={capital}
          stroke="#374151"
          strokeDasharray="4 4"
        />
        {strategies.map(name => (
          <Line
            key={name}
            type="monotone"
            dataKey={name}
            stroke={STRATEGY_COLORS[name] || "#ffffff"}
            strokeWidth={name === "RegimeRoute" ? 2.5 : 1.5}
            dot={false}
            opacity={name === "RegimeRoute" ? 1 : 0.7}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}