import { ComposedChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Bar } from "recharts"

const REGIME_COLORS = {
  0: "#10b981",  // emerald — low vol
  1: "#ef4444",  // red — high vol
  2: "#f59e0b",  // amber — transitional
}

const REGIME_NAMES = {
  0: "Low Vol",
  1: "High Vol",
  2: "Transitional",
}

export default function RegimeChart({ data }) {
  const formatted = data.map(d => ({
    date: d.date,
    value: d.value,
    regime: d.regime,
    regimeColor: REGIME_COLORS[d.regime],
  }))

  return (
    <div>
      {/* Legend */}
      <div className="flex gap-4 mb-4">
        {Object.entries(REGIME_NAMES).map(([k, v]) => (
          <div key={k} className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: REGIME_COLORS[k] }} />
            <span className="text-xs text-gray-400">{v}</span>
          </div>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={formatted}>
          <XAxis
            dataKey="date"
            tick={{ fill: "#6b7280", fontSize: 11 }}
            tickLine={false}
            interval={200}
          />
          <YAxis hide />
          <Tooltip
            contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", borderRadius: 8 }}
            labelStyle={{ color: "#9ca3af", fontSize: 11 }}
            formatter={(v, n, props) => [REGIME_NAMES[props.payload.regime], "Regime"]}
          />
          <Bar dataKey="value" barSize={2}>
            {formatted.map((entry, i) => (
              <Cell key={i} fill={REGIME_COLORS[entry.regime]} opacity={0.8} />
            ))}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}