const PREDEFINED_PAIRS = [
  { a: "JPM",  b: "BAC",   sector: "Banks" },
  { a: "GS",   b: "MS",    sector: "Banks" },
  { a: "WFC",  b: "BAC",   sector: "Banks" },
  { a: "AAPL", b: "MSFT",  sector: "Tech" },
  { a: "AMZN", b: "GOOGL", sector: "Tech" },
  { a: "META", b: "GOOGL", sector: "Tech" },
  { a: "NVDA", b: "AMD",   sector: "Semis" },
  { a: "INTC", b: "AMD",   sector: "Semis" },
  { a: "XOM",  b: "CVX",   sector: "Energy" },
  { a: "XOM",  b: "COP",   sector: "Energy" },
  { a: "KO",   b: "PEP",   sector: "Consumer" },
  { a: "WMT",  b: "COST",  sector: "Consumer" },
  { a: "V",    b: "MA",    sector: "Payments" },
  { a: "JNJ",  b: "PFE",   sector: "Healthcare" },
  { a: "UNH",  b: "CVS",   sector: "Healthcare" },
]

const SECTOR_COLORS = {
  "Banks":      "bg-blue-900/40 border-blue-700 text-blue-400",
  "Tech":       "bg-purple-900/40 border-purple-700 text-purple-400",
  "Semis":      "bg-cyan-900/40 border-cyan-700 text-cyan-400",
  "Energy":     "bg-yellow-900/40 border-yellow-700 text-yellow-400",
  "Consumer":   "bg-green-900/40 border-green-700 text-green-400",
  "Payments":   "bg-emerald-900/40 border-emerald-700 text-emerald-400",
  "Healthcare": "bg-rose-900/40 border-rose-700 text-rose-400",
}

export default function ActivePairs({ tickers }) {
  // Filter to only pairs where both tickers are in the universe
  const tickerSet = new Set(tickers)
  const activePairs = PREDEFINED_PAIRS.filter(
    p => tickerSet.has(p.a) && tickerSet.has(p.b)
  )
  const inactivePairs = PREDEFINED_PAIRS.filter(
    p => !(tickerSet.has(p.a) && tickerSet.has(p.b))
  )

  return (
    <div className="flex flex-col gap-3">

      {/* Active pairs */}
      <div>
        <p className="text-xs text-gray-400 mb-2">
          ACTIVE PAIRS — {activePairs.length} of {PREDEFINED_PAIRS.length} available in your universe
        </p>
        {activePairs.length === 0 ? (
          <div className="bg-yellow-900/20 border border-yellow-800 rounded-lg p-3 text-xs text-yellow-400">
            No predefined pairs found in your universe. Add more tickers to activate pairs trading.
            Suggested additions: JPM, BAC, GS, V, MA, XOM, CVX, KO, PEP
          </div>
        ) : (
          <div className="flex flex-wrap gap-2">
            {activePairs.map((p, i) => (
              <div
                key={i}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-medium ${
                  SECTOR_COLORS[p.sector] || "bg-gray-800 border-gray-700 text-gray-400"
                }`}
              >
                <span>{p.a}</span>
                <span className="opacity-50">↔</span>
                <span>{p.b}</span>
                <span className="opacity-40 text-xs ml-1">({p.sector})</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Inactive pairs */}
      {inactivePairs.length > 0 && (
        <div>
          <p className="text-xs text-gray-600 mb-2">
            INACTIVE — tickers not in universe
          </p>
          <div className="flex flex-wrap gap-2">
            {inactivePairs.map((p, i) => (
              <div
                key={i}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-gray-800 text-xs text-gray-600"
              >
                <span>{p.a}</span>
                <span className="opacity-50">↔</span>
                <span>{p.b}</span>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  )
}