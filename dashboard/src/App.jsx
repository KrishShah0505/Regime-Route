import { useState } from "react"
import Dashboard from "./pages/Dashboard"
import Backtest from "./pages/Backtest"
import Sandbox from "./pages/Sandbox"
import LiveRegime from "./components/LiveRegime"

export default function App() {
  const [page, setPage] = useState("backtest")

  return (
    <div className="min-h-screen bg-gray-950 text-white">

      {/* Nav */}
      <nav className="border-b border-gray-800 px-6 py-4 flex items-center gap-8">
        <span className="text-xl font-bold text-emerald-400">RegimeRoute</span>
        <button
          onClick={() => setPage("backtest")}
          className={`text-sm ${page === "backtest" ? "text-white" : "text-gray-400 hover:text-white"}`}
        >
          Backtest
        </button>
        <button
          onClick={() => setPage("dashboard")}
          className={`text-sm ${page === "dashboard" ? "text-white" : "text-gray-400 hover:text-white"}`}
        >
          Results
        </button>
        <button
          onClick={() => setPage("sandbox")}
          className={`text-sm ${page === "sandbox" ? "text-white" : "text-gray-400 hover:text-white"}`}
        >
          Sandbox
        </button>
      </nav>

      {/* Live Regime Bar — always visible */}
      <LiveRegime />

      {/* Page */}
      <main className="p-6">
        {page === "backtest" && <Backtest onResult={() => setPage("dashboard")} />}
        {page === "dashboard" && <Dashboard />}
        {page === "sandbox" && <Sandbox />}
      </main>

    </div>
  )
}