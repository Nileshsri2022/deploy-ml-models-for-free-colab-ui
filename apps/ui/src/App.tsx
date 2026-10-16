import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { Toasts } from './components/Toasts'
import { useStore } from './state/store'
import { formatSeconds } from './components/ui'
import CachePage from './pages/CachePage'
import ConnectPage from './pages/ConnectPage'
import Dashboard from './pages/Dashboard'
import EndpointsPage from './pages/EndpointsPage'
import KeysPage from './pages/KeysPage'
import LogsPage from './pages/LogsPage'
import ModelsPage from './pages/ModelsPage'
import PlaygroundPage from './pages/PlaygroundPage'
import RuntimesPage from './pages/RuntimesPage'

/**
 * Shell: sidebar navigation + the routes.
 *
 * Every page reads from the store (which polls the gateway and shelves the
 * last-known-good responses), so the UI keeps showing something useful even
 * while the gateway or a runtime is unreachable.
 */
export default function App() {
  const store = useStore()
  const status = store.status
  const online = status?.runtimes_online ?? 0
  const degraded = status?.runtimes_degraded ?? 0

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">C</span>
          <span>
            ColabDeploy
            <div className="brand-sub">control plane</div>
          </span>
        </div>

        <nav className="nav">
          <NavLink to="/" end>
            ⌂ Dashboard
          </NavLink>
          <NavLink to="/connect">🔌 Connect a runtime</NavLink>

          <div className="nav-section">Deploy</div>
          <NavLink to="/models">📦 Models</NavLink>
          <NavLink to="/endpoints">🌐 Endpoints</NavLink>
          <NavLink to="/playground">🧪 Playground</NavLink>

          <div className="nav-section">Operate</div>
          <NavLink to="/cache">⚡ Upstash cache</NavLink>
          <NavLink to="/runtimes">🖥 Runtimes</NavLink>
          <NavLink to="/logs">📜 Logs</NavLink>
          <NavLink to="/keys">🔑 API keys</NavLink>
        </nav>

        <div style={{ marginTop: 'auto', borderTop: '1px solid var(--border)', paddingTop: 12 }}>
          <div className="row" style={{ marginBottom: 6 }}>
            <span
              className="badge"
              style={{
                background: online ? 'var(--green-soft)' : 'var(--amber-soft)',
                color: online ? 'var(--green)' : 'var(--amber)',
              }}
            >
              <span className="dot" />
              {online ? `${online} runtime${online === 1 ? '' : 's'} online` : 'no runtime online'}
            </span>
          </div>
          <div className="faint small">
            {status?.healthy ? 'gateway healthy' : 'gateway unreachable'}
            {degraded ? ` · ${degraded} degraded` : ''} · up {formatSeconds(status?.uptime_s)}
          </div>
          <div className="faint small" style={{ marginTop: 2 }}>
            {store.hasKey ? 'authenticated' : 'no API key set'}
            {store.statusStale ? ' · cached view' : ''}
          </div>
        </div>
      </aside>

      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/connect" element={<ConnectPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/models/:modelId" element={<ModelsPage />} />
          <Route path="/endpoints" element={<EndpointsPage />} />
          <Route path="/endpoints/:endpointId" element={<EndpointsPage />} />
          <Route path="/playground" element={<PlaygroundPage />} />
          <Route path="/cache" element={<CachePage />} />
          <Route path="/runtimes" element={<RuntimesPage />} />
          <Route path="/logs" element={<LogsPage />} />
          <Route path="/keys" element={<KeysPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <Toasts />
    </div>
  )
}
