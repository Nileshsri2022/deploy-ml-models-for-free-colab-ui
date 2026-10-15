import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, api, getApiKey, getGatewayUrl, setApiKey, setGatewayUrl } from '../api/client'
import type { BootstrapInfo, RuntimeInfo, RuntimeTokenInfo } from '../api/types'
import {
  Banner,
  Card,
  CodeBlock,
  CopyButton,
  Empty,
  KeyValue,
  RuntimeBadge,
  Spinner,
  formatSeconds,
  relativeTime,
} from '../components/ui'
import { useStore } from '../state/store'

/**
 * The "attach a Colab VM" page — the only manual step in the whole system.
 *
 * Flow: set an API key → mint a runtime token → paste the generated cell into a
 * Colab notebook → the VM dials *out* to `WS /internal/relay`, so there is no
 * public ingress to configure and no SSH tunnel to keep alive.
 */
export default function ConnectPage() {
  const { status, toast, refresh } = useStore()
  const [bootstrap, setBootstrap] = useState<BootstrapInfo | null>(null)
  const [runtimes, setRuntimes] = useState<RuntimeInfo[]>([])
  const [label, setLabel] = useState('colab-runtime')
  const [ttlHours, setTtlHours] = useState(24)
  const [token, setToken] = useState<RuntimeTokenInfo | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [keyDraft, setKeyDraft] = useState(() => getApiKey())
  const [gatewayDraft, setGatewayDraft] = useState(() => getGatewayUrl())

  const load = useCallback(async () => {
    try {
      const [boot, list] = await Promise.all([
        api.get<BootstrapInfo>('/api/bootstrap'),
        api.get<{ runtimes: RuntimeInfo[] }>('/api/runtimes'),
      ])
      setBootstrap(boot)
      setRuntimes(list.runtimes ?? [])
      setError(null)
    } catch (err) {
      if (err instanceof ApiError && err.isAuthError) {
        setError('Set an API key with the manage scope below to mint runtime tokens.')
      } else {
        setError((err as Error).message)
      }
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const mint = async () => {
    setBusy(true)
    try {
      const result = await api.post<RuntimeTokenInfo>(
        '/api/runtimes/token',
        undefined,
        { label: label || 'colab-runtime', ttl_hours: ttlHours },
      )
      setToken(result)
      toast('success', 'Runtime token minted', 'paste the cell below into Colab')
      await load()
    } catch (err) {
      toast('error', 'Could not mint a token', (err as Error).message)
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const rotate = async (runtimeId: string) => {
    try {
      const result = await api.post<RuntimeTokenInfo>(`/api/runtimes/${runtimeId}/rotate`, undefined)
      setToken(result)
      toast('success', 'Token rotated', `runtime ${runtimeId} — the old token is now invalid`)
      await load()
    } catch (err) {
      toast('error', 'Rotate failed', (err as Error).message)
    }
  }

  const reconcile = async (runtimeId: string) => {
    try {
      const result = await api.post<{ pushed?: boolean; deployments?: unknown[] }>(
        `/api/runtimes/${runtimeId}/reconcile`,
      )
      toast('success', 'Desired state pushed', JSON.stringify(result).slice(0, 140))
    } catch (err) {
      toast('error', 'Reconcile failed', (err as Error).message)
    }
  }

  const revoke = async (runtimeId: string) => {
    if (!window.confirm(`Revoke ${runtimeId}? Its tunnel is dropped and the token stops working.`)) return
    try {
      await api.delete(`/api/runtimes/${runtimeId}`)
      toast('info', 'Runtime revoked')
      await load()
      await refresh()
    } catch (err) {
      toast('error', 'Revoke failed', (err as Error).message)
    }
  }

  const applyKey = () => {
    setApiKey(keyDraft)
    setGatewayUrl(gatewayDraft)
    toast('success', keyDraft ? 'API key saved' : 'API key cleared', 'stored in this browser only')
    setError(null)
    void load()
    void refresh()
  }

  const cell =
    token?.notebook_cell ??
    bootstrap?.notebook_cell ??
    '# mint a runtime token first (needs an API key with the manage scope)'

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">Connect a runtime</h1>
          <p className="page-sub">
            A runtime is any machine that can run the agent — a Colab session, a Kaggle kernel, a
            spare laptop. It dials out to the gateway over a websocket, so no ports, no public IP, no
            ngrok.
          </p>
        </div>
        <div className="row">
          <a className="btn" href={bootstrap?.colab_relaunch_url ?? 'https://colab.research.google.com'} target="_blank" rel="noreferrer">
            Open Colab ↗
          </a>
          <button type="button" onClick={() => void load()}>
            ↻ Refresh
          </button>
        </div>
      </div>

      {error && <Banner tone="red">{error}</Banner>}

      <Card
        title="1 · Credentials"
        sub="The browser talks to the control plane with an API key. It never leaves this device."
      >
        <div className="grid cols-2">
          <label className="field">
            <span>API key</span>
            <input
              type="password"
              value={keyDraft}
              onChange={(event) => setKeyDraft(event.target.value)}
              placeholder="CDP_API_…"
              autoComplete="off"
            />
            <div className="hint">
              Needs the <code>manage</code> scope.{' '}
              <Link to="/keys">Create or view keys</Link>.
            </div>
          </label>
          <label className="field">
            <span>Gateway URL override (optional)</span>
            <input
              value={gatewayDraft}
              onChange={(event) => setGatewayDraft(event.target.value)}
              placeholder="leave empty = same origin"
              autoComplete="off"
            />
            <div className="hint">
              Only set this if the UI is served from somewhere other than the gateway.
            </div>
          </label>
        </div>
        <div className="row">
          <button type="button" className="primary" onClick={applyKey}>
            Save credentials
          </button>
          <button
            type="button"
            className="ghost"
            onClick={() => {
              setKeyDraft('')
              setApiKey('')
              toast('info', 'API key cleared')
            }}
          >
            Clear key
          </button>
          {status ? (
            <span className="hint">
              connected to {getGatewayUrl() || window.location.origin} · gateway v{status.version}
            </span>
          ) : null}
        </div>
      </Card>

      <Card
        title="2 · Mint a runtime token"
        sub="Each token maps to one VM. Rotate it any time; the old one stops working immediately."
      >
        <div className="grid cols-2">
          <label className="field">
            <span>Label</span>
            <input value={label} onChange={(event) => setLabel(event.target.value)} placeholder="colab-runtime" />
            <div className="hint">Shown in the runtimes table so you can tell sessions apart.</div>
          </label>
          <label className="field">
            <span>Token lifetime (hours)</span>
            <input
              type="number"
              min={1}
              max={720}
              value={ttlHours}
              onChange={(event) => setTtlHours(Number(event.target.value) || 24)}
            />
            <div className="hint">
              Colab caps a session at ~12h of activity; a longer token simply lets the same VM re-attach
              after a refresh.
            </div>
          </label>
        </div>
        <div className="row">
          <button type="button" className="primary" disabled={busy} onClick={() => void mint()}>
            {busy ? <Spinner label="minting…" /> : 'Mint token'}
          </button>
          {token ? (
            <span className="hint">
              runtime <code className="mono">{token.runtime_id}</code> · expires{' '}
              {token.expires_at ? new Date(token.expires_at).toLocaleString() : 'never'}
            </span>
          ) : null}
        </div>

        {token ? (
          <div style={{ marginTop: 14 }}>
            <Banner tone="green">
              <strong>Shown once.</strong> Copy the token or the whole cell now — the gateway only keeps
              a hash of it.
            </Banner>
            <CodeBlock label="token" code={token.token} />
          </div>
        ) : null}
      </Card>

      <Card
        title="3 · Run this in Colab"
        sub="Paste into a code cell. It installs the agent, probes the hardware, and opens the tunnel in a background thread — the notebook stays interactive."
        actions={<CopyButton value={cell} label="Copy cell" />}
      >
        <CodeBlock code={cell} />
        {bootstrap ? (
          <div style={{ marginTop: 12 }}>
            <KeyValue
              items={[
                ['gateway url', <span className="mono">{bootstrap.gateway_url}</span>],
                ['relay websocket', <span className="mono">{bootstrap.relay_ws_url}</span>],
                ['protocol', `v${bootstrap.protocol_version}`],
                [
                  'already installed',
                  `${bootstrap.installed.runtimes} runtimes · ${bootstrap.installed.models} models · ${bootstrap.installed.endpoints} endpoints`,
                ],
              ]}
            />
          </div>
        ) : null}
        <p className="hint" style={{ marginTop: 10 }}>
          The agent survives the notebook being idle, and reconnects with exponential backoff + jitter
          forever. It cannot stop Colab from recycling a session after ~12 hours — when that happens,
          re-run the cell; nothing else needs redoing.
        </p>
      </Card>

      <Card
        title="4 · Attached runtimes"
        sub="State is derived from heartbeats: online → stale (no beat for 3× the interval) → offline."
        actions={
          <Link className="btn small" to="/runtimes">
            Details
          </Link>
        }
      >
        {runtimes.length === 0 ? (
          <Empty
            title="No runtimes yet"
            body="Mint a token above and run the cell in Colab. The row appears here within a couple of seconds."
          />
        ) : (
          <table>
            <thead>
              <tr>
                <th>Runtime</th>
                <th>State</th>
                <th>Loaded models</th>
                <th>Heartbeat</th>
                <th>Session left</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {runtimes.map((runtime) => (
                <tr key={runtime.id}>
                  <td>
                    <div className="mono small">{runtime.label || runtime.id}</div>
                    <div className="faint small">{runtime.id}</div>
                  </td>
                  <td>
                    <RuntimeBadge state={runtime.state} />
                  </td>
                  <td className="small">
                    {runtime.loaded_models.length ? (
                      <div className="pill-row">
                        {runtime.loaded_models.slice(0, 3).map((model) => (
                          <span key={model} className="badge blue">
                            {model}
                          </span>
                        ))}
                        {runtime.loaded_models.length > 3 ? (
                          <span className="faint">+{runtime.loaded_models.length - 3}</span>
                        ) : null}
                      </div>
                    ) : (
                      <span className="faint">none</span>
                    )}
                  </td>
                  <td className="small faint">{relativeTime(runtime.last_heartbeat_at)}</td>
                  <td className="small">{formatSeconds(runtime.seconds_until_session_end)}</td>
                  <td>
                    <div className="row" style={{ gap: 6 }}>
                      <button type="button" className="small" onClick={() => void reconcile(runtime.id)} title="Force-push desired state + warm-up samples">
                        reconcile
                      </button>
                      <button type="button" className="small" onClick={() => void rotate(runtime.id)}>
                        rotate
                      </button>
                      <button type="button" className="small danger" onClick={() => void revoke(runtime.id)}>
                        revoke
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </>
  )
}
