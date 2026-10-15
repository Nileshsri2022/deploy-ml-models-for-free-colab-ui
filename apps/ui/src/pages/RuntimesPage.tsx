import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, api, cachedGet } from '../api/client'
import type { RuntimeInfo, RuntimeTokenInfo } from '../api/types'
import {
  Badge,
  Banner,
  Card,
  CodeBlock,
  CopyButton,
  DeploymentBadge,
  Empty,
  KeyValue,
  Meter,
  RuntimeBadge,
  Spinner,
  Stat,
  formatBytes,
  formatSeconds,
  formatTime,
  prettyJson,
  relativeTime,
} from '../components/ui'
import { useStore } from '../state/store'

/**
 * Runtimes: hardware, tunnel health, loaded models and the desired-state diff.
 *
 * This is where the "24/7" story is visible — session countdown, heartbeat age,
 * reconnect counters, and the models the VM currently has warm.
 */
export default function RuntimesPage() {
  const { toast, telemetry, refresh } = useStore()
  const [runtimes, setRuntimes] = useState<RuntimeInfo[]>([])
  const [stale, setStale] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [token, setToken] = useState<RuntimeTokenInfo | null>(null)

  const load = useCallback(async () => {
    try {
      const result = await cachedGet<{ runtimes: RuntimeInfo[] }>('/api/runtimes')
      setRuntimes(result.data.runtimes ?? [])
      setStale(result.stale)
      setError(null)
      setSelected((current) => current ?? result.data.runtimes?.[0]?.id ?? null)
    } catch (err) {
      setStale(true)
      setError(
        err instanceof ApiError && err.isAuthError
          ? 'An API key with the manage scope is required.'
          : (err as Error).message,
      )
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
    const timer = setInterval(() => void load(), 6000)
    return () => clearInterval(timer)
  }, [load])

  const detail = runtimes.find((r) => r.id === selected) ?? null
  const relay = telemetry?.relay
  const connected = relay?.connected ?? []

  const act = async (label: string, fn: () => Promise<void>) => {
    try {
      await fn()
      await load()
      await refresh()
    } catch (err) {
      toast('error', `${label} failed`, (err as Error).message)
    }
  }

  const rotate = (runtime: RuntimeInfo) =>
    act('Rotate', async () => {
      const result = await api.post<RuntimeTokenInfo>(`/api/runtimes/${runtime.id}/rotate`)
      setToken(result)
      toast('success', 'Token rotated', 'the old token stops working immediately')
    })

  const reconcile = (runtime: RuntimeInfo) =>
    act('Reconcile', async () => {
      const result = await api.post<Record<string, unknown>>(`/api/runtimes/${runtime.id}/reconcile`)
      toast('success', 'Desired state pushed', JSON.stringify(result).slice(0, 160))
    })

  const revoke = (runtime: RuntimeInfo) => {
    if (!window.confirm(`Revoke ${runtime.label || runtime.id}? Its tunnel drops and the token dies.`)) return
    return act('Revoke', async () => {
      await api.delete(`/api/runtimes/${runtime.id}`)
      toast('info', 'Runtime revoked')
    })
  }

  if (loading && runtimes.length === 0) {
    return (
      <Card>
        <Spinner label="loading runtimes…" />
      </Card>
    )
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">Runtimes</h1>
          <p className="page-sub">
            Each runtime is an outbound websocket from a VM to the gateway. No inbound ports, no
            public hostname — which is why this works on Colab at all.
          </p>
        </div>
        <div className="row">
          <button type="button" onClick={() => void load()}>
            ↻ Refresh
          </button>
          <Link className="btn primary" to="/connect">
            + Attach a runtime
          </Link>
        </div>
      </div>

      {error && <Banner tone="red">{error}</Banner>}
      {stale && !error && <Banner>Showing last-known-good runtimes — the control plane did not answer.</Banner>}

      <div className="grid cols-4">
        <Card>
          <Stat
            label="Attached"
            value={`${runtimes.filter((r) => r.state === 'online' || r.state === 'degraded').length} / ${runtimes.length}`}
            tone={runtimes.some((r) => r.state === 'online') ? 'green' : 'amber'}
          />
        </Card>
        <Card>
          <Stat label="Tunnel connections" value={relay?.connections ?? '—'} hint={`${relay?.disconnects ?? 0} disconnects`} />
        </Card>
        <Card>
          <Stat
            label="Relayed requests"
            value={relay?.requests ?? '—'}
            hint={`${relay?.timeouts ?? 0} timeouts · ${relay?.errors ?? 0} errors`}
          />
        </Card>
        <Card>
          <Stat label="In flight" value={relay?.pending_requests ?? '—'} hint="exec requests awaiting an agent reply" />
        </Card>
      </div>

      {runtimes.length === 0 ? (
        <Card>
          <Empty
            title="No runtimes registered"
            body="Mint a token and paste the generated cell into Colab. The VM connects out to the gateway and stays attached until the session ends."
            action={
              <Link className="btn primary" to="/connect">
                Connect a runtime
              </Link>
            }
          />
        </Card>
      ) : (
        <div className="grid cols-2" style={{ marginTop: 14 }}>
          <Card title={`Runtimes (${runtimes.length})`} sub="Click a row for hardware, session budget and loaded models.">
            <table>
              <thead>
                <tr>
                  <th>Runtime</th>
                  <th>State</th>
                  <th>Hardware</th>
                  <th>Beat</th>
                </tr>
              </thead>
              <tbody>
                {runtimes.map((runtime) => (
                  <tr
                    key={runtime.id}
                    style={{ cursor: 'pointer', background: runtime.id === selected ? 'var(--bg-inset)' : undefined }}
                    onClick={() => setSelected(runtime.id)}
                  >
                    <td>
                      <strong>{runtime.label || runtime.id}</strong>
                      <div className="faint small mono">{runtime.id}</div>
                    </td>
                    <td>
                      <RuntimeBadge state={runtime.state} />
                    </td>
                    <td className="small">
                      {runtime.capabilities.gpu ? (
                        <Badge tone="green">GPU</Badge>
                      ) : (
                        <Badge tone="gray">CPU</Badge>
                      )}{' '}
                      {runtime.capabilities.cpu_count ?? '?'} vCPU
                      {runtime.capabilities.ram_mb ? ` · ${(runtime.capabilities.ram_mb / 1024).toFixed(0)} GB` : ''}
                    </td>
                    <td className="small faint">{relativeTime(runtime.last_heartbeat_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <div>
            {detail ? (
              <>
                <Card
                  title={detail.label || detail.id}
                  actions={
                    <div className="row">
                      <RuntimeBadge state={detail.state} />
                      {detail.revoked ? <Badge tone="red">revoked</Badge> : null}
                    </div>
                  }
                  sub={`tunnel: ${detail.tunnel_kind ?? 'websocket'} · agent ${detail.capabilities.agent_version ?? '?'}`}
                >
                  <KeyValue
                    items={[
                      ['runtime id', <span className="mono">{detail.id}</span>],
                      ['registered', formatTime(detail.registered_at)],
                      ['session start', formatTime(detail.session_started_at)],
                      ['last heartbeat', `${relativeTime(detail.last_heartbeat_at)} (${formatSeconds(detail.seconds_since_heartbeat)} ago)`],
                      [
                        'session budget',
                        detail.seconds_until_session_end !== null ? (
                          <span>
                            {formatSeconds(detail.seconds_until_session_end)} left of {detail.session_max_hours}h
                          </span>
                        ) : (
                          'unknown'
                        ),
                      ],
                      ['gpu', detail.capabilities.gpu ? detail.capabilities.gpu_name ?? 'yes' : 'none'],
                      ['cuda', detail.capabilities.cuda_version],
                      ['python', detail.capabilities.python_version],
                      ['ram', detail.capabilities.ram_mb ? formatBytes(detail.capabilities.ram_mb * 1024 * 1024) : null],
                      ['disk free', detail.capabilities.disk_free_mb ? formatBytes(detail.capabilities.disk_free_mb * 1024 * 1024) : null],
                      ['adapters', detail.capabilities.adapters?.length ? detail.capabilities.adapters.join(', ') : null],
                      ['colab url', detail.colab_url ? <a href={detail.colab_url} target="_blank" rel="noreferrer">open ↗</a> : null],
                      ['error', detail.error ? <span style={{ color: 'var(--red)' }}>{detail.error}</span> : null],
                    ]}
                  />
                  {detail.seconds_until_session_end !== null ? (
                    <div style={{ marginTop: 10 }}>
                      <div className="faint small" style={{ marginBottom: 4 }}>
                        session budget
                      </div>
                      <Meter
                        value={Math.max(0, detail.seconds_until_session_end)}
                        max={Math.max(1, detail.session_max_hours * 3600)}
                      />
                    </div>
                  ) : null}
                  <div className="row" style={{ marginTop: 12 }}>
                    <button type="button" className="small" onClick={() => void reconcile(detail)}>
                      Push desired state
                    </button>
                    <button type="button" className="small" onClick={() => void rotate(detail)}>
                      Rotate token
                    </button>
                    <button type="button" className="small danger" onClick={() => void revoke(detail)}>
                      Revoke
                    </button>
                  </div>
                </Card>

                <Card title={`Loaded models (${detail.loaded_models.length})`} sub="Held in an LRU pool on the VM; warm means no cold-start on the next request.">
                  {detail.deployments.length === 0 ? (
                    <p className="hint">
                      Nothing deployed to this runtime yet. Deploy a model and it lands here within a
                      second if the tunnel is up.
                    </p>
                  ) : (
                    <table>
                      <thead>
                        <tr>
                          <th>Model</th>
                          <th>State</th>
                          <th>Warm</th>
                          <th>Warm-up</th>
                          <th>Error</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.deployments.map((deployment) => (
                          <tr key={deployment.id}>
                            <td className="small">
                              <span className="mono">{deployment.model_id}</span>
                              <div className="faint">v{deployment.model_version}</div>
                            </td>
                            <td>
                              <DeploymentBadge state={deployment.state} />
                            </td>
                            <td className="small">{deployment.warm ? <Badge tone="green">warm</Badge> : <Badge tone="gray">cold</Badge>}</td>
                            <td className="num small">{deployment.warmup_ms !== null ? `${Math.round(deployment.warmup_ms)} ms` : '—'}</td>
                            <td className="small" style={{ color: 'var(--red)' }}>
                              {deployment.error ?? ''}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </Card>

                {detail.desired_state ? (
                  <Card title="Desired state" sub={`revision ${detail.desired_state.revision} · warm-up top ${detail.desired_state.warmup_top_k}`}>
                    <CodeBlock code={prettyJson(detail.desired_state)} />
                  </Card>
                ) : null}
              </>
            ) : (
              <Card>
                <Empty title="Select a runtime" body="Hardware, session countdown and loaded models appear here." />
              </Card>
            )}
          </div>
        </div>
      )}

      {connected.length > 0 ? (
        <Card title="Live tunnels" sub="Straight from the relay — these sockets are carrying exec traffic right now.">
          <table>
            <thead>
              <tr>
                <th>Runtime</th>
                <th>Agent</th>
                <th>Connected for</th>
                <th>Idle</th>
              </tr>
            </thead>
            <tbody>
              {connected.map((socket) => (
                <tr key={socket.runtime_id}>
                  <td className="mono small">{socket.runtime_id}</td>
                  <td className="small">{socket.agent_version}</td>
                  <td className="small">{formatSeconds(socket.connected_s)}</td>
                  <td className="small">{formatSeconds(socket.idle_s)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      ) : null}

      {token ? (
        <Card
          title="New runtime token"
          sub="Shown once — the gateway only stores a hash."
          actions={<CopyButton value={token.notebook_cell ?? token.token} label="Copy cell" />}
        >
          <Banner tone="green">
            Re-run this cell in the notebook to re-attach the same VM (or a fresh one) without touching
            your deployments.
          </Banner>
          <CodeBlock label="notebook cell" code={token.notebook_cell ?? ''} />
        </Card>
      ) : null}
    </>
  )
}
