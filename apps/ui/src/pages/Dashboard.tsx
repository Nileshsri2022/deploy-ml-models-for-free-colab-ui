import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, api, cachedGet, shelfAgeLabel } from '../api/client'
import type { DeploymentInfo, EndpointInfo } from '../api/types'
import {
  Banner,
  Card,
  CodeBlock,
  DeploymentBadge,
  Empty,
  RuntimeBadge,
  Spinner,
  Stat,
  formatBytes,
  formatSeconds,
  prettyJson,
  relativeTime,
} from '../components/ui'
import { useStore } from '../state/store'

/**
 * Overview: is a runtime attached, what is serving, and how healthy the cache is.
 *
 * Everything here comes from `GET /v1/status` + `GET /api/telemetry`, both of
 * which the store already polls; the two extra fetches are the endpoint and
 * deployment lists, which go through `cachedGet` so a dropped control plane
 * still renders the last known truth.
 */
export default function Dashboard() {
  const { status, statusStale, statusAge, statusError, telemetry, logs, refresh, loading, toast } =
    useStore()
  const [endpoints, setEndpoints] = useState<EndpointInfo[]>([])
  const [deployments, setDeployments] = useState<DeploymentInfo[]>([])
  const [endpointsStale, setEndpointsStale] = useState(false)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const result = await cachedGet<{ endpoints: EndpointInfo[] }>('/api/endpoints')
        if (cancelled) return
        setEndpoints(result.data.endpoints ?? [])
        setEndpointsStale(result.stale)
      } catch (error) {
        if (!cancelled && !(error instanceof ApiError && error.isAuthError)) {
          // A 401 here just means "no key yet" - the banner handles that.
          setEndpointsStale(true)
        }
      }
      try {
        const result = await cachedGet<{ deployments: DeploymentInfo[] }>('/api/deployments')
        if (!cancelled) setDeployments(result.data.deployments ?? [])
      } catch {
        /* non-essential */
      }
    }
    void load()
    const timer = setInterval(load, 8000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [])

  const cache = status?.cache
  const summary = telemetry?.summary
  const runtimes = status?.runtimes ?? []
  const online = runtimes.filter((r) => r.state === 'online' || r.state === 'degraded')
  const serving = endpoints.filter((e) => e.status === 'ready' || e.deployment_state === 'ready')

  const saveCache = async () => {
    try {
      const result = await api.post<{ saved: boolean; backend: string }>('/api/cache/save')
      toast(
        result.saved ? 'success' : 'info',
        result.saved ? 'Cache snapshot written' : 'Cache backend cannot snapshot',
        `backend: ${result.backend}`,
      )
    } catch (error) {
      toast('error', 'Snapshot failed', (error as Error).message)
    }
  }

  if (loading && !status) {
    return (
      <div className="card">
        <Spinner label="loading control-plane status…" />
      </div>
    )
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">Dashboard</h1>
          <p className="page-sub">
            One gateway, any number of Colab VMs. Deploy a model, get an HTTP endpoint, and keep
            serving from cache when the notebook goes away.
          </p>
        </div>
        <div className="row">
          <button type="button" onClick={() => void refresh()}>
            ↻ Refresh
          </button>
          <Link className="btn primary" to="/connect">
            Connect a runtime
          </Link>
        </div>
      </div>

      {statusStale && (
        <Banner tone={statusError ? 'red' : 'amber'}>
          <strong>Showing last-known state{statusAge ? ` (${shelfAgeLabel(statusAge)})` : ''}.</strong>{' '}
          {statusError
            ? `The control plane is unreachable: ${statusError}.`
            : 'The control plane did not answer the last poll.'}{' '}
          Cached views stay readable by design — this is the state-retention guarantee, not a crash.
        </Banner>
      )}

      {runtimes.length === 0 && (
        <Banner tone="blue">
          No runtime is registered yet.{' '}
          <Link to="/connect">Mint a token and paste one cell into Colab</Link> to bring a GPU (or
          CPU) online — that is the only manual step.
        </Banner>
      )}
      {runtimes.length > 0 && online.length === 0 && (
        <Banner>
          Registered but nothing online. Colab sessions end; re-run the launch cell in{' '}
          <Link to="/connect">Connect a runtime</Link> (the same token works until it expires) and the
          agent will re-attach and re-warm your models automatically.
        </Banner>
      )}

      <div className="grid cols-4">
        <Card>
          <Stat
            label="Runtimes online"
            value={`${online.length} / ${runtimes.length}`}
            tone={online.length ? 'green' : 'amber'}
            hint={online[0] ? `${online[0].label} · ${online[0].capabilities.gpu ? 'GPU' : 'CPU'}` : 'none attached'}
          />
        </Card>
        <Card>
          <Stat
            label="Endpoints serving"
            value={`${serving.length} / ${status?.endpoints ?? endpoints.length}`}
            hint={`${status?.deployments_ready ?? deployments.filter((d) => d.state === 'ready').length} warm`}
          />
        </Card>
        <Card>
          <Stat
            label="Cache hit ratio"
            value={cache ? `${(cache.hit_ratio * 100).toFixed(1)}%` : '—'}
            tone={cache && cache.hit_ratio >= 0.5 ? 'green' : undefined}
            hint={cache ? `${cache.entries} entries · ${formatBytes(cache.size_bytes)}` : 'no cache yet'}
          />
        </Card>
        <Card>
          <Stat
            label="p95 latency"
            value={summary ? `${Math.round(summary.latency_p95_ms)} ms` : '—'}
            hint={summary ? `${summary.requests_total} requests · ${summary.requests_cached} cached` : 'no traffic yet'}
          />
        </Card>
      </div>

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <Card
          title="Runtimes"
          sub="Colab / Kaggle VMs attached over the outbound websocket tunnel."
          actions={
            <Link className="btn small" to="/runtimes">
              Manage
            </Link>
          }
        >
          {runtimes.length === 0 ? (
            <Empty
              title="No runtimes"
              body="A runtime is just a notebook running the agent. Mint a token and paste two lines into Colab."
              action={
                <Link className="btn primary" to="/connect">
                  Connect now
                </Link>
              }
            />
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Label</th>
                  <th>State</th>
                  <th>Hardware</th>
                  <th>Heartbeat</th>
                  <th>Session left</th>
                </tr>
              </thead>
              <tbody>
                {runtimes.map((runtime) => (
                  <tr key={runtime.id}>
                    <td>
                      <span className="mono">{runtime.label || runtime.id}</span>
                    </td>
                    <td>
                      <RuntimeBadge state={runtime.state} />
                    </td>
                    <td className="small">
                      {runtime.capabilities.gpu
                        ? runtime.capabilities.gpu_name ?? 'GPU'
                        : `${runtime.capabilities.cpu_count ?? '?'} vCPU`}
                      {runtime.capabilities.ram_mb
                        ? ` · ${(runtime.capabilities.ram_mb / 1024).toFixed(0)} GB`
                        : ''}
                    </td>
                    <td className="small faint">{relativeTime(runtime.last_heartbeat_at)}</td>
                    <td className="small">
                      {formatSeconds(runtime.seconds_until_session_end) || '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>

        <Card
          title="Endpoints"
          sub="The common API surface every model is exposed through."
          actions={
            <Link className="btn small" to="/endpoints">
              All endpoints
            </Link>
          }
        >
          {endpoints.length === 0 ? (
            <Empty
              title="Nothing deployed"
              body="Register a model (upload an artifact, or point at a Drive/HuggingFace URI) and deploy it in one click."
              action={
                <Link className="btn primary" to="/models">
                  Add a model
                </Link>
              }
            />
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Route</th>
                  <th>Model</th>
                  <th>Deployment</th>
                  <th>Cache</th>
                </tr>
              </thead>
              <tbody>
                {endpoints.slice(0, 6).map((endpoint) => (
                  <tr key={endpoint.id}>
                    <td className="mono small">/{endpoint.route}</td>
                    <td className="small">
                      {endpoint.model_name ?? endpoint.model_id}
                      <span className="faint"> v{endpoint.model_version}</span>
                    </td>
                    <td>
                      <DeploymentBadge state={endpoint.deployment_state} />
                    </td>
                    <td className="small faint">
                      ttl {formatSeconds(endpoint.cache_ttl_s)} · swr{' '}
                      {formatSeconds(endpoint.stale_while_revalidate_s)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {endpointsStale && endpoints.length > 0 ? (
            <p className="hint" style={{ marginTop: 8 }}>
              last-known-good list (control plane unreachable)
            </p>
          ) : null}
        </Card>
      </div>

      <Card
        title="Call an endpoint"
        sub="Every model answers at the same shape, whichever framework produced it."
        actions={
          <Link className="btn small" to="/playground">
            Open playground
          </Link>
        }
      >
        {serving[0] ? (
          <CodeBlock
            label={`POST ${serving[0].url ?? `/${serving[0].route}`}`}
            code={
              serving[0].curl ??
              `curl -X POST '${serving[0].url ?? `/${serving[0].route}`}' \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify(serving[0].schema_in?.example ?? {}, null, 0)}'`
            }
          />
        ) : (
          <CodeBlock
            label="POST /v1/models/{model_id}/predict"
            code={`curl -X POST '/v1/models/{model_id}/predict' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer <api key>' \\
  -d '{"inputs": {"text": "hello"}}'`}
          />
        )}
        <div className="row" style={{ marginTop: 12 }}>
          <button type="button" className="small" onClick={() => void saveCache()}>
            💾 Snapshot cache to disk
          </button>
          <span className="hint">
            Forces the in-memory cache to persist now rather than on the next interval — this is what
            survives a gateway restart.
          </span>
        </div>
      </Card>

      <Card
        title="Live activity"
        sub="Gateway, relay and agent lines streamed over SSE."
        actions={
          <Link className="btn small" to="/logs">
            Full log view
          </Link>
        }
      >
        {logs.length === 0 ? (
          <p className="hint">
            No lines yet. The tail opens automatically once an API key is set in{' '}
            <Link to="/keys">API keys</Link>.
          </p>
        ) : (
          <div className="log-viewer" style={{ height: 200 }}>
            {logs.slice(-40).map((line, index) => (
              <div key={`${line.ts}-${index}`} className={`log-line ${line.level}`}>
                <span className="ts">{line.ts.slice(11, 19)}</span>
                <span className="faint">{line.level.slice(0, 4)}</span>
                <span className="msg">{line.message}</span>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card title="Raw status" sub="Exactly what /v1/status returned — handy when reporting a bug.">
        <details>
          <summary className="muted small" style={{ cursor: 'pointer' }}>
            expand JSON
          </summary>
          <div style={{ marginTop: 10 }}>
            <CodeBlock code={prettyJson(status)} />
          </div>
        </details>
      </Card>
    </>
  )
}
