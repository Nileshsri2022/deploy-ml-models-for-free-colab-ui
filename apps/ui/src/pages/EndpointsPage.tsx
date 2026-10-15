import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { ApiError, api, cachedGet } from '../api/client'
import type { DegradationMode, EndpointInfo, JobInfo, ModelInfo } from '../api/types'
import { DeployWizard } from '../components/DeployWizard'
import {
  Badge,
  Banner,
  Card,
  CodeBlock,
  DeploymentBadge,
  Empty,
  KeyValue,
  RuntimeBadge,
  Spinner,
  formatSeconds,
  formatTime,
  prettyJson,
} from '../components/ui'
import { useStore } from '../state/store'

/**
 * Endpoints: the public API surface plus the knobs that make it survivable.
 *
 * Every endpoint exposes the same routes regardless of the framework behind it,
 * and carries its own cache TTL, stale window, degradation policy and rate limit.
 */
export default function EndpointsPage() {
  const { endpointId } = useParams()
  const { toast, refresh } = useStore()
  const [endpoints, setEndpoints] = useState<EndpointInfo[]>([])
  const [models, setModels] = useState<ModelInfo[]>([])
  const [jobs, setJobs] = useState<JobInfo[]>([])
  const [stale, setStale] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(endpointId ?? null)
  const [deploying, setDeploying] = useState(false)

  const load = useCallback(async () => {
    try {
      const [ep, md] = await Promise.all([
        cachedGet<{ endpoints: EndpointInfo[] }>('/api/endpoints'),
        cachedGet<{ models: ModelInfo[] }>('/api/models'),
      ])
      setEndpoints(ep.data.endpoints ?? [])
      setModels(md.data.models ?? [])
      setStale(ep.stale || md.stale)
      setError(null)
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
    try {
      const queued = await api.get<{ jobs: JobInfo[] }>('/api/jobs', { state: 'queued' })
      setJobs(queued.jobs ?? [])
    } catch {
      /* jobs are a detail; ignore failures */
    }
  }, [])

  useEffect(() => {
    void load()
    const timer = setInterval(() => void load(), 8000)
    return () => clearInterval(timer)
  }, [load])

  useEffect(() => {
    if (endpointId) setSelected(endpointId)
  }, [endpointId])

  const detail = endpoints.find((e) => e.id === selected) ?? null

  const invalidate = async (id: string | undefined, scope: 'all' | 'endpoint') => {
    try {
      const result = await api.post<{ invalidated: number; scope: string }>(
        '/api/cache/invalidate',
        undefined,
        scope === 'endpoint' && id ? { endpoint_id: id } : undefined,
      )
      toast('success', 'Cache invalidated', `${result.invalidated} entries (${result.scope})`)
      await load()
    } catch (err) {
      toast('error', 'Invalidate failed', (err as Error).message)
    }
  }

  const remove = async (id: string) => {
    if (!window.confirm(`Delete endpoint ${id}? Deployments and cached entries for it are dropped.`)) return
    try {
      await api.delete(`/api/endpoints/${id}`)
      toast('info', 'Endpoint deleted')
      if (selected === id) setSelected(null)
      await load()
      await refresh()
    } catch (err) {
      toast('error', 'Delete failed', (err as Error).message)
    }
  }

  const rollback = async (id: string, version: number) => {
    if (!window.confirm(`Roll back to model version ${version}?`)) return
    try {
      const result = await api.post<{ rolled_back_to: number; pushed_to_runtime: boolean }>(
        `/api/endpoints/${id}/rollback`,
        undefined,
        { model_version: version },
      )
      toast(
        'success',
        `Rolled back to v${result.rolled_back_to}`,
        result.pushed_to_runtime ? 'pushed to the runtime' : 'applied on next reconnect',
      )
      await load()
      await refresh()
    } catch (err) {
      toast('error', 'Rollback failed', (err as Error).message)
    }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">Endpoints</h1>
          <p className="page-sub">
            One shape for every model: <code className="mono">POST /v1/models/{'{route}'}/predict</code>,
            plus <code className="mono">:batch</code>, <code className="mono">/schema</code> and the
            status/metrics views. Route names are stable across redeployments — that is what makes them
            safe to hand to another service.
          </p>
        </div>
        <div className="row">
          <button type="button" onClick={() => void load()}>
            ↻ Refresh
          </button>
          <button type="button" className="primary" onClick={() => setDeploying((v) => !v)}>
            {deploying ? 'Close deploy wizard' : '+ Deploy a model'}
          </button>
        </div>
      </div>

      {error && <Banner tone="red">{error}</Banner>}
      {stale && !error && <Banner>Showing last-known-good endpoints — the control plane did not answer.</Banner>}
      {jobs.length > 0 && (
        <Banner tone="blue">
          {jobs.length} request{jobs.length === 1 ? '' : 's'} queued for a runtime that is offline. They run
          automatically when it reconnects, then fail after the retry budget is spent.
        </Banner>
      )}

      {deploying ? (
        <DeployWizard models={models} onDeployed={() => void load()} />
      ) : null}

      {loading && endpoints.length === 0 ? (
        <Card>
          <Spinner label="loading endpoints…" />
        </Card>
      ) : endpoints.length === 0 ? (
        <Card>
          <Empty
            title="No endpoints yet"
            body="Deploy a model to create one. The wizard records the intent even if no VM is attached, so you can wire this up before Colab is running."
            action={
              <button type="button" className="primary" onClick={() => setDeploying(true)}>
                Deploy a model
              </button>
            }
          />
        </Card>
      ) : (
        <div className="grid cols-2">
          <Card title={`Endpoints (${endpoints.length})`} sub="Click a row for its contract, curl and policy knobs.">
            <table>
              <thead>
                <tr>
                  <th>Route</th>
                  <th>Model</th>
                  <th>Serving</th>
                  <th>Policy</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {endpoints.map((endpoint) => (
                  <tr
                    key={endpoint.id}
                    style={{ cursor: 'pointer', background: endpoint.id === selected ? 'var(--bg-inset)' : undefined }}
                    onClick={() => setSelected(endpoint.id)}
                  >
                    <td>
                      <span className="mono small">/{endpoint.route}</span>
                      <div className="faint small">{endpoint.visibility}</div>
                    </td>
                    <td className="small">
                      {endpoint.model_name ?? endpoint.model_id}
                      <span className="faint"> v{endpoint.model_version}</span>
                    </td>
                    <td>
                      <DeploymentBadge state={endpoint.deployment_state} />
                      {endpoint.runtime_state ? (
                        <div style={{ marginTop: 3 }}>
                          <RuntimeBadge state={endpoint.runtime_state} />
                        </div>
                      ) : null}
                    </td>
                    <td className="small faint nowrap">
                      ttl {formatSeconds(endpoint.cache_ttl_s)} · {endpoint.degradation_mode}
                    </td>
                    <td>
                      <button
                        type="button"
                        className="small ghost"
                        onClick={(event) => {
                          event.stopPropagation()
                          void invalidate(endpoint.id, 'endpoint')
                        }}
                        title="Drop cached responses for this endpoint"
                      >
                        flush
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <div>
            {detail ? (
              <EndpointDetail
                endpoint={detail}
                onPatch={async (patch) => {
                  try {
                    await api.patch(`/api/endpoints/${detail.id}`, patch)
                    toast('success', 'Endpoint updated')
                    await load()
                    await refresh()
                  } catch (err) {
                    toast('error', 'Update failed', (err as Error).message)
                  }
                }}
                onRollback={(version) => void rollback(detail.id, version)}
                onDelete={() => void remove(detail.id)}
                onInvalidate={() => void invalidate(detail.id, 'endpoint')}
                versions={
                  models.find((m) => m.id === detail.model_id)?.all_versions ?? [detail.model_version]
                }
              />
            ) : (
              <Card>
                <Empty title="Select an endpoint" body="Its route, schemas, curl snippet and resilience policy appear here." />
              </Card>
            )}
          </div>
        </div>
      )}
    </>
  )
}

function EndpointDetail({
  endpoint,
  onPatch,
  onRollback,
  onDelete,
  onInvalidate,
  versions,
}: {
  endpoint: EndpointInfo
  onPatch: (patch: Record<string, unknown>) => Promise<void>
  onRollback: (version: number) => void
  onDelete: () => void
  onInvalidate: () => void
  versions: number[]
}) {
  const [ttl, setTtl] = useState(endpoint.cache_ttl_s)
  const [swr, setSwr] = useState(endpoint.stale_while_revalidate_s)
  const [timeoutMs, setTimeoutMs] = useState(endpoint.timeout_ms)
  const [rpm, setRpm] = useState(endpoint.rate_limit_rpm)
  const [degradation, setDegradation] = useState<DegradationMode>(endpoint.degradation_mode)
  const [visibility, setVisibility] = useState<'public' | 'private'>(endpoint.visibility)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    setTtl(endpoint.cache_ttl_s)
    setSwr(endpoint.stale_while_revalidate_s)
    setTimeoutMs(endpoint.timeout_ms)
    setRpm(endpoint.rate_limit_rpm)
    setDegradation(endpoint.degradation_mode)
    setVisibility(endpoint.visibility)
  }, [endpoint])

  const save = async () => {
    setBusy(true)
    try {
      await onPatch({
        cache_ttl_s: ttl,
        stale_while_revalidate_s: swr,
        timeout_ms: timeoutMs,
        rate_limit_rpm: rpm,
        degradation_mode: degradation,
        visibility,
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <Card
        title={<span className="mono">/{endpoint.route}</span>}
        sub={`${endpoint.model_name ?? endpoint.model_id} v${endpoint.model_version} · ${endpoint.adapter ?? 'adapter unknown'}`}
        actions={
          <div className="row">
            <Badge tone={endpoint.status === 'ready' ? 'green' : 'gray'}>{endpoint.status}</Badge>
            <DeploymentBadge state={endpoint.deployment_state} />
          </div>
        }
      >
        <KeyValue
          items={[
            ['id', <span className="mono">{endpoint.id}</span>],
            ['predict url', <span className="mono">{endpoint.url ?? endpoint.path}</span>],
            ['runtime', endpoint.serving_runtime_id ? <span className="mono">{endpoint.serving_runtime_id}</span> : null],
            ['live', endpoint.live ? <Badge tone="green" dot>serving now</Badge> : <Badge tone="amber" dot>not serving</Badge>],
            ['created', formatTime(endpoint.created_at)],
          ]}
        />
        <div className="row" style={{ marginTop: 12 }}>
          <button type="button" className="small" onClick={onInvalidate}>
            Flush cache
          </button>
          <select
            value={endpoint.model_version}
            onChange={(event) => onRollback(Number(event.target.value))}
            style={{ width: 'auto' }}
            title="Roll back to another registered version"
          >
            {versions.map((version) => (
              <option key={version} value={version}>
                roll back to v{version}
              </option>
            ))}
          </select>
          <button type="button" className="small danger" onClick={onDelete}>
            Delete endpoint
          </button>
        </div>
      </Card>

      <Card title="Call it" sub="The gateway adds cache, retry and degradation behaviour around your model.">
        <CodeBlock label="curl" code={endpoint.curl ?? `curl -X POST '${endpoint.url ?? endpoint.path}'`} />
        <div style={{ marginTop: 12 }}>
          <CodeBlock
            label="request schema"
            code={prettyJson(endpoint.schema_in ?? {})}
          />
        </div>
        <div style={{ marginTop: 10 }}>
          <CodeBlock label="response schema" code={prettyJson(endpoint.schema_out ?? {})} />
        </div>
        <p className="hint" style={{ marginTop: 10 }}>
          Every response carries <code className="mono">X-ColabDeploy-Cache</code> (HIT / MISS / STALE /
          BYPASS), <code className="mono">X-ColabDeploy-Latency-Ms</code> and{' '}
          <code className="mono">X-ColabDeploy-Request-Id</code> — the same envelope the UI shows.
        </p>
      </Card>

      <Card title="Policy" sub="Per-endpoint resilience. Changes apply immediately and survive a VM restart.">
        <div className="grid cols-4">
          <label className="field">
            <span>Cache TTL (s)</span>
            <input type="number" min={0} value={ttl} onChange={(event) => setTtl(Number(event.target.value) || 0)} />
          </label>
          <label className="field">
            <span>Serve-stale (s)</span>
            <input type="number" min={0} value={swr} onChange={(event) => setSwr(Number(event.target.value) || 0)} />
          </label>
          <label className="field">
            <span>Timeout (ms)</span>
            <input type="number" min={500} step={500} value={timeoutMs} onChange={(event) => setTimeoutMs(Number(event.target.value) || 30000)} />
          </label>
          <label className="field">
            <span>Rate limit (rpm)</span>
            <input type="number" min={1} value={rpm} onChange={(event) => setRpm(Number(event.target.value) || 600)} />
          </label>
        </div>
        <div className="grid cols-2">
          <label className="field">
            <span>Degradation when the runtime is gone</span>
            <select value={degradation} onChange={(event) => setDegradation(event.target.value as DegradationMode)}>
              <option value="serve_cached">serve_cached — stale answer, flagged</option>
              <option value="queue">queue — 202 + job id</option>
              <option value="fail">fail — 503 + Retry-After</option>
            </select>
          </label>
          <label className="field">
            <span>Visibility</span>
            <select value={visibility} onChange={(event) => setVisibility(event.target.value as 'public' | 'private')}>
              <option value="public">public</option>
              <option value="private">private</option>
            </select>
          </label>
        </div>
        <div className="row">
          <button type="button" className="primary" disabled={busy} onClick={() => void save()}>
            {busy ? <Spinner label="saving…" /> : 'Save policy'}
          </button>
          <span className="hint">
            TTL 0 disables caching for this endpoint; the stale window only applies while{' '}
            <code className="mono">serve_cached</code> is selected.
          </span>
        </div>
      </Card>
    </>
  )
}
