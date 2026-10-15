import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import type { DegradationMode, DeployResult, ModelInfo, RuntimeInfo } from '../api/types'
import { useStore } from '../state/store'
import { Banner, Card, CodeBlock, DeploymentBadge, Spinner } from './ui'

/**
 * The deploy wizard.
 *
 * One call — `POST /api/deploy` — pins a model version to an endpoint (creating
 * it when needed) and pushes desired state to a live VM. When no VM is online the
 * deploy still succeeds: the intent is recorded and reconciled on reconnect, which
 * is the whole point of the desired-state model.
 */
export function DeployWizard({
  models,
  initialModelId,
  onDeployed,
}: {
  models: ModelInfo[]
  initialModelId?: string
  onDeployed?: (result: DeployResult) => void
}) {
  const { status, toast, refresh } = useStore()
  const runtimes: RuntimeInfo[] = status?.runtimes ?? []
  const online = useMemo(
    () => runtimes.filter((r) => r.state === 'online' || r.state === 'degraded'),
    [runtimes],
  )

  const [modelId, setModelId] = useState(initialModelId ?? models[0]?.id ?? '')
  const [version, setVersion] = useState<number | ''>('')
  const [endpointId, setEndpointId] = useState('')
  const [route, setRoute] = useState('')
  const [runtimeId, setRuntimeId] = useState('auto')
  const [ttl, setTtl] = useState(300)
  const [swr, setSwr] = useState(3600)
  const [timeoutMs, setTimeoutMs] = useState(30000)
  const [rpm, setRpm] = useState(600)
  const [degradation, setDegradation] = useState<DegradationMode>('serve_cached')
  const [visibility, setVisibility] = useState<'public' | 'private'>('public')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<DeployResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!modelId && models.length) setModelId(models[0].id)
  }, [models, modelId])

  const model = models.find((m) => m.id === modelId) ?? null
  const versions = model?.all_versions ?? []

  const submit = async () => {
    if (!modelId) {
      setError('pick a model first')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const payload = {
        model_id: modelId,
        model_version: version === '' ? null : Number(version),
        endpoint_id: endpointId.trim() || null,
        runtime_id: runtimeId === 'auto' ? null : runtimeId,
        endpoint: endpointId.trim()
          ? null // updating an existing endpoint: keep its settings
          : {
              route: route.trim() || null,
              visibility,
              cache_ttl_s: ttl,
              stale_while_revalidate_s: swr,
              degradation_mode: degradation,
              timeout_ms: timeoutMs,
              rate_limit_rpm: rpm,
            },
      }
      const deployed = await api.post<DeployResult>('/api/deploy', payload)
      setResult(deployed)
      toast('success', 'Deployed', deployed.message)
      await refresh()
      onDeployed?.(deployed)
    } catch (err) {
      setError((err as Error).message)
      toast('error', 'Deploy failed', (err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card
      title="Deploy"
      sub="Pin a model version to an endpoint. If the target VM is online it loads and warms up immediately; if not, it reconciles the moment it reconnects."
    >
      {models.length === 0 ? (
        <Banner>Register a model first — the wizard needs something to deploy.</Banner>
      ) : (
        <>
          <div className="grid cols-2">
            <label className="field">
              <span>Model</span>
              <select value={modelId} onChange={(event) => setModelId(event.target.value)}>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} · v{m.version} · {m.adapter}
                  </option>
                ))}
              </select>
              {model ? (
                <div className="hint">
                  {model.framework} artifact · <code className="mono">{model.artifact.uri.slice(0, 48)}</code>
                </div>
              ) : null}
            </label>

            <label className="field">
              <span>Version</span>
              <select value={version} onChange={(event) => setVersion(event.target.value === '' ? '' : Number(event.target.value))}>
                <option value="">latest ({model?.version ?? '?'})</option>
                {versions.map((v) => (
                  <option key={v} value={v}>
                    v{v}
                  </option>
                ))}
              </select>
              <div className="hint">Pinning an older version is how you roll back.</div>
            </label>

            <label className="field">
              <span>Endpoint</span>
              <input
                value={endpointId}
                onChange={(event) => setEndpointId(event.target.value)}
                placeholder="leave empty to create a new one"
              />
              <div className="hint">
                Existing endpoint id → update it in place (route, cache and rate limits are kept).
              </div>
            </label>

            <label className="field">
              <span>Route</span>
              <input
                value={route}
                onChange={(event) => setRoute(event.target.value)}
                placeholder={model ? model.name.toLowerCase().replace(/[^a-z0-9]+/g, '-') : 'my-model'}
                disabled={Boolean(endpointId.trim())}
              />
              <div className="hint">
                Becomes <code className="mono">/v1/models/{route || '<route>'}/predict</code>.
              </div>
            </label>

            <label className="field">
              <span>Runtime</span>
              <select value={runtimeId} onChange={(event) => setRuntimeId(event.target.value)}>
                <option value="auto">auto — any online runtime</option>
                {runtimes.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.label || r.id} ({r.state})
                  </option>
                ))}
              </select>
              <div className="hint">
                {online.length
                  ? `${online.length} runtime(s) can serve right now.`
                  : 'Nothing online — the deploy is recorded and applied on reconnect.'}
              </div>
            </label>

            <label className="field">
              <span>Visibility</span>
              <select value={visibility} onChange={(event) => setVisibility(event.target.value as 'public' | 'private')}>
                <option value="public">public — any inference key</option>
                <option value="private">private — manage scope only</option>
              </select>
            </label>
          </div>

          <div className="grid cols-4">
            <label className="field">
              <span>Cache TTL (s)</span>
              <input type="number" min={0} value={ttl} onChange={(event) => setTtl(Number(event.target.value) || 0)} />
            </label>
            <label className="field">
              <span>Serve-stale window (s)</span>
              <input type="number" min={0} value={swr} onChange={(event) => setSwr(Number(event.target.value) || 0)} />
            </label>
            <label className="field">
              <span>Timeout (ms)</span>
              <input
                type="number"
                min={500}
                step={500}
                value={timeoutMs}
                onChange={(event) => setTimeoutMs(Number(event.target.value) || 30000)}
              />
            </label>
            <label className="field">
              <span>Rate limit (req/min)</span>
              <input type="number" min={1} value={rpm} onChange={(event) => setRpm(Number(event.target.value) || 600)} />
            </label>
          </div>

          <label className="field">
            <span>When the runtime is unreachable</span>
            <select value={degradation} onChange={(event) => setDegradation(event.target.value as DegradationMode)}>
              <option value="serve_cached">serve_cached — return the last known answer, marked stale</option>
              <option value="queue">queue — accept as a job (202) and run it when the VM is back</option>
              <option value="fail">fail — 503 with Retry-After, never serve stale data</option>
            </select>
            <div className="hint">
              This is the resilience policy that makes a free, flaky GPU usable behind a real API.
            </div>
          </label>

          {error ? <Banner tone="red">{error}</Banner> : null}

          <div className="row">
            <button type="button" className="primary" disabled={busy} onClick={() => void submit()}>
              {busy ? <Spinner label="deploying…" /> : 'Deploy model'}
            </button>
            {online.length === 0 ? (
              <span className="hint">
                no runtime online — see the{' '}
                <span className="badge amber">
                  <span className="dot" />
                  queued for reconnect
                </span>{' '}
                behaviour
              </span>
            ) : (
              <span className="hint">
                target: {online.map((r) => r.label || r.id).join(', ')}{' '}
                {runtimeId !== 'auto' ? null : '(auto)'}
              </span>
            )}
          </div>

          {result ? (
            <div style={{ marginTop: 16 }}>
              <Banner tone={result.pushed_to_runtime ? 'green' : 'amber'}>
                <strong>{result.message}.</strong>{' '}
                {result.endpoint.url ? (
                  <>
                    Serving at <code className="mono">{result.endpoint.url}</code>{' '}
                  </>
                ) : null}
                {result.runtime_id ? (
                  <>
                    on runtime <code className="mono">{result.runtime_id}</code>
                  </>
                ) : (
                  <>— no runtime claimed it yet</>
                )}
              </Banner>
              <CodeBlock label="try it" code={result.curl} />
              <div className="row" style={{ marginTop: 10 }}>
                <span className="faint small">
                  deployment {result.deployment.id} · v{result.deployment.model_version} ·{' '}
                  <DeploymentBadge state={result.deployment.state} />
                </span>
              </div>
            </div>
          ) : null}
        </>
      )}
    </Card>
  )
}
