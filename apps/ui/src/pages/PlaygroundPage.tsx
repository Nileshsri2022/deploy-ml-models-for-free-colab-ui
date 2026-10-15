import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, cachedGet } from '../api/client'
import type { EndpointInfo, InferenceResponse } from '../api/types'
import {
  Badge,
  Banner,
  CacheBadge,
  Card,
  CodeBlock,
  Empty,
  KeyValue,
  Spinner,
  prettyJson,
} from '../components/ui'
import { useStore } from '../state/store'

interface Call {
  id: number
  at: number
  route: string
  ms: number
  cache: string
  stale: boolean
  ok: boolean
  detail: string
}

/**
 * Playground: hit a live endpoint from the browser and watch the cache behave.
 *
 * The interesting part is the history strip — send the same payload twice and the
 * second call comes back as HIT with sub-millisecond latency, which is the
 * behaviour that keeps serving while the Colab VM is gone.
 */
export default function PlaygroundPage() {
  const { toast, hasKey } = useStore()
  const [endpoints, setEndpoints] = useState<EndpointInfo[]>([])
  const [route, setRoute] = useState('')
  const [payload, setPayload] = useState('{\n  "inputs": {}\n}')
  const [response, setResponse] = useState<InferenceResponse | null>(null)
  const [raw, setRaw] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [batch, setBatch] = useState(false)
  const [history, setHistory] = useState<Call[]>([])
  const [elapsed, setElapsed] = useState<number | null>(null)

  const load = useCallback(async () => {
    try {
      const result = await cachedGet<{ endpoints: EndpointInfo[] }>('/api/endpoints')
      setEndpoints(result.data.endpoints ?? [])
    } catch {
      /* the dashboard banner already covers an unreachable control plane */
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const current = useMemo(() => endpoints.find((e) => e.route === route) ?? null, [endpoints, route])

  useEffect(() => {
    if (!route && endpoints.length) setRoute(endpoints[0].route)
  }, [endpoints, route])

  useEffect(() => {
    if (!current) return
    const example = current.schema_in?.example
    setPayload(JSON.stringify(example ? { inputs: example } : { inputs: {} }, null, 2))
  }, [current?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  const send = async () => {
    let body: unknown
    try {
      body = JSON.parse(payload)
    } catch (err) {
      setError(`payload is not valid JSON: ${(err as Error).message}`)
      return
    }
    setBusy(true)
    setError(null)
    setElapsed(null)
    const started = performance.now()
    const path = batch ? `/v1/models/${route}/predict:batch` : `/v1/models/${route}/predict`
    try {
      const result = await api.post<InferenceResponse>(path, body)
      const ms = performance.now() - started
      setElapsed(ms)
      setResponse(result)
      setRaw(prettyJson(result))
      if (result.error) setError(`${result.error.code}: ${result.error.message}`)
      setHistory((previous) =>
        [
          {
            id: started,
            at: Date.now(),
            route,
            ms: result.latency_ms ?? ms,
            cache: result.cache ?? 'BYPASS',
            stale: Boolean(result.stale),
            ok: !result.error,
            detail: result.error ? result.error.code : result.cache ?? '',
          },
          ...previous,
        ].slice(0, 20),
      )
      toast(
        result.error ? 'error' : result.cache === 'HIT' ? 'success' : 'info',
        result.error ? `${result.error.code}` : `${result.cache} in ${Math.round(result.latency_ms ?? ms)} ms`,
        result.stale ? 'served from cache while the runtime was unreachable' : undefined,
      )
    } catch (err) {
      setElapsed(performance.now() - started)
      setResponse(null)
      setRaw(null)
      setError((err as Error).message)
      toast('error', 'Request failed', (err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">Playground</h1>
          <p className="page-sub">
            Send a real request through the gateway — cache lookup, runtime relay, degradation policy
            and all. Nothing here is mocked; this is the same path your curl uses.
          </p>
        </div>
      </div>

      {!hasKey && (
        <Banner>
          No API key set. Public endpoints may answer anonymously, but set a key in{' '}
          <strong>API keys</strong> to be sure.
        </Banner>
      )}

      {endpoints.length === 0 ? (
        <Card>
          <Empty
            title="No endpoints to call"
            body="Deploy a model first — the playground talks to real endpoints only."
          />
        </Card>
      ) : (
        <>
          <Card title="Request">
            <div className="grid cols-2">
              <label className="field">
                <span>Endpoint</span>
                <select value={route} onChange={(event) => setRoute(event.target.value)}>
                  {endpoints.map((endpoint) => (
                    <option key={endpoint.id} value={endpoint.route}>
                      /{endpoint.route} — {endpoint.model_name ?? endpoint.model_id} v{endpoint.model_version}
                      {endpoint.live ? ' · live' : ' · no runtime'}
                    </option>
                  ))}
                </select>
                <div className="hint">
                  {current
                    ? `${current.degradation_mode} · ttl ${current.cache_ttl_s}s · stale window ${current.stale_while_revalidate_s}s · timeout ${current.timeout_ms}ms`
                    : 'select an endpoint'}
                </div>
              </label>
              <label className="field">
                <span>Mode</span>
                <select value={batch ? 'batch' : 'single'} onChange={(event) => setBatch(event.target.value === 'batch')}>
                  <option value="single">single — POST /v1/models/{'{route}'}/predict</option>
                  <option value="batch">batch — POST /v1/models/{'{route}'}/predict:batch</option>
                </select>
                <div className="hint">
                  {batch
                    ? 'Body must be {"inputs_list": [{…}, {…}]} — one round trip to the VM, one row per result.'
                    : 'Body is passed straight to the model as the payload.'}
                </div>
              </label>
            </div>

            <label className="field">
              <span>Payload (JSON)</span>
              <textarea rows={10} value={payload} onChange={(event) => setPayload(event.target.value)} />
            </label>

            {error ? <Banner tone="red">{error}</Banner> : null}

            <div className="row">
              <button type="button" className="primary" disabled={busy} onClick={() => void send()}>
                {busy ? <Spinner label="calling…" /> : 'Send request'}
              </button>
              <button
                type="button"
                className="ghost"
                onClick={() => {
                  const example = current?.schema_in?.example
                  setPayload(JSON.stringify(batch ? { inputs_list: [example ?? {}] } : { inputs: example ?? {} }, null, 2))
                }}
              >
                Reset to schema example
              </button>
              {elapsed !== null ? <span className="hint">round trip {Math.round(elapsed)} ms (browser clock)</span> : null}
            </div>
          </Card>

          <div className="grid cols-2" style={{ marginTop: 14 }}>
            <Card
              title="Response"
              actions={response ? <CacheBadge status={response.cache ?? 'BYPASS'} /> : null}
              sub={response ? undefined : 'Send a request to see the envelope.'}
            >
              {busy ? (
                <Spinner label="waiting for the runtime…" />
              ) : response ? (
                <>
                  <KeyValue
                    items={[
                      ['cache', <CacheBadge status={response.cache ?? 'BYPASS'} />],
                      ['latency', `${Math.round(response.latency_ms ?? 0)} ms`],
                      ['stale', response.stale ? <Badge tone="amber">yes — served from cache</Badge> : 'no'],
                      ['runtime', response.runtime_id ? <span className="mono">{response.runtime_id}</span> : 'none (cached/queued)'],
                      ['version', `v${response.model_version}`],
                      ['request id', <span className="mono">{response.request_id}</span>],
                      ['job id', response.job_id ? <span className="mono">{response.job_id}</span> : null],
                    ]}
                  />
                  <div style={{ marginTop: 12 }}>
                    <CodeBlock label="body" code={raw ?? prettyJson(response)} />
                  </div>
                </>
              ) : (
                <Empty title="No response yet" body="The response envelope always includes the cache verdict, so you can see whether the VM was involved." />
              )}
            </Card>

            <Card title="Recent calls" sub="Newest first. Repeat a payload to watch it flip to HIT.">
              {history.length === 0 ? (
                <p className="hint">No calls yet.</p>
              ) : (
                <table>
                  <thead>
                    <tr>
                      <th>Route</th>
                      <th>Cache</th>
                      <th>Latency</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((call) => (
                      <tr key={call.id}>
                        <td className="mono small">/{call.route}</td>
                        <td>
                          <CacheBadge status={call.cache} />
                        </td>
                        <td className="num small">{Math.round(call.ms)} ms</td>
                        <td className="small">
                          {call.ok ? (
                            <span className="faint">{call.stale ? 'stale-while-revalidate' : 'ok'}</span>
                          ) : (
                            <span style={{ color: 'var(--red)' }}>{call.detail}</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Card>
          </div>
        </>
      )}
    </>
  )
}
