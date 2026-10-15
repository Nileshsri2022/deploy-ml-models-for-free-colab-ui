import { useCallback, useEffect, useState } from 'react'
import { api, cachedGet } from '../api/client'
import type { CacheEntryInfo, CacheStats } from '../api/types'
import {
  Badge,
  Banner,
  Card,
  CodeBlock,
  Empty,
  KeyValue,
  Meter,
  Spinner,
  Stat,
  formatBytes,
  formatTime,
  prettyJson,
} from '../components/ui'
import { useStore } from '../state/store'

interface HotKey {
  key: string
  endpoint_id: string
  model_version: number
  hits: number
  sample: unknown
}

interface EntryDetail {
  key: string
  status: string
  value: unknown
  age_s: number
  endpoint_id: string | null
  model_version: number | null
}

/**
 * Cache inspector.
 *
 * The cache is the reason a Colab VM can disappear without the API going dark:
 * fresh entries answer directly, stale ones answer while the gateway re-fetches,
 * and the hot keys are what the agent re-warms on reconnect.
 */
export default function CachePage() {
  const { toast, refresh } = useStore()
  const [stats, setStats] = useState<CacheStats | null>(null)
  const [entries, setEntries] = useState<CacheEntryInfo[]>([])
  const [hot, setHot] = useState<HotKey[]>([])
  const [detail, setDetail] = useState<EntryDetail | null>(null)
  const [limit, setLimit] = useState(50)
  const [stale, setStale] = useState(false)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const [statsResult, entriesResult, hotResult] = await Promise.all([
        cachedGet<CacheStats>('/api/cache/stats'),
        cachedGet<{ entries: CacheEntryInfo[] }>('/api/cache/entries', { limit }),
        api.get<{ hot_keys: HotKey[] }>('/api/cache/hot', { limit: 20 }),
      ])
      setStats(statsResult.data)
      setEntries(entriesResult.data.entries ?? [])
      setHot(hotResult.hot_keys ?? [])
      setStale(statsResult.stale || entriesResult.stale)
      setError(null)
    } catch (err) {
      setStale(true)
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [limit])

  useEffect(() => {
    void load()
  }, [load])

  const inspect = async (key: string) => {
    try {
      const result = await api.get<EntryDetail>(`/api/cache/entry/${encodeURIComponent(key)}`)
      setDetail(result)
    } catch (err) {
      toast('error', 'Cannot read entry', (err as Error).message)
    }
  }

  const pin = async (key: string, pinned: boolean) => {
    try {
      await api.post(`/api/cache/entry/${encodeURIComponent(key)}/pin`, undefined, { pinned })
      toast('success', pinned ? 'Entry pinned' : 'Entry unpinned', 'pinned entries are never evicted')
      await load()
    } catch (err) {
      toast('error', 'Pin failed', (err as Error).message)
    }
  }

  const drop = async (key: string) => {
    try {
      await api.delete(`/api/cache/entry/${encodeURIComponent(key)}`)
      toast('info', 'Entry deleted')
      if (detail?.key === key) setDetail(null)
      await load()
    } catch (err) {
      toast('error', 'Delete failed', (err as Error).message)
    }
  }

  const save = async () => {
    setBusy(true)
    try {
      const result = await api.post<{ saved: boolean; backend: string; stats?: CacheStats }>(
        '/api/cache/save',
      )
      if (result.saved) {
        toast('success', 'Snapshot written to disk', 'this survives a gateway restart')
        if (result.stats) setStats(result.stats)
      } else {
        toast('info', 'Backend cannot snapshot', `backend: ${result.backend}`)
      }
      await refresh()
    } catch (err) {
      toast('error', 'Snapshot failed', (err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const invalidateAll = async () => {
    if (!window.confirm('Drop every cached response? Offline endpoints will stop answering.')) return
    try {
      const result = await api.post<{ invalidated: number }>('/api/cache/invalidate')
      toast('info', `Invalidated ${result.invalidated} entries`)
      await load()
      await refresh()
    } catch (err) {
      toast('error', 'Invalidate failed', (err as Error).message)
    }
  }

  if (loading && !stats) {
    return (
      <Card>
        <Spinner label="loading cache…" />
      </Card>
    )
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">Cache</h1>
          <p className="page-sub">
            Two-tier state retention: the gateway cache answers while a VM is gone, and the hot keys are
            replayed to the agent as warm-up samples the instant it reconnects.
          </p>
        </div>
        <div className="row">
          <button type="button" onClick={() => void load()}>
            ↻ Refresh
          </button>
          <button type="button" disabled={busy} onClick={() => void save()}>
            {busy ? <Spinner label="saving…" /> : '💾 Snapshot now'}
          </button>
          <button type="button" className="danger" onClick={() => void invalidateAll()}>
            Invalidate all
          </button>
        </div>
      </div>

      {error && <Banner tone="red">{error}</Banner>}
      {stale && !error && <Banner>Showing the last-known-good cache view — the control plane did not answer.</Banner>}

      <div className="grid cols-4">
        <Card>
          <Stat
            label="Hit ratio"
            value={stats ? `${(stats.hit_ratio * 100).toFixed(1)}%` : '—'}
            tone={stats && stats.hit_ratio >= 0.5 ? 'green' : undefined}
            hint={stats ? `${stats.hits} hits · ${stats.misses} misses · ${stats.stale_hits} stale` : undefined}
          />
        </Card>
        <Card>
          <Stat label="Entries" value={stats?.entries ?? '—'} hint={`${stats?.pinned ?? 0} pinned`} />
        </Card>
        <Card>
          <Stat
            label="Memory used"
            value={formatBytes(stats?.size_bytes)}
            hint={stats ? `max ${stats.max_entries} entries` : undefined}
          />
          {stats ? <Meter value={stats.entries} max={stats.max_entries} /> : null}
        </Card>
        <Card>
          <Stat label="Backend" value={stats?.backend ?? '—'} hint={`${stats?.evictions ?? 0} evictions`} />
        </Card>
      </div>

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <Card
          title={`Entries (${entries.length})`}
          sub="Fresh entries answer without touching the VM; stale ones answer while it is re-fetched."
          actions={
            <select value={limit} onChange={(event) => setLimit(Number(event.target.value))} style={{ width: 'auto' }}>
              {[25, 50, 100, 250].map((n) => (
                <option key={n} value={n}>
                  {n} rows
                </option>
              ))}
            </select>
          }
        >
          {entries.length === 0 ? (
            <Empty
              title="Cache is empty"
              body="Send a request through an endpoint and its response lands here — keyed by endpoint, model version and the canonical JSON of the payload."
            />
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Key</th>
                  <th>Version</th>
                  <th>Hits</th>
                  <th>Size</th>
                  <th>State</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => {
                  const now = Date.now() / 1000
                  const fresh = entry.expires_at > now
                  const servable = entry.stale_at > now
                  return (
                    <tr key={entry.key} style={{ cursor: 'pointer' }} onClick={() => void inspect(entry.key)}>
                      <td className="mono small truncate" style={{ maxWidth: 220 }} title={entry.key}>
                        {entry.key.slice(0, 22)}…
                        <div className="faint">{entry.endpoint_id}</div>
                      </td>
                      <td className="small">v{entry.model_version}</td>
                      <td className="num small">{entry.hits}</td>
                      <td className="num small">{formatBytes(entry.size_bytes)}</td>
                      <td>
                        {fresh ? (
                          <Badge tone="green" dot>fresh</Badge>
                        ) : servable ? (
                          <Badge tone="amber" dot>stale</Badge>
                        ) : (
                          <Badge tone="red" dot>expired</Badge>
                        )}
                        {entry.pinned ? (
                          <div style={{ marginTop: 3 }}>
                            <Badge tone="accent">pinned</Badge>
                          </div>
                        ) : null}
                      </td>
                      <td>
                        <div className="row" style={{ gap: 4 }}>
                          <button
                            type="button"
                            className="small ghost"
                            onClick={(event) => {
                              event.stopPropagation()
                              void pin(entry.key, !entry.pinned)
                            }}
                          >
                            {entry.pinned ? 'unpin' : 'pin'}
                          </button>
                          <button
                            type="button"
                            className="small ghost"
                            style={{ color: 'var(--red)' }}
                            onClick={(event) => {
                              event.stopPropagation()
                              void drop(entry.key)
                            }}
                          >
                            drop
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </Card>

        <div>
          <Card title="Entry detail" sub={detail ? detail.key : 'Select an entry to inspect its stored response.'}>
            {detail ? (
              <>
                <KeyValue
                  items={[
                    ['status', <Badge tone={detail.status === 'HIT' ? 'green' : detail.status === 'STALE' ? 'amber' : 'gray'}>{detail.status}</Badge>],
                    ['age', `${detail.age_s.toFixed(1)} s`],
                    ['endpoint', detail.endpoint_id ? <span className="mono">{detail.endpoint_id}</span> : null],
                    ['version', detail.model_version !== null ? `v${detail.model_version}` : null],
                  ]}
                />
                <div style={{ marginTop: 12 }}>
                  <CodeBlock label="stored value" code={prettyJson(detail.value)} />
                </div>
              </>
            ) : (
              <p className="hint">
                A stored value keeps the model output plus the original request, so the gateway can
                rebuild the full response envelope while the runtime is offline.
              </p>
            )}
          </Card>

          <Card
            title={`Warm-up keys (${hot.length})`}
            sub="Replayed to the agent on reconnect so the first real request after a disconnect is not cold."
          >
            {hot.length === 0 ? (
              <p className="hint">
                Nothing hot yet — keys rank up as they get hits. The top-K is sent with every
                desired-state push.
              </p>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Endpoint</th>
                    <th>Hits</th>
                    <th>Sample</th>
                  </tr>
                </thead>
                <tbody>
                  {hot.map((key) => (
                    <tr key={key.key}>
                      <td className="mono small">
                        {key.endpoint_id}
                        <div className="faint">v{key.model_version}</div>
                      </td>
                      <td className="num small">{key.hits}</td>
                      <td className="mono small truncate" style={{ maxWidth: 200 }}>
                        {key.sample ? JSON.stringify(key.sample).slice(0, 60) : <span className="faint">—</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>

          {stats ? (
            <Card title="Raw stats">
              <div className="faint small" style={{ marginBottom: 8 }}>
                polled {formatTime(new Date().toISOString())}
              </div>
              <CodeBlock code={prettyJson(stats)} />
            </Card>
          ) : null}
        </div>
      </div>
    </>
  )
}
