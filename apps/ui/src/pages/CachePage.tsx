import { useCallback, useEffect, useState } from 'react'
import { api, cachedGet } from '../api/client'
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
  prettyJson,
} from '../components/ui'
import { useStore } from '../state/store'

interface UpstashKey {
  key: string
  collection: string
  id: string
  kind: string
  size_bytes: number
  preview: string
  value: unknown
}

interface UpstashView {
  connected: boolean
  enabled?: boolean
  backend: string
  namespace?: string
  entries: number
  shown?: number
  size_bytes: number
  pending: number
  collections: Record<string, number>
  keys: UpstashKey[]
  /** Why the mirror is not live: 'no_url' | 'bad_url' | 'library_missing' | … */
  reason?: string
  /** 'connected' | 'retry' | 'backoff' | 'unreachable' | 'invalid_url' | 'disabled' */
  state?: string
  configured?: boolean
  host?: string
  /** The exact client error (ConnectionError, TimeoutError, NOAUTH…), if any. */
  last_error?: string
  retry_in_s?: number
  restored?: number
  message?: string
  hint?: string
  error?: string
}

/**
 * Upstash inspector.
 *
 * The Cache page shows only the Redis registry mirror (models, endpoints,
 * runtimes, API keys, …). Inference responses stay in the local gateway
 * buffer and are not listed here.
 */
export default function CachePage() {
  const { toast } = useStore()
  const [view, setView] = useState<UpstashView | null>(null)
  const [detail, setDetail] = useState<{ key: string; value: unknown; size_bytes: number } | null>(null)
  const [limit, setLimit] = useState(200)
  const [collection, setCollection] = useState<string>('')
  const [stale, setStale] = useState(false)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const result = await cachedGet<UpstashView>('/api/upstash/keys', {
        limit,
        ...(collection ? { collection } : {}),
      })
      setView(result.data)
      setStale(result.stale)
      setError(null)
    } catch (err) {
      setStale(true)
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [limit, collection])

  useEffect(() => {
    void load()
  }, [load])

  const inspect = async (key: string) => {
    try {
      const result = await api.get<{ key: string; value: unknown; size_bytes: number }>('/api/upstash/key', {
        key,
      })
      setDetail(result)
    } catch (err) {
      toast('error', 'Cannot read Upstash key', (err as Error).message)
    }
  }

  const drop = async (key: string) => {
    try {
      await api.delete('/api/upstash/key', { key })
      toast('info', 'Dropped Redis key')
      if (detail?.key === key) setDetail(null)
      await load()
    } catch (err) {
      toast('error', 'Delete failed', (err as Error).message)
    }
  }

  const syncNow = async () => {
    setBusy(true)
    try {
      const result = await api.post<{
        synced: boolean
        reason?: string
        message?: string
        hint?: string
        pushed?: number
      }>('/api/upstash/sync')
      if (result.synced) {
        toast('success', 'Flushed local registry to Upstash', result.pushed ? `${result.pushed} key(s) pushed` : '')
      } else {
        // "not synced" is never a silent no-op: name the hop that failed.
        toast('error', result.message || 'Upstash did not accept the write', result.hint || result.reason || '')
      }
      await load()
    } catch (err) {
      toast('error', 'Sync failed', (err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  /** Retry the connection now, skipping the 30 s reconnect backoff. */
  const reconnect = async () => {
    setBusy(true)
    try {
      const status = await api.post<UpstashView>('/api/upstash/reconnect')
      if (status.connected) toast('success', 'Connected to Upstash', status.host ?? '')
      else toast('info', status.message || 'Upstash still unreachable', status.hint ?? '')
      await load()
    } catch (err) {
      toast('error', 'Reconnect failed', (err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  if (loading && !view) {
    return (
      <Card>
        <Spinner label="loading Upstash…" />
      </Card>
    )
  }

  const collections = view?.collections ?? {}
  const collectionNames = Object.keys(collections).sort()
  const connected = Boolean(view?.connected)

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">Upstash cache</h1>
          <p className="page-sub">
            Live keys in the Redis registry mirror. This is models, endpoints, runtimes and API keys —
            not prediction responses.
          </p>
        </div>
        <div className="row">
          <button type="button" onClick={() => void load()}>
            ↻ Refresh
          </button>
          <button type="button" disabled={busy} onClick={() => void reconnect()}>
            ⚡ Reconnect
          </button>
          <button type="button" disabled={busy || !connected} onClick={() => void syncNow()}>
            {busy ? <Spinner label="syncing…" /> : '☁ Sync to Upstash'}
          </button>
        </div>
      </div>

      {error && <Banner tone="red">{error}</Banner>}
      {stale && !error && (
        <Banner>Showing the last-known-good Upstash view — the control plane did not answer.</Banner>
      )}
      {view && !view.connected && (
        <Banner tone="amber">
          <div className="stack">
            <strong>{view.message || 'Upstash is not connected.'}</strong>
            <span className="small">
              {view.enabled
                ? 'The gateway was told to mirror the registry'
                : 'The gateway is not running with CDP_REGISTRY_BACKEND=redis'}
              {view.configured ? ', and it did read a CDP_REDIS_URL' : ', and this process has no CDP_REDIS_URL'}.
              {view.retry_in_s ? ` Retrying in ${Math.ceil(view.retry_in_s)} s.` : ''}
            </span>
            {view.hint ? <span className="small">{view.hint}</span> : null}
            {view.last_error ? (
              <code className="small mono" style={{ overflowWrap: 'anywhere' }}>
                {view.last_error}
              </code>
            ) : null}
            <span className="small">
              Env changes only take effect on the next deploy: the gateway reads CDP_* once at process start.
            </span>
          </div>
        </Banner>
      )}
      {view?.error && <Banner tone="red">{view.error}</Banner>}

      <div className="grid cols-4">
        <Card>
          <Stat
            label="Status"
            value={connected ? 'connected' : (view?.state ?? 'offline')}
            tone={connected ? 'green' : view?.enabled && view?.configured ? 'red' : 'amber'}
            hint={
              connected
                ? view?.host || view?.namespace
                : view?.reason || view?.namespace || 'no namespace'
            }
          />
        </Card>
        <Card>
          <Stat label="Keys" value={view?.entries ?? 0} hint={`${view?.pending ?? 0} pending local writes`} />
        </Card>
        <Card>
          <Stat
            label="Shown size"
            value={formatBytes(view?.size_bytes)}
            hint={view?.shown != null ? `${view.shown} listed` : undefined}
          />
          {view && view.entries > 0 ? <Meter value={view.shown ?? view.keys.length} max={view.entries} /> : null}
        </Card>
        <Card>
          <Stat
            label="Collections"
            value={collectionNames.length}
            hint={collectionNames.join(', ') || '—'}
          />
        </Card>
      </div>

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <Card
          title={`Upstash keys (${view?.keys.length ?? 0})`}
          sub="One Redis key per registry record. Inference cache is not mirrored here."
          actions={
            <div className="row" style={{ gap: 8 }}>
              <select
                value={collection}
                onChange={(event) => setCollection(event.target.value)}
                style={{ width: 'auto' }}
              >
                <option value="">all collections</option>
                {collectionNames.map((name) => (
                  <option key={name} value={name}>
                    {name} ({collections[name]})
                  </option>
                ))}
              </select>
              <select value={limit} onChange={(event) => setLimit(Number(event.target.value))} style={{ width: 'auto' }}>
                {[50, 100, 200, 500].map((n) => (
                  <option key={n} value={n}>
                    {n} rows
                  </option>
                ))}
              </select>
            </div>
          }
        >
          {!connected || (view?.keys.length ?? 0) === 0 ? (
            <Empty
              title={connected ? 'Upstash is empty' : 'No Upstash connection'}
              body={
                connected
                  ? 'Deploy a model or publish an endpoint — each registry write is mirrored here after the next sync window.'
                  : view?.enabled && view?.configured
                    ? 'The gateway is configured for Upstash and has the registry on disk, but the connection is not up yet — the banner above names the exact error, and Reconnect retries without waiting for the backoff.'
                    : 'The gateway is running on local JSON only. Add an Upstash Redis URL to persist the registry across host recycles.'
              }
            />
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Collection</th>
                  <th>Id</th>
                  <th>Kind</th>
                  <th>Size</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {view!.keys.map((entry) => (
                  <tr key={entry.key} style={{ cursor: 'pointer' }} onClick={() => void inspect(entry.key)}>
                    <td className="mono small">{entry.collection}</td>
                    <td className="mono small truncate" style={{ maxWidth: 220 }} title={entry.key}>
                      {entry.id}
                    </td>
                    <td>
                      <Badge tone={entry.kind === 'record' ? 'green' : entry.kind === 'meta' ? 'accent' : 'gray'}>
                        {entry.kind}
                      </Badge>
                    </td>
                    <td className="num small">{formatBytes(entry.size_bytes)}</td>
                    <td>
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
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>

        <div>
          <Card title="Key detail" sub={detail ? detail.key : 'Select a Redis key to inspect the JSON blob.'}>
            {detail ? (
              <>
                <KeyValue
                  items={[
                    ['key', <span className="mono small">{detail.key}</span>],
                    ['size', formatBytes(detail.size_bytes)],
                  ]}
                />
                <div style={{ marginTop: 12 }}>
                  <CodeBlock label="stored value" code={prettyJson(detail.value)} />
                </div>
              </>
            ) : (
              <p className="hint">
                Each key is one registry document (or collection meta / keepalive). Dropping a key
                removes it from Redis only — the local JSON file is unchanged until the next restore.
              </p>
            )}
          </Card>

          {view ? (
            <Card title="Collections">
              {collectionNames.length === 0 ? (
                <p className="hint">Nothing mirrored yet.</p>
              ) : (
                <table>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Keys</th>
                    </tr>
                  </thead>
                  <tbody>
                    {collectionNames.map((name) => (
                      <tr key={name}>
                        <td className="mono small">{name}</td>
                        <td className="num small">{collections[name]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Card>
          ) : null}
        </div>
      </div>
    </>
  )
}
