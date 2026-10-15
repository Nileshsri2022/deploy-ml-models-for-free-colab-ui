import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, cachedGet } from '../api/client'
import type { LogLine } from '../api/types'
import {
  Badge,
  Banner,
  Card,
  Empty,
  KeyValue,
  Spinner,
  formatTime,
  prettyJson,
} from '../components/ui'
import { useStore } from '../state/store'

interface AuditEntry {
  ts?: string
  actor?: string
  action?: string
  target?: string
  detail?: unknown
  [key: string]: unknown
}

const LEVELS = ['debug', 'info', 'warning', 'error']

/**
 * Logs: the live SSE tail plus the persisted ring buffer and the audit trail.
 *
 * Agent lines arrive over the same relay as exec traffic, so a VM's own logs are
 * visible here even though the gateway never touches its stdout.
 */
export default function LogsPage() {
  const { logs: live, clearLogs, hasKey, toast } = useStore()
  const [history, setHistory] = useState<LogLine[]>([])
  const [audit, setAudit] = useState<AuditEntry[]>([])
  const [tab, setTab] = useState<'live' | 'history' | 'audit'>('live')
  const [level, setLevel] = useState<string>('all')
  const [runtimeId, setRuntimeId] = useState<string>('all')
  const [query, setQuery] = useState('')
  const [limit, setLimit] = useState(300)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [follow, setFollow] = useState(true)

  const loadHistory = useCallback(async () => {
    setLoading(true)
    try {
      const result = await cachedGet<{ logs: LogLine[] }>('/api/logs', {
        limit,
        level: level === 'all' ? undefined : level,
        runtime_id: runtimeId === 'all' ? undefined : runtimeId,
      })
      setHistory(result.data.logs ?? [])
      setError(null)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [limit, level, runtimeId])

  const loadAudit = useCallback(async () => {
    try {
      const result = await api.get<{ entries: AuditEntry[] }>('/api/audit', { limit: 100 })
      setAudit(result.entries ?? [])
    } catch {
      /* the audit trail is optional decoration */
    }
  }, [])

  useEffect(() => {
    void loadHistory()
  }, [loadHistory])

  useEffect(() => {
    if (tab === 'audit') void loadAudit()
  }, [tab, loadAudit])

  // Auto-scroll the live tail unless the user scrolled away.
  useEffect(() => {
    if (!follow || tab !== 'live') return
    const viewer = document.querySelector('.log-viewer')
    if (viewer) viewer.scrollTop = viewer.scrollHeight
  }, [live, follow, tab])

  const runtimeIds = useMemo(() => {
    const ids = new Set<string>()
    for (const line of [...live, ...history]) if (line.runtime_id) ids.add(line.runtime_id)
    return [...ids].sort()
  }, [live, history])

  const filter = (lines: LogLine[]): LogLine[] =>
    lines.filter((line) => {
      if (level !== 'all' && line.level !== level) return false
      if (runtimeId !== 'all' && line.runtime_id !== runtimeId) return false
      if (query && !`${line.message} ${line.source} ${line.runtime_id ?? ''}`.toLowerCase().includes(query.toLowerCase()))
        return false
      return true
    })

  const shown = tab === 'live' ? filter(live).slice(-limit) : filter(history)

  const download = () => {
    const text = shown.map((line) => `${line.ts}\t${line.level}\t${line.source}\t${line.message}`).join('\n')
    const blob = new Blob([text], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `colabdeploy-logs-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.txt`
    anchor.click()
    URL.revokeObjectURL(url)
    toast('success', 'Logs exported', `${shown.length} lines`)
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">Logs</h1>
          <p className="page-sub">
            Gateway, relay and agent lines in one stream. The live tail uses Server-Sent Events; the
            history tab reads the persisted ring buffer, which survives a restart.
          </p>
        </div>
        <div className="row">
          <button type="button" className={follow ? 'primary' : ''} onClick={() => setFollow((v) => !v)}>
            {follow ? '⏸ Following' : '▶ Follow'}
          </button>
          <button type="button" onClick={download}>
            ⬇ Export
          </button>
          {tab === 'live' ? (
            <button type="button" className="ghost" onClick={clearLogs}>
              Clear view
            </button>
          ) : (
            <button type="button" className="ghost" onClick={() => void loadHistory()}>
              ↻ Reload
            </button>
          )}
        </div>
      </div>

      {!hasKey && (
        <Banner>
          The live tail needs an API key (browser <code className="mono">EventSource</code> cannot send
          headers, so the key is passed as a query parameter). Set one in <strong>API keys</strong>.
        </Banner>
      )}
      {error && <Banner tone="red">{error}</Banner>}

      <div className="tabs">
        <button type="button" className={tab === 'live' ? 'active' : ''} onClick={() => setTab('live')}>
          Live ({live.length})
        </button>
        <button type="button" className={tab === 'history' ? 'active' : ''} onClick={() => setTab('history')}>
          History ({history.length})
        </button>
        <button type="button" className={tab === 'audit' ? 'active' : ''} onClick={() => setTab('audit')}>
          Audit ({audit.length})
        </button>
      </div>

      {tab !== 'audit' ? (
        <Card
          title={tab === 'live' ? 'Live tail' : 'Persisted history'}
          sub={
            tab === 'live'
              ? 'Newest at the bottom, capped at 500 lines in the browser.'
              : 'The gateway keeps a bounded ring buffer; older lines are dropped, not written to disk forever.'
          }
          actions={
            <div className="row" style={{ gap: 8 }}>
              <select value={level} onChange={(event) => setLevel(event.target.value)} style={{ width: 'auto' }}>
                <option value="all">all levels</option>
                {LEVELS.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
              <select value={runtimeId} onChange={(event) => setRuntimeId(event.target.value)} style={{ width: 'auto' }}>
                <option value="all">all runtimes</option>
                {runtimeIds.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="filter text…"
                style={{ width: 180 }}
              />
              {tab === 'history' ? (
                <select value={limit} onChange={(event) => setLimit(Number(event.target.value))} style={{ width: 'auto' }}>
                  {[100, 300, 1000, 2000].map((n) => (
                    <option key={n} value={n}>
                      {n} lines
                    </option>
                  ))}
                </select>
              ) : null}
            </div>
          }
        >
          {loading && shown.length === 0 ? (
            <Spinner label="loading logs…" />
          ) : shown.length === 0 ? (
            <Empty
              title={tab === 'live' ? 'Waiting for lines' : 'Nothing matches'}
              body={
                tab === 'live'
                  ? 'Traffic, heartbeats and agent output show up here as they happen.'
                  : 'Loosen the filters or reload with a larger limit.'
              }
            />
          ) : (
            <div className="log-viewer">
              {shown.map((line, index) => (
                <div key={`${line.ts}-${index}`} className={`log-line ${line.level}`}>
                  <span className="ts">{line.ts.slice(11, 19)}</span>
                  <span className="faint">{line.level.slice(0, 4)}</span>
                  <span className="msg">
                    <span className="faint">{line.source}</span> {line.message}
                    {line.runtime_id ? <span className="faint"> [{line.runtime_id.slice(0, 8)}]</span> : null}
                    {line.data && Object.keys(line.data).length ? (
                      <span className="faint"> {JSON.stringify(line.data).slice(0, 120)}</span>
                    ) : null}
                  </span>
                </div>
              ))}
            </div>
          )}
        </Card>
      ) : (
        <Card title="Audit trail" sub="Who changed what: deploys, key mints, rollbacks, revocations.">
          {audit.length === 0 ? (
            <Empty title="No audit entries" body="Actions taken through the API are recorded here." />
          ) : (
            <table>
              <thead>
                <tr>
                  <th>When</th>
                  <th>Actor</th>
                  <th>Action</th>
                  <th>Target</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {audit.map((entry, index) => (
                  <tr key={index}>
                    <td className="small nowrap">{formatTime((entry.ts as string) ?? null)}</td>
                    <td className="small mono">{String(entry.actor ?? '—')}</td>
                    <td>
                      <Badge tone="blue">{String(entry.action ?? '—')}</Badge>
                    </td>
                    <td className="small mono">{String(entry.target ?? '—')}</td>
                    <td className="small faint truncate" style={{ maxWidth: 320 }}>
                      {entry.detail ? JSON.stringify(entry.detail).slice(0, 120) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
      )}

      {tab !== 'audit' && shown.length > 0 ? (
        <Card title="Selection detail" sub="The last line, expanded — useful when a message is truncated.">
          <KeyValue
            items={[
              ['timestamp', shown[shown.length - 1].ts],
              ['level', shown[shown.length - 1].level],
              ['source', shown[shown.length - 1].source],
              ['runtime', shown[shown.length - 1].runtime_id],
              ['message', shown[shown.length - 1].message],
            ]}
          />
          {shown[shown.length - 1].data ? (
            <div style={{ marginTop: 10 }}>
              <pre className="code">{prettyJson(shown[shown.length - 1].data)}</pre>
            </div>
          ) : null}
        </Card>
      ) : null}
    </>
  )
}
