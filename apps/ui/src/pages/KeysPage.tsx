import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, api, clearShelf, getApiKey, getGatewayUrl, setApiKey, setGatewayUrl } from '../api/client'
import type { ApiKeyInfo } from '../api/types'
import {
  Badge,
  Banner,
  Card,
  CodeBlock,
  CopyButton,
  Empty,
  KeyValue,
  Spinner,
  Stat,
  formatTime,
  prettyJson,
  relativeTime,
} from '../components/ui'
import { useStore } from '../state/store'

const SCOPES = [
  { id: 'inference', label: 'inference', hint: 'call /v1 predict endpoints' },
  { id: 'manage', label: 'manage', hint: 'register models, deploy, read logs and telemetry' },
]

/**
 * API keys + local UI settings.
 *
 * Keys are hashed server-side: the plaintext is returned once at creation and
 * never again, so "use this key" only works on the row that was just minted.
 */
export default function KeysPage() {
  const { status, toast, refresh, hasKey } = useStore()
  const [keys, setKeys] = useState<ApiKeyInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [label, setLabel] = useState('browser')
  const [scopes, setScopes] = useState<string[]>(['inference', 'manage'])
  const [expiresDays, setExpiresDays] = useState<number | ''>('')
  const [busy, setBusy] = useState(false)
  const [minted, setMinted] = useState<ApiKeyInfo | null>(null)
  const [activeKey, setActiveKey] = useState(() => getApiKey())
  const [gateway, setGateway] = useState(() => getGatewayUrl())

  const load = useCallback(async () => {
    try {
      const result = await api.get<{ keys: ApiKeyInfo[] }>('/api/keys')
      setKeys(result.keys ?? [])
      setError(null)
    } catch (err) {
      setError(
        err instanceof ApiError && err.isAuthError
          ? 'Listing keys needs a key with the manage scope — paste one below.'
          : (err as Error).message,
      )
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const create = async () => {
    setBusy(true)
    try {
      const created = await api.post<ApiKeyInfo>('/api/keys', {
        label: label.trim() || 'browser',
        scopes,
        expires_in_days: expiresDays === '' ? null : Number(expiresDays),
      })
      setMinted(created)
      toast('success', 'API key created', `${created.prefix}… — copy it now`)
      await load()
    } catch (err) {
      setError((err as Error).message)
      toast('error', 'Could not create a key', (err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const revoke = async (key: ApiKeyInfo) => {
    if (!window.confirm(`Revoke ${key.label} (${key.prefix}…)? Anything using it stops working now.`)) return
    try {
      await api.delete(`/api/keys/${key.id}`)
      toast('info', 'Key revoked')
      if (getApiKey().startsWith(key.prefix)) {
        setApiKey('')
        setActiveKey('')
      }
      await load()
      await refresh()
    } catch (err) {
      toast('error', 'Revoke failed', (err as Error).message)
    }
  }

  const useKey = (token: string) => {
    setApiKey(token)
    setActiveKey(token)
    toast('success', 'This browser now authenticates with that key', 'stored in localStorage only')
    void load()
    void refresh()
  }

  const saveGateway = () => {
    setGatewayUrl(gateway)
    toast('success', gateway ? `Pointing at ${gateway}` : 'Using same-origin gateway')
    void refresh()
  }

  const wipeLocal = () => {
    const removed = clearShelf()
    setApiKey('')
    setActiveKey('')
    setGatewayUrl('')
    setGateway('')
    toast('info', `Cleared ${removed} cached views and the stored key`)
    void refresh()
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">API keys &amp; settings</h1>
          <p className="page-sub">
            Two scopes: <code className="mono">inference</code> for callers of your endpoints and{' '}
            <code className="mono">manage</code> for this UI. Runtime tokens are separate — those
            authenticate a VM, not a client.
          </p>
        </div>
      </div>

      {error && <Banner tone="red">{error}</Banner>}

      <div className="grid cols-4">
        <Card>
          <Stat label="Active keys" value={keys.filter((k) => !k.revoked).length} hint={`${keys.length} total`} />
        </Card>
        <Card>
          <Stat
            label="This browser"
            value={hasKey ? 'authenticated' : 'anonymous'}
            tone={hasKey ? 'green' : 'amber'}
            hint={activeKey ? `${activeKey.slice(0, 12)}…` : 'no key stored'}
          />
        </Card>
        <Card>
          <Stat label="Requests seen" value={status?.requests_total ?? '—'} hint={`${status?.requests_cached ?? 0} served from cache`} />
        </Card>
        <Card>
          <Stat label="Gateway" value={status?.version ?? '—'} hint={status ? `${Math.round(status.uptime_s)}s uptime` : 'unreachable'} />
        </Card>
      </div>

      <div className="grid cols-2" style={{ marginTop: 14 }}>
        <Card title="Create a key" sub="The plaintext token is shown exactly once — the gateway stores a hash.">
          <label className="field">
            <span>Label</span>
            <input value={label} onChange={(event) => setLabel(event.target.value)} placeholder="browser / ci / mobile" />
          </label>
          <label className="field">
            <span>Scopes</span>
            <div className="pill-row">
              {SCOPES.map((scope) => {
                const active = scopes.includes(scope.id)
                return (
                  <button
                    key={scope.id}
                    type="button"
                    className={active ? 'primary small' : 'small'}
                    title={scope.hint}
                    onClick={() =>
                      setScopes((current) =>
                        current.includes(scope.id) ? current.filter((s) => s !== scope.id) : [...current, scope.id],
                      )
                    }
                  >
                    {scope.id}
                  </button>
                )
              })}
            </div>
            <div className="hint">At least one scope is required; inference-only keys cannot read this page.</div>
          </label>
          <label className="field">
            <span>Expires in (days)</span>
            <input
              type="number"
              min={1}
              value={expiresDays}
              onChange={(event) => setExpiresDays(event.target.value === '' ? '' : Number(event.target.value))}
              placeholder="never"
            />
          </label>
          <div className="row">
            <button type="button" className="primary" disabled={busy || scopes.length === 0} onClick={() => void create()}>
              {busy ? <Spinner label="creating…" /> : 'Create key'}
            </button>
          </div>

          {minted ? (
            <div style={{ marginTop: 14 }}>
              <Banner tone="green">
                <strong>Copy it now.</strong> This is the only time the plaintext appears.
              </Banner>
              <CodeBlock label={`token · ${minted.prefix}…`} code={minted.token ?? ''} />
              <div className="row" style={{ marginTop: 8 }}>
                <button type="button" className="small primary" onClick={() => useKey(minted.token ?? '')} disabled={!minted.token}>
                  Use this key in the browser
                </button>
                <CopyButton value={minted.token ?? ''} label="Copy token" />
              </div>
            </div>
          ) : null}
        </Card>

        <Card title="This browser" sub="Local settings only — nothing here is sent anywhere except as request headers.">
          <label className="field">
            <span>API key in use</span>
            <input
              type="password"
              value={activeKey}
              onChange={(event) => setActiveKey(event.target.value)}
              placeholder="CDP_API_…"
              autoComplete="off"
            />
            <div className="hint">Sent as <code className="mono">Authorization: Bearer …</code> on every call.</div>
          </label>
          <label className="field">
            <span>Gateway URL override</span>
            <input value={gateway} onChange={(event) => setGateway(event.target.value)} placeholder="empty = same origin" autoComplete="off" />
            <div className="hint">
              Leave empty unless the UI is hosted apart from the gateway. Relative URLs are what make the
              preview proxy work.
            </div>
          </label>
          <div className="row">
            <button type="button" className="primary" onClick={() => useKey(activeKey)}>
              Save key
            </button>
            <button type="button" onClick={saveGateway}>
              Save gateway
            </button>
            <button type="button" className="danger" onClick={wipeLocal}>
              Clear local state
            </button>
          </div>
          <p className="hint" style={{ marginTop: 10 }}>
            "Clear local state" drops the stored key, the gateway override and every last-known-good
            response the UI shelved — the offline view goes blank until the next successful poll.
          </p>
          {status ? (
            <div style={{ marginTop: 10 }}>
              <KeyValue
                items={[
                  ['gateway url', <span className="mono">{getGatewayUrl() || window.location.origin}</span>],
                  ['healthy', status.healthy ? <Badge tone="green" dot>yes</Badge> : <Badge tone="red" dot>no</Badge>],
                  ['runtimes', `${status.runtimes_online} online · ${status.runtimes_degraded} degraded · ${status.runtimes_total} total`],
                  ['models / endpoints', `${status.models} / ${status.endpoints}`],
                  ['cache', `${status.cache.entries} entries · ${(status.cache.hit_ratio * 100).toFixed(1)}% hit`],
                  ['jobs queued', status.jobs_queued],
                ]}
              />
            </div>
          ) : null}
        </Card>
      </div>

      <Card title={`Keys (${keys.length})`} sub="Revoking is immediate: in-flight requests finish, the next one is rejected.">
        {loading && keys.length === 0 ? (
          <Spinner label="loading keys…" />
        ) : keys.length === 0 ? (
          <Empty
            title="No keys"
            body="Create one above. If the gateway runs with auth disabled (development), inference works without a key but this page still needs manage scope."
            action={
              <Link className="btn primary" to="/connect">
                Connect a runtime instead
              </Link>
            }
          />
        ) : (
          <table>
            <thead>
              <tr>
                <th>Label</th>
                <th>Prefix</th>
                <th>Scopes</th>
                <th>Created</th>
                <th>Last used</th>
                <th>Expires</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {keys.map((key) => (
                <tr key={key.id} style={{ opacity: key.revoked ? 0.55 : 1 }}>
                  <td>
                    <strong>{key.label}</strong>
                    <div className="faint small mono">{key.id}</div>
                  </td>
                  <td className="mono small">{key.prefix}…</td>
                  <td>
                    <div className="pill-row">
                      {key.scopes.map((scope) => (
                        <Badge key={scope} tone={scope === 'manage' ? 'accent' : 'blue'}>
                          {scope}
                        </Badge>
                      ))}
                    </div>
                  </td>
                  <td className="small">{formatTime(key.created_at)}</td>
                  <td className="small faint">{relativeTime(key.last_used_at)}</td>
                  <td className="small">{key.expires_at ? formatTime(key.expires_at) : 'never'}</td>
                  <td>
                    {key.revoked ? (
                      <Badge tone="red">revoked</Badge>
                    ) : (
                      <button type="button" className="small danger" onClick={() => void revoke(key)}>
                        revoke
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Card title="Status payload" sub="What the UI polls every 5 seconds — the source of truth for every badge you see.">
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
