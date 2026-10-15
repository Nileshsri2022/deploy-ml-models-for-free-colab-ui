/** Small presentational primitives shared by every page. */
import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { shelfAgeLabel } from '../api/client'
import type { DeploymentState, RuntimeState } from '../api/types'

// --------------------------------------------------------------------------- //
export function Badge({
  tone = 'gray',
  children,
  dot = false,
  title,
}: {
  tone?: 'green' | 'amber' | 'red' | 'blue' | 'gray' | 'accent'
  children: ReactNode
  dot?: boolean
  title?: string
}) {
  return (
    <span className={`badge ${tone}`} title={title}>
      {dot ? <span className="dot" /> : null}
      {children}
    </span>
  )
}

const RUNTIME_TONES: Record<RuntimeState, 'green' | 'amber' | 'red' | 'gray' | 'blue'> = {
  online: 'green',
  degraded: 'amber',
  connecting: 'blue',
  registered: 'gray',
  stale: 'amber',
  unknown: 'gray',
  offline: 'red',
}

export function RuntimeBadge({ state }: { state: RuntimeState }) {
  return (
    <Badge tone={RUNTIME_TONES[state] ?? 'gray'} dot>
      {state}
    </Badge>
  )
}

const DEPLOYMENT_TONES: Record<DeploymentState, 'green' | 'amber' | 'red' | 'gray'> = {
  ready: 'green',
  loading: 'amber',
  pending: 'gray',
  failed: 'red',
  unloaded: 'gray',
}

export function DeploymentBadge({ state }: { state: DeploymentState | null | undefined }) {
  if (!state) return <Badge tone="gray">not deployed</Badge>
  return (
    <Badge tone={DEPLOYMENT_TONES[state] ?? 'gray'} dot>
      {state}
    </Badge>
  )
}

export function CacheBadge({ status }: { status: string }) {
  const tone = status === 'HIT' ? 'green' : status === 'STALE' ? 'amber' : status === 'MISS' ? 'blue' : 'gray'
  return <Badge tone={tone}>{status}</Badge>
}

// --------------------------------------------------------------------------- //
export function Card({
  title,
  sub,
  actions,
  children,
}: {
  title?: ReactNode
  sub?: ReactNode
  actions?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="card">
      {(title || actions) && (
        <div className="spread" style={{ marginBottom: sub ? 2 : 12 }}>
          <h2 className="card-title">{title}</h2>
          {actions ? <div className="row">{actions}</div> : null}
        </div>
      )}
      {sub ? <p className="card-sub">{sub}</p> : null}
      {children}
    </section>
  )
}

export function Stat({
  label,
  value,
  hint,
  tone,
}: {
  label: string
  value: ReactNode
  hint?: ReactNode
  tone?: 'green' | 'amber' | 'red'
}) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value" style={tone ? { color: `var(--${tone})` } : undefined}>
        {value}
      </span>
      {hint ? <span className="stat-hint">{hint}</span> : null}
    </div>
  )
}

export function Empty({
  title,
  body,
  action,
}: {
  title: string
  body: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      <p>{body}</p>
      {action}
    </div>
  )
}

export function Banner({
  tone = 'amber',
  children,
}: {
  tone?: 'amber' | 'red' | 'green' | 'blue'
  children: ReactNode
}) {
  return <div className={`banner ${tone === 'amber' ? '' : tone}`}>{children}</div>
}

// --------------------------------------------------------------------------- //
export function CopyButton({ value, label = 'Copy' }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) return
    const timer = setTimeout(() => setCopied(false), 1400)
    return () => clearTimeout(timer)
  }, [copied])

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
    } catch {
      // Clipboard API needs a secure context; fall back to a hidden textarea.
      const area = document.createElement('textarea')
      area.value = value
      area.style.position = 'fixed'
      area.style.opacity = '0'
      document.body.appendChild(area)
      area.select()
      try {
        document.execCommand('copy')
        setCopied(true)
      } catch {
        /* nothing else to try */
      }
      document.body.removeChild(area)
    }
  }, [value])

  return (
    <button type="button" className="small ghost" onClick={copy} title="Copy to clipboard">
      {copied ? '✓ Copied' : label}
    </button>
  )
}

export function CodeBlock({
  label,
  code,
  copy = true,
}: {
  label?: string
  code: string
  copy?: boolean
}) {
  return (
    <div>
      {label || copy ? (
        <div className="code-head">
          <span className="code-label">{label}</span>
          {copy ? <CopyButton value={code} /> : null}
        </div>
      ) : null}
      <pre className="code">{code}</pre>
    </div>
  )
}

// --------------------------------------------------------------------------- //
export function Spinner({ label }: { label?: string }) {
  return (
    <span className="row" style={{ gap: 8 }}>
      <span className="spinner" />
      {label ? <span className="muted small">{label}</span> : null}
    </span>
  )
}

export function Meter({ value, max }: { value: number; max: number }) {
  const pct = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0
  return (
    <div className="meter" title={`${Math.round(pct)}%`}>
      <div style={{ width: `${pct}%` }} />
    </div>
  )
}

export function KeyValue({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="kv">
      {items.map(([key, value]) => (
        <div key={key} style={{ display: 'contents' }}>
          <dt>{key}</dt>
          <dd>{value ?? <span className="faint">—</span>}</dd>
        </div>
      ))}
    </dl>
  )
}

// --------------------------------------------------------------------------- //
// Formatting helpers
// --------------------------------------------------------------------------- //
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—'
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unit]}`
}

export function formatSeconds(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '—'
  const abs = Math.abs(seconds)
  if (abs < 60) return `${Math.round(seconds)}s`
  if (abs < 3600) return `${Math.round(seconds / 60)}m`
  if (abs < 86400) return `${(seconds / 3600).toFixed(1)}h`
  return `${Math.round(seconds / 86400)}d`
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleTimeString([], { hour12: false })
  } catch {
    return iso
  }
}

export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return 'never'
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return iso
  return shelfAgeLabel(then)
}

/** Pretty-print anything the API returns, capped so huge outputs stay usable. */
export function prettyJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2).slice(0, 20000)
  } catch {
    return String(value).slice(0, 2000)
  }
}

/** Parse a JSON textarea, returning an error message instead of throwing. */
export function parseJsonField(raw: string): { value?: unknown; error?: string } {
  const text = raw.trim()
  if (!text) return { value: undefined }
  try {
    return { value: JSON.parse(text) }
  } catch (error) {
    return { error: (error as Error).message }
  }
}
