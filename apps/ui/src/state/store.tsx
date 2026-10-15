/**
 * App-wide state: polled status, live SSE logs, and toasts.
 *
 * Status is polled rather than pushed because the dashboard must keep working
 * when the SSE stream is blocked by a proxy; logs use SSE because a tail is
 * naturally one-directional. Both degrade to "last known" via `cachedGet`.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { ApiError, api, cachedGet, getApiKey } from '../api/client'
import type { LogLine, StatusResponse, Telemetry } from '../api/types'

export interface Toast {
  id: number
  kind: 'success' | 'error' | 'info'
  message: string
  detail?: string
}

interface Store {
  status: StatusResponse | null
  statusStale: boolean
  statusAge: number | null
  statusError: string | null
  telemetry: Telemetry | null
  logs: LogLine[]
  connected: boolean
  hasKey: boolean
  loading: boolean
  refresh: () => Promise<void>
  clearLogs: () => void
  toast: (kind: Toast['kind'], message: string, detail?: string) => void
  toasts: Toast[]
  dismissToast: (id: number) => void
}

const StoreContext = createContext<Store | null>(null)

const STATUS_POLL_MS = 5000
const TELEMETRY_POLL_MS = 10000
const MAX_LOGS = 500

export function StoreProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [statusStale, setStatusStale] = useState(false)
  const [statusAge, setStatusAge] = useState<number | null>(null)
  const [statusError, setStatusError] = useState<string | null>(null)
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const [logs, setLogs] = useState<LogLine[]>([])
  const [connected, setConnected] = useState(false)
  const [loading, setLoading] = useState(true)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [hasKey, setHasKey] = useState<boolean>(() => Boolean(getApiKey()))
  const toastId = useRef(0)

  const toast = useCallback((kind: Toast['kind'], message: string, detail?: string) => {
    const id = ++toastId.current
    setToasts((current) => [...current.slice(-4), { id, kind, message, detail }])
    setTimeout(() => setToasts((current) => current.filter((t) => t.id !== id)), kind === 'error' ? 9000 : 4500)
  }, [])

  const dismissToast = useCallback((id: number) => {
    setToasts((current) => current.filter((t) => t.id !== id))
  }, [])

  const refresh = useCallback(async () => {
    try {
      const result = await cachedGet<StatusResponse>('/v1/status')
      setStatus(result.data)
      setStatusStale(result.stale)
      setStatusAge(result.fetchedAt)
      setStatusError(result.error ? result.error.message : null)
      setConnected(!result.stale)
    } catch (error) {
      setConnected(false)
      setStatusStale(true)
      setStatusError(error instanceof ApiError ? error.message : String(error))
    } finally {
      setLoading(false)
    }
    try {
      setTelemetry(await api.get<Telemetry>('/api/telemetry'))
    } catch {
      /* telemetry is a nicety; a failure here must not blank the dashboard */
    }
    setHasKey(Boolean(getApiKey()))
  }, [])

  // Status polling.
  useEffect(() => {
    void refresh()
    const timer = setInterval(() => void refresh(), STATUS_POLL_MS)
    return () => clearInterval(timer)
  }, [refresh])

  // Slower telemetry poll, only while the tab is visible.
  useEffect(() => {
    const timer = setInterval(() => {
      if (document.visibilityState === 'visible') {
        api
          .get<Telemetry>('/api/telemetry')
          .then(setTelemetry)
          .catch(() => undefined)
      }
    }, TELEMETRY_POLL_MS)
    return () => clearInterval(timer)
  }, [])

  // Live log tail over SSE.
  useEffect(() => {
    if (!hasKey) return
    let source: EventSource | null = null
    let retry: number | undefined
    let stopped = false

    const open = () => {
      if (stopped) return
      try {
        // EventSource cannot set headers, so the key rides in the query string.
        source = new EventSource(`/api/logs/stream?api_key=${encodeURIComponent(getApiKey())}`)
      } catch {
        return
      }
      source.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as LogLine & { event?: string }
          if (payload.event === 'ready' || payload.event === 'error') return
          setLogs((current) => [...current, payload].slice(-MAX_LOGS))
        } catch {
          /* ignore malformed frames */
        }
      }
      source.onerror = () => {
        source?.close()
        source = null
        if (!stopped) retry = window.setTimeout(open, 3000)
      }
    }

    open()
    return () => {
      stopped = true
      if (retry) clearTimeout(retry)
      source?.close()
    }
  }, [hasKey])

  const clearLogs = useCallback(() => setLogs([]), [])

  const value = useMemo<Store>(
    () => ({
      status,
      statusStale,
      statusAge,
      statusError,
      telemetry,
      logs,
      connected,
      hasKey,
      loading,
      refresh,
      clearLogs,
      toast,
      toasts,
      dismissToast,
    }),
    [
      status,
      statusStale,
      statusAge,
      statusError,
      telemetry,
      logs,
      connected,
      hasKey,
      loading,
      refresh,
      clearLogs,
      toast,
      toasts,
      dismissToast,
    ],
  )

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>
}

export function useStore(): Store {
  const store = useContext(StoreContext)
  if (!store) throw new Error('useStore must be used inside <StoreProvider>')
  return store
}

/** Convenience: the runtimes that are actually able to serve right now. */
export function useOnlineRuntimes() {
  const { status } = useStore()
  return useMemo(
    () => (status?.runtimes ?? []).filter((r) => r.state === 'online' || r.state === 'degraded'),
    [status],
  )
}
