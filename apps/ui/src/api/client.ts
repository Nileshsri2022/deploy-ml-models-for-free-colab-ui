/**
 * API client.
 *
 * Two properties matter more than convenience here:
 *
 * 1. **Relative URLs only.** The browser must never be told where the backend
 *    lives - Vite proxies `/api`, `/v1` and `/internal` in dev, and in
 *    production the gateway serves this SPA itself. That is what makes the app
 *    work behind a preview proxy host without CORS or hardcoded origins.
 *
 * 2. **Offline-tolerant reads.** The whole point of this project is that the
 *    connection drops. `cachedGet` keeps the last good response in localStorage
 *    and hands it back with `stale: true` when the network or the gateway is
 *    unreachable, so the dashboard shows "last known state · 2 min ago" instead
 *    of an empty screen (architecture.md §8, Tier 1).
 */

const API_KEY_STORAGE = 'colabdeploy.apiKey'
const GATEWAY_STORAGE = 'colabdeploy.gatewayUrl'
const CACHE_PREFIX = 'colabdeploy.cache.'
const CACHE_TTL_MS = 5 * 60 * 1000

export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly detail: unknown

  constructor(message: string, status: number, code = 'ERROR', detail: unknown = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.detail = detail
  }

  get isAuthError(): boolean {
    return this.status === 401 || this.status === 403
  }

  get isOffline(): boolean {
    return this.status === 0 || this.status === 503 || this.code === 'RUNTIME_OFFLINE'
  }
}

/** Response plus a flag saying whether it came from the network or the shelf. */
export interface Result<T> {
  data: T
  stale: boolean
  fetchedAt: number | null
  error?: ApiError
}

// --------------------------------------------------------------------------- //
// Credentials
// --------------------------------------------------------------------------- //
export function getApiKey(): string {
  try {
    return localStorage.getItem(API_KEY_STORAGE) ?? ''
  } catch {
    return ''
  }
}

export function setApiKey(key: string): void {
  try {
    if (key) localStorage.setItem(API_KEY_STORAGE, key.trim())
    else localStorage.removeItem(API_KEY_STORAGE)
  } catch {
    /* private mode: keep going without persistence */
  }
}

/**
 * Optional gateway override. Empty means "same origin", which is correct for
 * both the Vite proxy and the gateway-served build. It exists so you can point
 * the static UI at a remote control plane.
 */
export function getGatewayUrl(): string {
  try {
    return localStorage.getItem(GATEWAY_STORAGE) ?? ''
  } catch {
    return ''
  }
}

export function setGatewayUrl(url: string): void {
  try {
    if (url) localStorage.setItem(GATEWAY_STORAGE, url.trim().replace(/\/+$/, ''))
    else localStorage.removeItem(GATEWAY_STORAGE)
  } catch {
    /* ignore */
  }
}

function apiUrl(path: string): string {
  const base = getGatewayUrl()
  const clean = path.startsWith('/') ? path : `/${path}`
  return `${base}${clean}`
}

/**
 * Fully-qualified URL for a path. `apiUrl` keeps relative URLs for fetch (so
 * the proxy decides where traffic goes); this one resolves against the browser
 * origin for things that are *displayed* - curl snippets, notebook cells, the
 * relay websocket URL.
 */
export function absoluteUrl(path: string): string {
  const base = getGatewayUrl()
  const clean = path.startsWith('/') ? path : `/${path}`
  if (base) return `${base}${clean}`
  if (typeof window === 'undefined') return clean
  return `${window.location.origin}${clean}`
}

/** Same, but pointed at the gateway rather than the UI origin. */
export function gatewayUrl(): string {
  return getGatewayUrl() || (typeof window === 'undefined' ? '' : window.location.origin)
}

function newRequestId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`
}

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const key = getApiKey()
  return {
    Accept: 'application/json',
    'X-Request-Id': newRequestId(),
    ...(key ? { Authorization: `Bearer ${key}` } : {}),
    ...extra,
  }
}

// --------------------------------------------------------------------------- //
// Core request
// --------------------------------------------------------------------------- //
async function parseError(response: Response): Promise<ApiError> {
  let message = `HTTP ${response.status}`
  let code = `HTTP_${response.status}`
  let detail: unknown = null
  try {
    const body = await response.json()
    if (body && typeof body === 'object') {
      const error = (body as { error?: { message?: string; code?: string; detail?: unknown } }).error
      if (error) {
        message = error.message ?? message
        code = error.code ?? code
        detail = error.detail ?? null
      } else if ('detail' in body) {
        // FastAPI's own validation errors
        message = JSON.stringify(body.detail)
        code = 'VALIDATION_ERROR'
        detail = body.detail
      }
    }
  } catch {
    try {
      message = (await response.text()).slice(0, 300) || message
    } catch {
      /* keep the default */
    }
  }
  return new ApiError(message, response.status, code, detail)
}

export async function request<T>(
  method: string,
  path: string,
  options: {
    body?: unknown
    form?: FormData
    query?: Record<string, string | number | boolean | undefined | null>
    signal?: AbortSignal
    timeoutMs?: number
  } = {},
): Promise<T> {
  const { body, form, query, signal } = options
  let url = apiUrl(path)
  if (query) {
    const search = new URLSearchParams()
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined && value !== null && value !== '') search.set(key, String(value))
    }
    const qs = search.toString()
    if (qs) url += `?${qs}`
  }

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs ?? 60_000)
  if (signal) {
    if (signal.aborted) controller.abort()
    else signal.addEventListener('abort', () => controller.abort(), { once: true })
  }

  let response: Response
  try {
    response = await fetch(url, {
      method,
      headers: headers(form ? {} : body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      body: form ?? (body !== undefined ? JSON.stringify(body) : undefined),
      signal: controller.signal,
    })
  } catch (error) {
    const offline = error instanceof DOMException && error.name === 'AbortError'
    throw new ApiError(
      offline
        ? `request timed out after ${Math.round((options.timeoutMs ?? 60_000) / 1000)}s`
        : `cannot reach the control plane at ${apiUrl(path)} (${(error as Error).message})`,
      offline ? 408 : 0,
      offline ? 'TIMEOUT' : 'GATEWAY_UNREACHABLE',
    )
  } finally {
    clearTimeout(timeout)
  }

  if (!response.ok) throw await parseError(response)
  if (response.status === 204) return undefined as T
  const text = await response.text()
  if (!text) return undefined as T
  try {
    return JSON.parse(text) as T
  } catch {
    return text as unknown as T
  }
}

export const api = {
  get: <T>(path: string, query?: Record<string, string | number | boolean | undefined | null>) =>
    request<T>('GET', path, { query }),
  post: <T>(path: string, body?: unknown, query?: Record<string, string | number | boolean | undefined>) =>
    request<T>('POST', path, { body, query }),
  patch: <T>(path: string, body?: unknown) => request<T>('PATCH', path, { body }),
  delete: <T>(path: string, query?: Record<string, string | number | boolean | undefined>) =>
    request<T>('DELETE', path, { query }),
  upload: <T>(path: string, form: FormData) => request<T>('POST', path, { form, timeoutMs: 300_000 }),
}

// --------------------------------------------------------------------------- //
// Offline read-through cache (Tier 1)
// --------------------------------------------------------------------------- //
function shelfKey(path: string, query?: Record<string, unknown>): string {
  return CACHE_PREFIX + path + (query ? `?${JSON.stringify(query)}` : '')
}

function readShelf<T>(key: string): { data: T; fetchedAt: number } | null {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return null
    const parsed = JSON.parse(raw) as { data: T; fetchedAt: number }
    if (!parsed || typeof parsed.fetchedAt !== 'number') return null
    return parsed
  } catch {
    return null
  }
}

function writeShelf(key: string, data: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify({ data, fetchedAt: Date.now() }))
  } catch {
    /* quota exceeded or private mode - the UI still works, just not offline */
  }
}

/**
 * GET with a last-known-good fallback.
 *
 * On success the value is shelved and returned fresh. On failure a shelved value
 * is returned with `stale: true` so the caller can render it with a "showing
 * cached state" banner; if nothing is shelved, the error is rethrown.
 */
export async function cachedGet<T>(
  path: string,
  query?: Record<string, string | number | boolean | undefined | null>,
  options: { maxAgeMs?: number } = {},
): Promise<Result<T>> {
  const key = shelfKey(path, query ?? undefined)
  try {
    const data = await api.get<T>(path, query)
    writeShelf(key, data)
    return { data, stale: false, fetchedAt: Date.now() }
  } catch (error) {
    const shelved = readShelf<T>(key)
    const age = shelved ? Date.now() - shelved.fetchedAt : Number.POSITIVE_INFINITY
    // An entry older than `maxAgeMs` is worse than no data at all: it would show
    // a state the operator cannot recognise as ancient. Default is "always serve".
    const usable = shelved && (options.maxAgeMs === undefined || age <= options.maxAgeMs)
    if (shelved && usable) {
      return {
        data: shelved.data,
        stale: true,
        fetchedAt: shelved.fetchedAt,
        error: error instanceof ApiError ? error : new ApiError(String(error), 0),
      }
    }
    throw error
  }
}

/** Drop every shelved response (used after deploys and by "Clear cache"). */
export function clearShelf(): number {
  let removed = 0
  try {
    for (const key of Object.keys(localStorage)) {
      if (key.startsWith(CACHE_PREFIX)) {
        localStorage.removeItem(key)
        removed += 1
      }
    }
  } catch {
    /* ignore */
  }
  return removed
}

export function shelfAgeLabel(fetchedAt: number | null): string {
  if (!fetchedAt) return 'never'
  const seconds = Math.max(0, Math.round((Date.now() - fetchedAt) / 1000))
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`
  return `${Math.round(seconds / 3600)}h ago`
}

export { CACHE_TTL_MS }
