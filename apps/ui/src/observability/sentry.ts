import * as Sentry from '@sentry/react'
import { gatewayUrl } from '../api/client'

export interface SentryPublicConfig {
  sentry_dsn?: string | null
  sentry_environment?: string | null
  sentry_release?: string | null
  sentry_traces_sample_rate?: number
}

const SECRET_KEY = /authorization|cookie|password|passwd|secret|token|api[_-]?key|body|payload|inputs?|outputs?/i
const INLINE_SECRET = /\b(authorization|cookie|x-runtime-token|api_key|token|password|secret)\s*[:=]\s*([^\s&,;]+)/gi
const CDP_TOKEN = /\bCDP_(?:API|RT)_[A-Za-z0-9_-]+\b/g
const REDIS_CREDENTIALS = /(rediss?:\/\/)[^/@\s]+@/gi

function scrubText(value: string): string {
  return value
    .replace(REDIS_CREDENTIALS, '$1[redacted]@')
    .replace(CDP_TOKEN, '[redacted-token]')
    .replace(INLINE_SECRET, '$1=[redacted]')
}

function scrubUrl(value: string): string {
  try {
    const url = new URL(value, window.location.origin)
    return `${url.origin}${url.pathname}`
  } catch {
    return value.split('?')[0].split('#')[0]
  }
}

function scrubValue(value: unknown, key = ''): unknown {
  const normalizedKey = key.toLowerCase()
  if (SECRET_KEY.test(key) || normalizedKey.includes('query') || normalizedKey === 'vars' || normalizedKey === 'variables') return '[redacted]'
  if (Array.isArray(value)) return value.map((item) => scrubValue(item))
  if (typeof value === 'string') {
    return ['url', 'http.url', 'url.full', 'http.target'].includes(normalizedKey)
      ? scrubUrl(value)
      : scrubText(value)
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([childKey, childValue]) => [
        childKey,
        scrubValue(childValue, childKey),
      ]),
    )
  }
  return value
}

function scrubEvent(event: unknown): unknown {
  if (!event || typeof event !== 'object') return event
  const safe = event as Record<string, unknown>
  const request = safe.request
  if (request && typeof request === 'object') {
    const safeRequest = request as Record<string, unknown>
    if (typeof safeRequest.url === 'string') safeRequest.url = scrubUrl(safeRequest.url)
    for (const key of ['headers', 'cookies', 'data', 'query_string', 'env']) delete safeRequest[key]
  }
  delete safe.user
  for (const key of ['breadcrumbs', 'extra', 'contexts', 'tags', 'spans']) {
    if (key in safe) safe[key] = scrubValue(safe[key])
  }
  const exception = safe.exception
  let hasException = false
  if (exception && typeof exception === 'object') {
    const values = (exception as Record<string, unknown>).values
    hasException = Array.isArray(values) && values.length > 0
    if (Array.isArray(values)) {
      for (const item of values) {
        if (item && typeof item === 'object') {
          const value = item as Record<string, unknown>
          if (typeof value.value === 'string') value.value = '[exception details redacted]'
          const stacktrace = value.stacktrace
          if (stacktrace && typeof stacktrace === 'object') {
            const frames = (stacktrace as Record<string, unknown>).frames
            if (Array.isArray(frames)) {
              for (const frame of frames) {
                if (frame && typeof frame === 'object') delete (frame as Record<string, unknown>).vars
              }
            }
          }
        }
      }
    }
  }
  for (const key of ['message', 'transaction']) {
    if (typeof safe[key] === 'string') {
      safe[key] = key === 'message' && hasException ? '[exception details redacted]' : scrubText(safe[key] as string)
    }
  }
  const logentry = safe.logentry
  if (logentry && typeof logentry === 'object') {
    const entry = logentry as Record<string, unknown>
    if (typeof entry.message === 'string') {
      entry.message = hasException ? '[exception details redacted]' : scrubText(entry.message)
    }
    if (Array.isArray(entry.params) || (entry.params && typeof entry.params === 'object')) {
      entry.params = scrubValue(entry.params)
    }
  }
  return safe
}

export function initializeSentry(config: SentryPublicConfig | null): boolean {
  const dsn = config?.sentry_dsn?.trim()
  if (!dsn) return false
  const sampleRate = Math.min(1, Math.max(0, Number(config?.sentry_traces_sample_rate ?? 0) || 0))

  const gatewayBase = gatewayUrl().replace(/\/+$/, '')
  const escapedGatewayBase = gatewayBase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

  try {
    Sentry.init({
      dsn,
      environment: config?.sentry_environment || 'production',
      release: config?.sentry_release ? `colabdeploy-ui@${config.sentry_release}` : undefined,
      tracesSampleRate: sampleRate,
      tracePropagationTargets: [
        /^\/(api|v1|internal)(\/|$)/,
        new RegExp(`^${escapedGatewayBase}/(api|v1|internal)(/|$)`, 'i'),
      ],
      integrations: (defaults) => [
        ...defaults.filter((integration) => integration.name !== 'Console' && integration.name !== 'Breadcrumbs'),
        // Keep safe request/navigation breadcrumbs, but never collect console output, DOM text, or Sentry event messages.
        Sentry.breadcrumbsIntegration({ dom: false, sentry: false }),
        ...(sampleRate > 0 ? [Sentry.browserTracingIntegration()] : []),
      ],
      beforeSend: (event) => scrubEvent(event) as typeof event,
      beforeSendTransaction: (event) => scrubEvent(event) as typeof event,
    })
    return true
  } catch {
    // A bad DSN must never prevent the control UI from booting.
    return false
  }
}
