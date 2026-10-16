import * as Sentry from '@sentry/react'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'
import { gatewayUrl } from './api/client'
import App from './App'
import { initializeSentry, type SentryPublicConfig } from './observability/sentry'
import { StoreProvider } from './state/store'
import './styles.css'

/*
  Hash routing, deliberately. The built bundle is served by the gateway itself
  (see `_mount_spa` in apps/gateway/gateway/main.py) on free hosts where nobody
  controls the reverse proxy in front of the app, so `/models` would 404 on a
  hard reload while `/#/models` never does. Relative asset URLs + hash routes =
  the same `dist/` works from any prefix, including a preview proxy.
*/

const container = document.getElementById('root')
if (!container) throw new Error('#root missing from index.html')
const root = createRoot(container)

async function loadSentryConfig(): Promise<SentryPublicConfig | null> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), 700)
  try {
    const response = await fetch(`${gatewayUrl()}/api/public-config`, {
      headers: { Accept: 'application/json' },
      credentials: 'omit',
      signal: controller.signal,
    })
    if (!response.ok) return null
    return (await response.json()) as SentryPublicConfig
  } catch {
    // Keep the UI bootable when the gateway is offline or Sentry config is absent.
    return null
  } finally {
    window.clearTimeout(timeout)
  }
}

async function bootstrap(): Promise<void> {
  const sentryConfig = await loadSentryConfig()
  initializeSentry(sentryConfig)

  root.render(
    <StrictMode>
      <HashRouter>
        <StoreProvider>
          <Sentry.ErrorBoundary
            fallback={
              <main className="page">
                <section className="card" role="alert">
                  <h1 className="page-title">Something went wrong</h1>
                  <p className="card-sub">
                    If Sentry is configured, the error was sent there. Otherwise check the browser console. Reload the page to try again.
                  </p>
                  <button type="button" onClick={() => window.location.reload()}>
                    Reload
                  </button>
                </section>
              </main>
            }
          >
            <App />
          </Sentry.ErrorBoundary>
        </StoreProvider>
      </HashRouter>
    </StrictMode>,
  )
}

void bootstrap()
