# Sentry integration specification

**Status:** Implemented — Sentry is opt-in until configured  
**Date:** 2026-10-16  
**Scope:** React/Vite UI, FastAPI gateway, and a Sentry Issues view inside the existing Logs page

## 1. Summary

Add an opt-in Sentry integration for unexpected browser and gateway errors, and let operators view recent Sentry issues from the existing **Logs** page.

There are two separate Sentry connections:

```text
React UI ── error events via public DSN ──► Sentry
FastAPI  ── error events via server DSN ──► Sentry
Logs page ── authenticated app API ──► Gateway ── read-only Sentry API token ──► Sentry Issues API
```

The Sentry SDK DSN sends events to Sentry; it does **not** let the Logs page read them back. The Logs page therefore needs a separate, read-only Sentry API credential kept on the gateway. The browser must never receive that credential.

Keep the current Live, History, and Audit views. Add a **Sentry** view within the Logs page for recent grouped Sentry issues, counts, last-seen time, and a link to the full issue in Sentry. This is an issue overview, not a mirror of every raw Sentry event or a replacement for the app’s regular logs.

## 2. Baseline and implementation notes

- The UI keeps Live, History, and Audit views. Live logs arrive over SSE; agent diagnostic messages are batched back to the gateway. The Sentry subtab is now an additional view, not a replacement.
- Gateway diagnostics use a bounded in-memory ring buffer. They are not a durable server-side log archive and are lost when the gateway process restarts. The UI may retain a last-known response in browser storage for offline display.
- Management audit records are separate and stored through the registry store.
- The gateway exposes summary telemetry at `/api/telemetry` and Prometheus-formatted metrics at `/v1/metrics`.
- Sentry reporting is opt-in: with DSNs configured, unexpected UI/gateway errors and Upstash outage transitions are reported externally. A separate read token is needed for the in-app Sentry issue list.
- The UI API client now sends `X-Request-Id`; the gateway accepts and returns it for correlation.

**History wording corrected:** the Logs page and API docs now describe diagnostic history as a bounded, process-local buffer. The audit-trail persistence description remains separate.

## 3. Goals

1. Report unexpected React/browser errors with useful stack traces and release/environment labels.
2. Report unexpected FastAPI gateway exceptions with Python tracebacks and safe request context.
3. Add a Sentry view inside the existing Logs page for recent Sentry issues.
4. Support optional, configurable browser-to-gateway performance tracing.
5. Preserve the current app behavior when Sentry is not configured or unreachable.
6. Avoid transmitting credentials, personal data, or model data.
7. Document configuration for local development and production deployments.

## 4. Non-goals for the first release

- Replacing the Live/History logs, audit trail, or `/v1/metrics`.
- Shipping every ordinary log line, every HTTP request, or every raw Sentry event to the Logs page.
- Recording request/response bodies, model inputs/outputs, uploaded files, API keys, or credentials.
- Session Replay. It is explicitly disabled in v1; it needs a separate privacy review and masking policy.
- Installing Sentry in the Colab agent or tracing execution across the custom WebSocket tunnel. Agent log forwarding continues to work as it does today. Agent-side Sentry and cross-tunnel spans can be considered later.
- Sending expected validation, authentication, quota, and offline/degraded-runtime outcomes as error issues.
- Copying Sentry issue data into the local audit store or treating it as an authoritative compliance audit log.

## 5. Proposed implementation

### 5.1 React/Vite UI

- Add the Sentry React SDK and initialize it before rendering the app in `apps/ui/src/main.tsx` (or a small `src/observability/sentry.ts` initializer imported there).
- Add a React error boundary at the application root. Keep normal UI error handling and toast behavior unchanged.
- Capture uncaught JavaScript errors and unhandled promise rejections. Do not turn expected `ApiError` states—such as 401/403, validation errors, quota responses, or expected offline/runtime-degraded behavior—into Sentry issues.
- Add Browser Tracing only when a non-zero trace sample rate is configured. Limit propagation to this application’s own `/api`, `/v1`, and `/internal` requests; do not propagate Sentry headers to arbitrary third-party origins.
- Add safe request breadcrumbs if useful (method, route, status, duration, request ID only). Do not include headers or request/response bodies.

### 5.2 FastAPI gateway

- Add the Python Sentry SDK with FastAPI support and initialize it once during application setup in `apps/gateway/gateway/main.py`.
- The app has a custom catch-all exception handler. That handler must explicitly report unexpected exceptions to Sentry before returning the existing 500 error envelope; do not rely only on automatic unhandled-exception capture.
- Do not report expected `HTTPException` or request-validation responses as unhandled issues. Preserve their current response status and JSON shape.
- Attach safe context such as environment, release, route, HTTP method, and the existing request ID. Do not attach authorization/runtime-token headers, query secrets, or request bodies.
- Optional FastAPI tracing uses the same configurable sample-rate policy as the browser. Sentry network/export failures must never delay or fail a user request.

### 5.3 Request correlation

- For UI API requests, generate and send an `X-Request-Id` when one is not already present. The gateway should continue to accept it and return it in the response.
- Include that ID as a Sentry tag/breadcrumb where safe, so an operator can compare a Sentry event with a gateway log or inference response.
- Use Sentry’s trace headers for HTTP performance trace propagation when tracing is enabled. This does not, by itself, trace work across the agent WebSocket tunnel; that remains out of scope for v1.

### 5.4 Sentry view in the Logs page

- Add a fourth subtab named **Sentry** (or **Sentry issues**) inside the existing Logs page. Do not add a separate navigation item or embed the Sentry website.
- Add an authenticated gateway endpoint such as `GET /api/sentry/issues?limit=...`. It must use the existing management authorization dependency.
- The gateway calls Sentry’s Issues API using the configured organization/project and a **read-only** API token. The Sentry token stays server-side in environment configuration; it must never appear in frontend JavaScript, API responses, logs, or error messages.
- Return only an allowlisted summary of each issue, for example: issue ID, short ID, title, level, status, event count, first/last seen, environment/release when available, and Sentry permalink. Do not return raw event payloads or user/model data.
- The UI loads issues when the Sentry subtab is selected, offers manual refresh, and may poll at a modest interval while that tab is open. This is a normal API poll, not an SSE stream. The gateway should cache the response briefly (about 30 seconds) and cap the result limit to protect Sentry API quotas.
- If issue details are needed, show a concise summary and an **Open in Sentry** link rather than duplicating full stack traces or event payloads in the app.
- If no read-only API credential is configured, show a clear “Sentry is not connected” explanation in this subtab. If Sentry is temporarily unavailable, show a local error banner while keeping Live, History, and Audit usable.
- The Sentry API integration is separate from the Sentry SDK DSN. Configuring a DSN alone enables event reporting but does not enable the in-app issue list.

### 5.5 Upstash Redis diagnostics

- Redis connection, restore, sync, keepalive, and inspector failures are intentionally caught so the gateway can continue using local JSON. Report a sanitized reason to the app's existing Logs stream/history and expose last-known status, error time, and last-success time through the Upstash status API and Cache page.
- If Sentry is configured, send one warning event per Redis outage episode—not one event on every 30-second retry. Emit a local recovery line when Redis reconnects and reset the episode guard so a later outage can be reported.
- Never include the Redis URL or password in log/Sentry events. Keep local-file fallback and inference/API availability unchanged.

### 5.6 Privacy and filtering

This is a hard requirement, not an optional setting:

- Explicitly disable collection of user identity, request headers/cookies/bodies, query parameters, generative-AI inputs/outputs, stack-frame locals, and source context. Disable Python log-event capture and browser console/DOM text breadcrumbs.
- Scrub `Authorization`, cookies, runtime tokens, API keys, and any values named `token`, `key`, `body`, `payload`, `input`, or `output` from event data, breadcrumbs, URLs, spans, and query strings.
- The current SSE URL contains an `api_key` query parameter because browser `EventSource` cannot set headers. Ensure this value can never be attached to a Sentry event; do not record SSE request URLs as breadcrumbs.
- Do not capture inference bodies, model inputs/outputs, artifact contents, or user-entered secrets. Preserve exception types and stack frames but redact free-text exception messages and remove frame locals.
- Keep Session Replay disabled. Do not enable console-log recording globally.
- Never return the Sentry read token or raw Sentry API response from the gateway.
- Add tests for the scrubber using fake credentials and fake model payloads.

### 5.7 Configuration and defaults

| Setting | Purpose | Default |
|---|---|---|
| `CDP_SENTRY_DSN` | Gateway Sentry DSN; also the UI DSN fallback | unset = disabled |
| `CDP_SENTRY_UI_DSN` | Optional separate public UI DSN | `CDP_SENTRY_DSN` |
| `CDP_SENTRY_ENVIRONMENT` | UI and gateway environment label | `CDP_ENV` |
| `CDP_SENTRY_TRACES_SAMPLE_RATE` | Gateway performance trace sampling | `0` (errors only) |
| `CDP_SENTRY_UI_TRACES_SAMPLE_RATE` | Browser trace sampling | `0` (errors only) |
| `CDP_SENTRY_READ_TOKEN` | Secret, read-only Sentry API credential for the Logs-page view | unset = issue list disabled |
| `CDP_SENTRY_ORG_SLUG` | Sentry organization slug used by the gateway API client | unset |
| `CDP_SENTRY_PROJECT_SLUG` | Sentry project slug used by the gateway API client | unset |
| `CDP_SENTRY_API_BASE_URL` | HTTPS Sentry API base URL for SaaS/regional/self-hosted deployments | `https://sentry.io` |

The frontend DSN is a public client identifier, not a server secret. The gateway DSN and read token are server-side deployment settings. Never put a Sentry auth token in frontend configuration. The read token should have only the minimum permissions needed to list the configured project’s issues/events. Use the same Sentry project for frontend and gateway events if you want both in the one-project Logs-page issue list.

The UI obtains its public DSN/environment/sample rate from the unauthenticated `GET /api/public-config` endpoint at runtime. That endpoint returns no read token or other server secrets, so the existing Docker image can use deployment environment variables without rebuilding the UI. If the gateway is offline during UI startup, the UI continues without Sentry for that browser session.

The Sentry SDK should be a no-op when its DSN is absent. Performance tracing remains off unless explicitly enabled with a sample rate. Error reporting is opt-in by configuring a DSN. The Sentry issue list requires its separate read-only API configuration.

## 6. Dependency and file impact (proposed)

Likely implementation touch points:

- `apps/ui/package.json` and lockfile — Sentry React SDK.
- `apps/ui/src/main.tsx` and `src/observability/sentry.ts` — runtime config bootstrap, browser setup, privacy scrubber, and React error boundary.
- `apps/ui/src/api/client.ts` — request ID propagation.
- `apps/ui/src/pages/LogsPage.tsx` — Sentry subtab and issue list/empty/error states.
- `apps/gateway/pyproject.toml` and `deploy/Dockerfile` — optional Sentry SDK extra included by the production image.
- `apps/gateway/gateway/core/sentry.py`, `core/config.py`, and `main.py` — initialization, settings, exception capture, and event scrubbing.
- `apps/gateway/gateway/core/redis_store.py` and `state.py` — bounded Upstash outage/recovery reporting to Logs and Sentry.
- `apps/gateway/gateway/api/management.py` and `services/sentry_issues.py` — public UI bootstrap config and authenticated, cached, allowlisted issue-list proxy.
- Tests and deployment/API documentation — no-network coverage, read-token protection, Redis diagnosis, and setup instructions.

The exact dependency strategy (base dependency versus optional extra) should keep the default behavior clear: if an operator configures Sentry in the supported deployment, the required SDK must actually be present in that deployment.

## 7. Acceptance criteria

1. With no DSN configured, the app starts normally, sends no Sentry events, and all existing tests/builds continue to pass.
2. With Sentry configured, an unexpected React error appears in Sentry with a readable stack trace and release/environment labels.
3. An unexpected FastAPI exception appears in Sentry with its Python traceback, without changing the existing 500 response contract.
4. Expected 4xx, validation, auth, quota, and routine offline/degraded states do not create noisy Sentry issues.
5. The existing Logs page has a Sentry subtab showing recent issue summaries and links to the corresponding Sentry issues.
6. Sentry issue data is fetched only by the authenticated gateway. The API token is never sent to or exposed by the browser.
7. If Sentry API configuration is absent or Sentry is unreachable, the Sentry subtab explains the state and the existing log/audit views still work.
8. Issue-list results are allowlisted, briefly cached, and bounded; raw events and request/model payloads are not displayed or persisted locally.
9. A shared request ID is visible on UI API requests and gateway responses; trace propagation is limited to this app’s own endpoints when enabled.
10. Automated tests prove that API keys, runtime tokens, SSE `api_key` query values, fake model payloads, and the Sentry read token are absent from captured events and UI/API responses.
11. A Sentry outage or slow exporter/API does not affect UI availability, inference, relay handling, or gateway startup/shutdown.
12. Upstash failures produce one sanitized warning per outage in the app Logs and, when configured, Sentry; a recovery is logged and Cache status exposes the reason/timestamps.
13. Existing Live, History, Audit, and metrics behavior remains intact; history is accurately described as process-local.
14. Deployment documentation explains where to configure event-reporting DSNs, the separate issue-list read token, and optional performance sampling.

## 8. Verification plan

- Unit-test configuration-disabled behavior and event redaction without sending network traffic.
- Unit-test that the generic FastAPI exception handler reports one event while retaining its current response body/status.
- Test the gateway Sentry API client with a mocked HTTP transport: valid issue summaries, timeouts, non-2xx responses, cache behavior, maximum limits, and no token leakage.
- UI-test the root error boundary, request-error filtering, Sentry issue list, empty/unconfigured state, and upstream-error state with mocked services.
- Run the existing Python test suite, UI typecheck, and production UI build.
- Perform a manual smoke test with a temporary Sentry project: trigger a development-only UI error and a development-only gateway exception, verify both arrive with expected labels, and verify the Logs page can fetch issue summaries. Then remove the triggers. No production test endpoint should be added.

## 9. Rollout

1. Add SDKs and safe exception reporting with tracing and Session Replay disabled.
2. Add the authenticated Sentry Issues API proxy and Logs-page subtab; verify token handling and upstream failure states.
3. Verify privacy filtering and deployment configuration in a non-production Sentry project.
4. Enable production DSNs and the read-only issue-list token. Keep trace sampling at zero initially; raise it only if performance visibility is needed and event volume is understood.
5. Consider a separate follow-up spec before enabling agent-side reporting, Session Replay, or durable centralized log shipping.
