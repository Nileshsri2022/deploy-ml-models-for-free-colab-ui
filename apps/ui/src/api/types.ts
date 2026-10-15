/**
 * Types mirroring `packages/shared-schema/colabdeploy_shared/schemas.py`.
 *
 * Hand-written rather than generated: the shared schema is the source of truth
 * and these mirror the fields the UI actually renders. Field names match the
 * JSON wire format exactly (`model_dump(mode="json")`), so enums arrive as
 * plain strings.
 */

export type RuntimeState =
  | 'unknown'
  | 'registered'
  | 'connecting'
  | 'online'
  | 'degraded'
  | 'stale'
  | 'offline'

export type DeploymentState = 'pending' | 'loading' | 'ready' | 'failed' | 'unloaded'

export type DegradationMode = 'serve_cached' | 'queue' | 'fail'

export type CacheStatus = 'HIT' | 'MISS' | 'STALE' | 'BYPASS'

export interface RuntimeCapabilities {
  gpu: boolean
  gpu_name: string | null
  gpu_memory_mb: number | null
  cpu_count: number | null
  ram_mb: number | null
  cuda_version: string | null
  python_version: string | null
  agent_version: string | null
  disk_free_mb: number | null
  adapters: string[]
}

export interface DeploymentInfo {
  id: string
  endpoint_id: string
  model_id: string
  model_version: number
  runtime_id: string | null
  state: DeploymentState
  error: string | null
  warm: boolean
  warmup_ms: number | null
  deployed_at: string
  updated_at: string
}

export interface RuntimeInfo {
  id: string
  label: string
  state: RuntimeState
  capabilities: RuntimeCapabilities
  colab_url: string | null
  tunnel_kind: string | null
  tunnel_url: string | null
  last_heartbeat_at: string | null
  session_started_at: string | null
  session_max_hours: number
  loaded_models: string[]
  deployments: DeploymentInfo[]
  error: string | null
  registered_at: string
  seconds_until_session_end: number | null
  seconds_since_heartbeat: number | null
  // added by GET /api/runtimes
  token_prefix?: string | null
  token_expires_at?: string | null
  connected?: boolean
  revoked?: boolean
  relaunch_url?: string
  desired_state?: DesiredState
}

export interface RuntimeTokenInfo {
  runtime_id: string
  token: string
  label: string
  expires_at: string | null
  created_at: string
  notebook_cell?: string
  colab_relaunch_url?: string
}

export interface ModelSchema {
  fields: Record<string, { type?: string; description?: string; enum?: string[] } | unknown>
  required: string[]
  example: Record<string, unknown> | null
  description: string | null
}

export interface ArtifactSpec {
  uri: string
  checksum: string | null
  size_bytes: number | null
  filename: string | null
}

export interface ModelInfo {
  id: string
  name: string
  version: number
  framework: string
  adapter: string
  artifact: ArtifactSpec
  schema_in: ModelSchema
  schema_out: ModelSchema
  warmup_sample: Record<string, unknown> | null
  entrypoint: string | null
  tags: string[]
  status: string
  owner_id: string
  created_at: string
  notes: string | null
  // added by the model view
  all_versions?: number[]
  endpoints?: string[]
}

export interface EndpointInfo {
  id: string
  model_id: string
  model_version: number
  route: string
  url: string | null
  visibility: 'public' | 'private'
  cache_ttl_s: number
  stale_while_revalidate_s: number
  degradation_mode: DegradationMode
  timeout_ms: number
  rate_limit_rpm: number
  max_body_bytes: number
  status: string
  created_at: string
  model_name: string | null
  adapter: string | null
  serving_runtime_id: string | null
  deployment_state: DeploymentState | null
  schema_in: ModelSchema
  schema_out: ModelSchema
  // added by the endpoint view
  path?: string
  runtime_state?: RuntimeState | null
  live?: boolean
  curl?: string
}

export interface DesiredDeployment {
  deployment_id: string
  endpoint_id: string
  model_id: string
  model_version: number
  adapter: string
  artifact: ArtifactSpec
  entrypoint: string | null
  schema_in: ModelSchema
  schema_out: ModelSchema
  warmup_sample: Record<string, unknown> | null
  state: DeploymentState
}

export interface DesiredState {
  runtime_id: string
  generated_at: string
  revision: number
  deployments: DesiredDeployment[]
  warmup_top_k: number
  settings: Record<string, unknown>
}

export interface CacheStats {
  backend: string
  entries: number
  hits: number
  misses: number
  stale_hits: number
  evictions: number
  hit_ratio: number
  size_bytes: number
  max_entries: number
  pinned: number
}

export interface CacheEntryInfo {
  key: string
  endpoint_id: string
  model_version: number
  created_at: number
  expires_at: number
  stale_at: number
  pinned: boolean
  hits: number
  size_bytes: number
  preview: { json: string } | null
}

export interface StatusResponse {
  healthy: boolean
  runtimes_total: number
  runtimes_online: number
  runtimes_degraded: number
  models: number
  endpoints: number
  endpoints_serving: number
  deployments_ready: number
  jobs_queued: number
  cache: CacheStats
  requests_total: number
  requests_cached: number
  uptime_s: number
  version: string
  runtimes: RuntimeInfo[]
}

export interface InferenceResponse {
  model_id: string
  endpoint_id: string
  model_version: number
  output: unknown
  cache: CacheStatus
  latency_ms: number
  runtime_id: string | null
  request_id: string
  job_id: string | null
  stale: boolean
  served_at: string
  error: { code: string; message: string; detail?: Record<string, unknown> | null } | null
}

export interface JobInfo {
  id: string
  endpoint_id: string
  model_id: string
  model_version: number
  inputs: Record<string, unknown>
  state: 'queued' | 'running' | 'succeeded' | 'failed' | 'expired'
  attempts: number
  result: unknown
  error: string | null
  webhook_url: string | null
  created_at: string
  updated_at: string
  done_at: string | null
}

export interface LogLine {
  ts: string
  runtime_id: string | null
  level: string
  source: string
  message: string
  data: Record<string, unknown> | null
}

export interface ApiKeyInfo {
  id: string
  label: string
  prefix: string
  scopes: string[]
  created_at: string
  expires_at: string | null
  last_used_at: string | null
  revoked: boolean
  token?: string | null
}

export interface BootstrapInfo {
  gateway_url: string
  relay_ws_url: string
  colab_relaunch_url: string
  notebook_cell: string
  protocol_version: number
  agent_version: string
  installed: { api_keys: number; runtimes: number; models: number; endpoints: number }
}

export interface DeployResult {
  deployment: DeploymentInfo
  endpoint: EndpointInfo
  model: ModelInfo
  runtime_id: string | null
  runtime_online: boolean
  pushed_to_runtime: boolean
  curl: string
  message: string
}

export interface Telemetry {
  summary: {
    requests_total: number
    requests_cached: number
    requests_failed: number
    requests_stale: number
    requests_queued: number
    cache_hit_ratio: number
    latency_p50_ms: number
    latency_p95_ms: number
    latency_p99_ms: number
    uptime_s: number
  }
  relay: {
    connected: { runtime_id: string; agent_version: string; connected_s: number; idle_s: number }[]
    pending_requests: number
    connections: number
    disconnects: number
    requests: number
    timeouts: number
    errors: number
  }
  cache: CacheStats
  sse_subscribers: number
}
