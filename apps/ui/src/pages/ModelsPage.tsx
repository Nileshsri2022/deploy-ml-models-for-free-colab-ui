import { useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { ApiError, api, cachedGet } from '../api/client'
import type { ModelInfo } from '../api/types'
import { DeployWizard } from '../components/DeployWizard'
import {
  Badge,
  Banner,
  Card,
  CodeBlock,
  Empty,
  KeyValue,
  Spinner,
  formatBytes,
  formatTime,
  parseJsonField,
  prettyJson,
} from '../components/ui'
import { useStore } from '../state/store'

type Tab = 'list' | 'register' | 'upload'

const URI_EXAMPLES = [
  ['Google Drive', 'gdrive://1AbCdefGhIjKlMnOpQrStUvWxYz012345'],
  ['Hugging Face', 'hf://org/repo/model.onnx'],
  ['HTTP(S)', 'https://example.com/model.pkl'],
  ['Already on the VM', 'file:///content/drive/MyDrive/model.joblib'],
  ['Uploaded via this UI', 'store://art_0123456789abcdef'],
]

/**
 * Model registry: register by URI, or upload an artifact through the browser.
 *
 * Registering only records *intent* — the bytes are fetched by the agent on the
 * VM (and cached there by content hash), so a 2 GB checkpoint never has to pass
 * through the gateway twice.
 */
export default function ModelsPage() {
  const { toast } = useStore()
  const { modelId } = useParams()
  const [tab, setTab] = useState<Tab>('list')
  const [models, setModels] = useState<ModelInfo[]>([])
  const [latest, setLatest] = useState<ModelInfo[]>([])
  const [stale, setStale] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(modelId ?? null)

  // register form
  const [name, setName] = useState('')
  const [uri, setUri] = useState('')
  const [framework, setFramework] = useState('auto')
  const [adapter, setAdapter] = useState('auto')
  const [checksum, setChecksum] = useState('')
  const [schemaIn, setSchemaIn] = useState('{\n  "fields": {},\n  "required": [],\n  "example": {}\n}')
  const [schemaOut, setSchemaOut] = useState('{\n  "fields": {},\n  "required": [],\n  "example": null\n}')
  const [warmup, setWarmup] = useState('')
  const [entrypoint, setEntrypoint] = useState('')
  const [tags, setTags] = useState('')
  const [busy, setBusy] = useState(false)

  // upload form
  const [file, setFile] = useState<File | null>(null)

  const load = useCallback(async () => {
    try {
      const result = await cachedGet<{ models: ModelInfo[]; latest: ModelInfo[] }>('/api/models')
      setModels(result.data.models ?? [])
      setLatest(result.data.latest ?? [])
      setStale(result.stale)
      setError(null)
    } catch (err) {
      setStale(true)
      setError(
        err instanceof ApiError && err.isAuthError
          ? 'An API key with the manage scope is required to read the registry.'
          : (err as Error).message,
      )
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
    const timer = setInterval(() => void load(), 10000)
    return () => clearInterval(timer)
  }, [load])

  const register = async () => {
    if (!name.trim() || !uri.trim()) {
      setError('name and artifact URI are both required')
      return
    }
    const parsedIn = parseJsonField(schemaIn)
    const parsedOut = parseJsonField(schemaOut)
    const parsedWarm = parseJsonField(warmup)
    if (parsedIn.error || parsedOut.error || parsedWarm.error) {
      setError(
        `invalid JSON: ${parsedIn.error ? `schema_in — ${parsedIn.error}` : ''}${
          parsedOut.error ? ` schema_out — ${parsedOut.error}` : ''
        }${parsedWarm.error ? ` warmup_sample — ${parsedWarm.error}` : ''}`,
      )
      return
    }
    setBusy(true)
    setError(null)
    try {
      const created = await api.post<ModelInfo>('/api/models', {
        name: name.trim(),
        framework: framework || 'auto',
        adapter: adapter || 'auto',
        artifact: {
          uri: uri.trim(),
          checksum: checksum.trim() || null,
          size_bytes: null,
          filename: null,
        },
        schema_in: parsedIn.value ?? {},
        schema_out: parsedOut.value ?? {},
        warmup_sample: (parsedWarm.value as Record<string, unknown> | undefined) ?? null,
        entrypoint: entrypoint.trim() || null,
        tags: tags
          .split(',')
          .map((t) => t.trim())
          .filter(Boolean),
      })
      toast('success', 'Model registered', `${created.name} v${created.version} — deploy it next`)
      setTab('list')
      setSelected(created.id)
      setUri('')
      await load()
    } catch (err) {
      setError((err as Error).message)
      toast('error', 'Register failed', (err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const upload = async () => {
    if (!file || !name.trim()) {
      setError('choose a file and give the model a name')
      return
    }
    const parsedIn = parseJsonField(schemaIn)
    const parsedOut = parseJsonField(schemaOut)
    const parsedWarm = parseJsonField(warmup)
    if (parsedIn.error || parsedOut.error || parsedWarm.error) {
      setError('invalid JSON in one of the schema/warm-up fields')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('name', name.trim())
      form.append('framework', framework || 'auto')
      form.append('adapter', adapter || 'auto')
      if (schemaIn.trim()) form.append('schema_in', schemaIn)
      if (schemaOut.trim()) form.append('schema_out', schemaOut)
      if (warmup.trim()) form.append('warmup_sample', warmup)
      if (entrypoint.trim()) form.append('entrypoint', entrypoint.trim())
      if (tags.trim()) form.append('tags', tags)
      const created = await api.upload<ModelInfo>('/api/models/upload', form)
      toast('success', 'Artifact uploaded', `${created.name} v${created.version} · ${created.artifact.uri}`)
      setTab('list')
      setSelected(created.id)
      setFile(null)
      await load()
    } catch (err) {
      setError((err as Error).message)
      toast('error', 'Upload failed', (err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const detail = models.find((m) => m.id === selected) ?? null
  const shown = latest.length ? latest : models

  return (
    <>
      <div className="page-head">
        <div>
          <h1 className="page-title">Models</h1>
          <p className="page-sub">
            The registry is framework-agnostic: sklearn, PyTorch, ONNX, XGBoost, Keras, HuggingFace or a
            plain Python function. The agent picks an adapter by probing the artifact, so "deploy any
            model" means one form, not one adapter per framework.
          </p>
        </div>
        <div className="row">
          <button type="button" onClick={() => void load()}>
            ↻ Refresh
          </button>
        </div>
      </div>

      {error && <Banner tone="red">{error}</Banner>}
      {stale && !error && (
        <Banner>Showing the last-known-good registry — the control plane did not answer.</Banner>
      )}

      <div className="tabs">
        <button type="button" className={tab === 'list' ? 'active' : ''} onClick={() => setTab('list')}>
          Registry ({models.length})
        </button>
        <button type="button" className={tab === 'register' ? 'active' : ''} onClick={() => setTab('register')}>
          Register by URI
        </button>
        <button type="button" className={tab === 'upload' ? 'active' : ''} onClick={() => setTab('upload')}>
          Upload artifact
        </button>
      </div>

      {tab === 'register' || tab === 'upload' ? (
        <Card
          title={tab === 'register' ? 'Register a model by URI' : 'Upload an artifact'}
          sub={
            tab === 'register'
              ? 'The agent fetches the bytes on the VM and caches them by content hash, so a big checkpoint is downloaded once per machine — not once per deploy.'
              : 'The gateway stores the file and hands the agent a store:// URI. Fine for small artifacts (pickles, ONNX graphs); for multi-GB weights prefer Drive or Hugging Face.'
          }
        >
          {tab === 'upload' ? (
            <label className="field">
              <span>Artifact file</span>
              <input
                type="file"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                style={{ padding: 6 }}
              />
              <div className="hint">
                {file ? `${file.name} · ${formatBytes(file.size)}` : '.pkl .joblib .pt .pth .onnx .h5 .pb .zip …'}
              </div>
            </label>
          ) : (
            <label className="field">
              <span>Artifact URI</span>
              <input
                value={uri}
                onChange={(event) => setUri(event.target.value)}
                placeholder="gdrive://1AbC… | hf://org/repo/model.onnx | https://… | file:///content/model.pkl"
                className="mono"
              />
              <div className="row" style={{ marginTop: 6 }}>
                {URI_EXAMPLES.map(([label, example]) => (
                  <button
                    key={label}
                    type="button"
                    className="small ghost"
                    onClick={() => setUri(example)}
                    title={example}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </label>
          )}

          <div className="grid cols-3">
            <label className="field">
              <span>Name</span>
              <input value={name} onChange={(event) => setName(event.target.value)} placeholder="sentiment-classifier" />
              <div className="hint">Re-registering the same name creates v2, v3, … — versions are how you roll back.</div>
            </label>
            <label className="field">
              <span>Framework hint</span>
              <input value={framework} onChange={(event) => setFramework(event.target.value)} placeholder="auto" />
              <div className="hint">auto = probe the artifact.</div>
            </label>
            <label className="field">
              <span>Adapter</span>
              <select value={adapter} onChange={(event) => setAdapter(event.target.value)}>
                {['auto', 'sklearn', 'pytorch', 'onnx', 'xgboost', 'keras', 'huggingface', 'python_function', 'echo'].map(
                  (kind) => (
                    <option key={kind} value={kind}>
                      {kind}
                    </option>
                  ),
                )}
              </select>
              <div className="hint">
                <code className="mono">python_function</code> + entrypoint is the escape hatch for anything custom.
              </div>
            </label>
          </div>

          <div className="grid cols-3">
            <label className="field">
              <span>Entrypoint (optional)</span>
              <input
                value={entrypoint}
                onChange={(event) => setEntrypoint(event.target.value)}
                placeholder="predict / module:callable / source snippet"
              />
            </label>
            <label className="field">
              <span>Checksum (optional)</span>
              <input
                value={checksum}
                onChange={(event) => setChecksum(event.target.value)}
                placeholder="sha256:…"
                className="mono"
              />
              <div className="hint">Verified after download; a mismatch fails the deploy loudly.</div>
            </label>
            <label className="field">
              <span>Tags</span>
              <input value={tags} onChange={(event) => setTags(event.target.value)} placeholder="nlp, v2, staging" />
            </label>
          </div>

          <div className="grid cols-3">
            <label className="field">
              <span>Input schema (JSON)</span>
              <textarea rows={8} value={schemaIn} onChange={(event) => setSchemaIn(event.target.value)} />
            </label>
            <label className="field">
              <span>Output schema (JSON)</span>
              <textarea rows={8} value={schemaOut} onChange={(event) => setSchemaOut(event.target.value)} />
            </label>
            <label className="field">
              <span>Warm-up sample (JSON)</span>
              <textarea
                rows={8}
                value={warmup}
                onChange={(event) => setWarmup(event.target.value)}
                placeholder='{"inputs": {"text": "hello"}}'
              />
              <div className="hint">
                Loaded and run once when a VM attaches, so the first real request is not cold.
              </div>
            </label>
          </div>

          <div className="row">
            <button
              type="button"
              className="primary"
              disabled={busy}
              onClick={() => void (tab === 'register' ? register() : upload())}
            >
              {busy ? <Spinner label={tab === 'register' ? 'registering…' : 'uploading…'} /> : tab === 'register' ? 'Register model' : 'Upload & register'}
            </button>
            <button type="button" className="ghost" onClick={() => setTab('list')}>
              Cancel
            </button>
          </div>
        </Card>
      ) : loading && models.length === 0 ? (
        <Card>
          <Spinner label="loading registry…" />
        </Card>
      ) : shown.length === 0 ? (
        <Card>
          <Empty
            title="No models registered"
            body="Register one by URI, or upload an artifact from this machine. Nothing is executed here — the gateway only records intent until you deploy."
            action={
              <button type="button" className="primary" onClick={() => setTab('register')}>
                Register your first model
              </button>
            }
          />
        </Card>
      ) : (
        <div className="grid cols-2">
          <Card title={`Latest versions (${shown.length})`} sub="One row per model name; older versions are kept for rollback.">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Adapter</th>
                  <th>Versions</th>
                  <th>Endpoints</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((model) => (
                  <tr
                    key={model.id}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setSelected(model.id)}
                  >
                    <td>
                      <strong>{model.name}</strong>
                      <div className="faint small mono">{model.artifact.uri.slice(0, 40)}</div>
                    </td>
                    <td>
                      <Badge tone="blue">{model.adapter}</Badge>
                      <div className="faint small">{model.framework}</div>
                    </td>
                    <td className="small">
                      v{model.version}
                      {(model.all_versions?.length ?? 0) > 1 ? (
                        <span className="faint"> of {model.all_versions?.length}</span>
                      ) : null}
                    </td>
                    <td className="small">
                      {model.endpoints?.length ? (
                        <Badge tone="green">{model.endpoints.length} endpoint(s)</Badge>
                      ) : (
                        <span className="faint">not deployed</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <div>
            {detail ? (
              <Card
                title={
                  <span>
                    {detail.name} <span className="faint">v{detail.version}</span>
                  </span>
                }
                sub={detail.notes ?? undefined}
                actions={
                  <div className="row">
                    <Badge tone={detail.status === 'active' ? 'green' : 'gray'}>{detail.status}</Badge>
                  </div>
                }
              >
                <KeyValue
                  items={[
                    ['id', <span className="mono">{detail.id}</span>],
                    ['adapter', detail.adapter],
                    ['framework', detail.framework],
                    ['artifact', <span className="mono">{detail.artifact.uri}</span>],
                    ['checksum', detail.artifact.checksum ? <span className="mono">{detail.artifact.checksum}</span> : null],
                    ['size', formatBytes(detail.artifact.size_bytes)],
                    ['entrypoint', detail.entrypoint],
                    ['tags', detail.tags.length ? detail.tags.join(', ') : null],
                    ['owner', detail.owner_id],
                    ['registered', formatTime(detail.created_at)],
                    ['versions', (detail.all_versions ?? []).map((v) => `v${v}`).join(', ') || `v${detail.version}`],
                  ]}
                />
                <div style={{ marginTop: 12 }}>
                  <CodeBlock label="input schema" code={prettyJson(detail.schema_in)} />
                </div>
                <div style={{ marginTop: 10 }}>
                  <CodeBlock label="output schema" code={prettyJson(detail.schema_out)} />
                </div>
                <div className="row" style={{ marginTop: 12 }}>
                  <button type="button" className="primary small" onClick={() => setTab('list')}>
                    ↓ deploy this model
                  </button>
                </div>
              </Card>
            ) : (
              <Card>
                <Empty title="Select a model" body="Pick a row to inspect its artifact, schemas and version history." />
              </Card>
            )}
          </div>
        </div>
      )}

      {tab === 'list' && models.length > 0 ? (
        <div style={{ marginTop: 14 }}>
          <DeployWizard models={models} initialModelId={selected ?? undefined} onDeployed={() => void load()} />
        </div>
      ) : null}
    </>
  )
}
