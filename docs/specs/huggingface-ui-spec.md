# Spec: Hugging Face models from the UI

Status: partially implemented (see section 15)
Branch: `arena/cec07287-deploy-ml-models-for-free-cola`

## 1. Goal

A user can deploy a common Hugging Face model from the UI without writing code or
running a notebook cell. The user pastes a model ID (or picks one from search), picks
a task, and the system downloads, loads, warms, and serves it.

"All types" means every task that loads through the standard `transformers.pipeline`
path (see section 5). It does not mean every model on the Hub. Some models cannot run
on a free Colab VM under any design.

## 2. Non-goals

- Running GGUF files (these need llama.cpp, not transformers).
- Diffusion and image-generation pipelines (diffusers). These stay on the existing
  `python_function` path with a user-supplied `.py` file.
- Training or fine-tuning anything.
- Models that need custom Python code (`trust_remote_code`) by default. See section 9.
- Models larger than the VM's disk or GPU memory. The UI warns; it does not try.

## 3. Current state (what exists today)

| Area | Current behaviour | Where |
| --- | --- | --- |
| Hub URI | `hf://<repo>` is accepted, but only the **first** path segment is kept as the repo ID. `hf://Qwen/Qwen2.5-...` resolves to repo `Qwen`, which is wrong. Only single-segment legacy IDs such as `distilgpt2` work. | `agent/colabdeploy/adapters/huggingface.py` (`load`) |
| Local snapshot | A directory path is loaded as a local model, which works for owner-prefixed models after a manual download. | same, `load` |
| Task | Taken from `extra.task`, then `entrypoint`, then `text-classification`. | same, `load` |
| Adapter choice | With `auto`, an entrypoint wins over the `hf://` prefix and routes to `python_function`. An explicit `huggingface` adapter avoids this. | `apps/gateway/gateway/services/deployments.py` (`detect_adapter`) |
| Schemas | Built automatically for text, image, and question-answering tasks, but only when the adapter is loaded. The form does not use them. | `huggingface.py` (`describe`) |
| Form | Artifact URI, name, framework hint, adapter, entrypoint, checksum, tags, input and output schema, warm-up sample. No task picker, no Hub search. | `apps/ui/src/pages/ModelsPage.tsx` |
| Downloads | The agent resolves artifacts on the VM and caches by content hash. Hub downloads with a filename use `hf_hub_download`. A repo with no filename is a placeholder that `pipeline()` downloads on load. | `agent/colabdeploy/artifacts.py` |
| Agent install | Colab installs the agent from GitHub `main`. Changes on other branches do not reach a Colab runtime until merged. | `README.md`, `agent/README.md` |

## 4. Users and flows

**Flow A, single-segment model (no change needed):** paste `hf://distilgpt2`, pick the
task, deploy.

**Flow B, owner-prefixed model (the main change):**
1. Open Models, choose the Hugging Face tab.
2. Paste `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4`, or search and select it.
3. The UI shows size, format, task, and warnings (section 7).
4. The UI pre-fills the task, adapter, schemas, and warm-up sample. The user can edit them.
5. Deploy. The runtime downloads, loads, and warms the model. The UI shows progress.
6. The Playground can send a test request.

**Flow C, advanced:** a user who knows the model can type the task and schema by hand.
Flow C is always available.

## 5. Supported tasks

Tasks are grouped by what the adapter can already feed to a pipeline.

| Phase | Task (pipeline name) | Input | Output the UI shows |
| --- | --- | --- | --- |
| 1 | `text-classification` | `text` | label, score |
| 1 | `token-classification` (NER) | `text` | list of entities |
| 1 | `zero-shot-classification` | `text`, `candidate_labels` | labels with scores |
| 1 | `text-generation` | `text` | `text` |
| 1 | `summarization` | `text` | `text` |
| 1 | `translation` | `text` (language pair may be set per model) | `text` |
| 1 | `question-answering` | `question`, `context` | answer, score |
| 1 | `fill-mask` | `text` containing the mask token | top candidates |
| 1 | `feature-extraction` | `text` | vector (warn: can be large) |
| 1 | `image-classification` | image (base64) | label, score |
| 2 | `object-detection`, `image-segmentation` | image | boxes or masks |
| 2 | `automatic-speech-recognition` | audio (base64) | `text` (needs a new input type) |

Phase 1 is in scope for the first release. Phase 2 needs new input handling and is out of
scope until phase 1 is done.

Each task has a default schema and a sample input, so the user rarely types JSON.

## 6. Requirements

### 6.1 Model input (UI)

- **R1.** The Hugging Face tab accepts `owner/name` and single-segment IDs, with an
  optional branch or revision.
- **R2.** A search box queries the Hub for models and shows results with name, task,
  format, and download size.
- **R3.** A task dropdown lists the phase 1 tasks. Choosing a task fills the
  entrypoint, schema, and warm-up sample.
- **R4.** The adapter is fixed to `huggingface` for this tab. The user cannot
  accidentally select `auto`, which would route to `python_function`.
- **R5.** An "advanced" section exposes revision, device (`auto`, `cpu`, `cuda`), and
  dtype. These map to the existing `extra` fields.

### 6.2 Resolution and download (agent and gateway)

- **R6.** The gateway passes the full `owner/name` to the agent. The adapter uses the
  full string as the repo ID. The parser keeps everything before an optional file path,
  and a file path is only split off when the user gives one explicitly.
- **R7.** The agent downloads the model snapshot to a cache directory on the VM and
  reuses it on later loads, keyed by repo, revision, and file list. A deploy must not
  re-download an unchanged model.
- **R8.** Downloads report progress in the Logs page and in the model's status (for
  example `downloading 40%`).
- **R9.** Gated models use a Hugging Face token. The token is stored by the control
  plane, never shown in the UI after saving, and never written to logs.
- **R10.** The existing `file://` and `/content/models/...` paths keep working, so
  current deployments are unaffected.

### 6.3 Validation before deploy

- **R11.** The gateway checks the repo before accepting the deploy: the repo exists,
  the format is supported (safetensors, PyTorch `.bin`, or a repo `transformers` can
  load), and the total download size is shown.
- **R12.** Blocking errors stop the deploy and explain why. Examples: GGUF-only repo,
  repo not found, gated without a token, size above the runtime's disk limit.
- **R13.** Warnings do not stop the deploy. Examples: chat or instruction model
  (plain-prompt input), reasoning model (long outputs), quantised model that needs extra
  packages.
- **R14.** Quantised models (GPTQ, AWQ) show which packages the runtime needs. The agent
  reports a missing package as a clear load error, not a generic one.

### 6.4 Runtime and serving

- **R15.** The agent loads the model with the task set from the deployment, not from a
  default. An unknown task fails with a message that lists the valid tasks.
- **R16.** Warm-up runs on attach, as today, so the first real request is not cold.
- **R17.** The Playground shows the same output envelope that the API returns, so users
  can see `output.text`, `output.label`, and so on.

### 6.5 Errors and status

- **R18.** Every load failure appears in the model's status with a short reason and a
  "what to do" hint. The raw traceback stays in Logs.
- **R19.** The Runtimes page lists loaded HF models with their task and size.

## 7. Warnings and blocking rules (summary)

| Condition | Result |
| --- | --- |
| GGUF-only repo | Block, suggest a different format or the custom path |
| Repo not found or private without token | Block |
| Diffusion pipeline | Block, point to `python_function` |
| Requires custom code (`trust_remote_code`) | Block unless the user opts in (section 9) |
| Total size larger than the runtime disk | Block |
| Chat or instruction model | Warn: input is sent as plain text, not a chat |
| Reasoning model | Warn: outputs are long and slower |
| GPTQ or AWQ | Warn: extra packages needed; show the list |
| Size over 2 GB | Warn: first load may take several minutes |

## 8. Backend changes (summary)

- Adapter: parse full repo IDs; accept a task from the deployment; report download
  progress; load gated repos with a token.
- Gateway: detect HF repos before the entrypoint rule; add a validation endpoint that
  checks the repo and returns size, format, and task hints; store the HF token.
- Agent: snapshot cache on the VM; progress reporting; missing-package detection.
- UI: Hugging Face tab with search, task dropdown, pre-filled schemas, validation panel.

Exact API routes and field names are for the implementation plan, not this spec.

## 9. Security and trust

- `trust_remote_code` is **off** by default. Enabling it runs code from the repo on the
  runtime, so it needs an explicit checkbox and a warning.
- The HF token is stored server-side and is never returned by any API after saving.
- Repos are not executed until they pass validation (R11).
- The UI must not show the token in logs, error messages, or the Playground.

## 10. Testing

- **Unit:** repo-ID parsing (single segment, owner/name, owner/name with file, local
  path); task resolution order; adapter choice for HF URIs.
- **Adapter:** each phase 1 task loads and returns the expected envelope, using a tiny
  test model in CI (not a 1.5B model).
- **Validation:** each blocking rule in section 7 has a test.
- **UI:** the task dropdown fills the schema; the adapter field cannot be `auto` on the
  Hugging Face tab; the validation panel shows warnings and blockers.
- **End to end:** deploy `distilgpt2` and one owner-prefixed model on a real Colab
  runtime; the Playground returns output for both.

## 11. Acceptance criteria

1. A user can deploy `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4` from the UI with no
   notebook cell, given the packages are available on the runtime.
2. A user can deploy a phase 1 task model from search results with the task and schema
   filled in automatically.
3. An unsupported model is blocked before deploy, with a reason.
4. A second deploy of an unchanged model does not re-download it.
5. Existing deployments (`file://`, `store://`, `/content/models/...`, sentiment and
   distilgpt2 demos) keep working unchanged.
6. All existing tests still pass.

## 12. Rollout

1. Merge the adapter and gateway changes to `main` first, because the Colab agent
   installs from `main`.
2. Ship the UI changes behind no flag; the Hugging Face tab already exists.
3. Keep the Colab download cell documented as a fallback.

## 13. Open questions

1. Where should the download cache live on the VM, and what is the size limit per
   runtime?
2. Should the gateway check the Hub (needs outbound access to huggingface.co from the
   control plane), or should only the agent check?
3. Should missing packages (`auto-gptq`, `optimum`) be installed automatically, or only
   reported? Automatic installs are convenient but change the runtime environment.
4. Who may enable `trust_remote_code`: any user, or only admin keys?
5. Is translation in phase 1 if the model needs a language pair the user must supply?
6. Do we keep `auto` as the default adapter for non-Hub artifacts?

## 14. Later phases (after the goal in section 1 is complete)

These items are excluded from the first release. Effort is relative to the first release.

| Item | Approach | Effort | Notes |
| --- | --- | --- | --- |
| Custom-code models (`trust_remote_code`) | Add an opt-in checkbox to the Hugging Face tab. The adapter already reads this setting from the deployment's `extra` field. | Small | Requires the warning and admin-only rule from section 9. |
| GGUF files | New adapter that loads GGUF through a llama.cpp binding. | Moderate | Adds a dependency. The gateway, UI, and other adapters stay the same. |
| Diffusion models (diffusers) | New adapter for diffusers pipelines. Until then, use the custom `python_function` path with a `.py` file. | Moderate | Image input and output handling is needed. |
| Training or fine-tuning | Out of scope for this product. | Not planned | Would be a separate system. |
| Models larger than the runtime's disk or GPU | No code change fixes this. Use a bigger runtime or a smaller or quantised model. | Hardware-bound | The validation in section 6.3 already blocks these before deploy. |

## 15. Implementation status

Implemented on this branch:

| Requirement | Status | Where |
| --- | --- | --- |
| R6 full `owner/name` repo IDs | Done | `colabdeploy_shared/hf.py` (`parse_hf_uri`), used by the adapter and the artifact resolver |
| R3 task dropdown fills entrypoint, schemas, warm-up | Done (10 presets, one per phase 1 task) | `ModelsPage.tsx` (`HF_TASKS`) |
| R4 adapter fixed to `huggingface` for Hub URIs | Done | UI effect plus the adapter is disabled for `hf://`; gateway routes `hf://` to huggingface |
| R11–R13 name checks and blockers/warnings | Done, name-based only | `check_hf_uri`, `POST /api/models/hf-check`, block at registration |
| R7 no re-download of an unchanged model | Relies on the Hugging Face cache on the VM | `transformers` cache; not a separate cache |
| R17 Playground shows the envelope | Already done before this work | Playground `body` block |
| R18/R19 load errors and loaded models | Partly; existing status and Logs pages | unchanged |

Not implemented yet (deferred, with reasons):

- **R2 Hub search.** Needs the browser to call huggingface.co. Not built or tested here.
- **R11 existence and size check against the Hub.** The control plane would need outbound
  access to huggingface.co. The name checks and the agent's load-time error cover most of it.
- **R5 revision, device and dtype fields.** The form does not send the `extra` settings yet.
- **R8 download progress.** Needs a progress hook in the agent.
- **R9 Hugging Face token for gated models.** Not built. The agent would read a token from
  its own environment; the gateway should never store it.
- **R14 automatic package install.** Only warned about. See open question 3.
- **Section 14 later phases** stay out of scope, as agreed.

Verification:

- `pytest`: 304 passed, 1 skipped. New tests are in `tests/test_hf.py` (28 tests): URI
  parsing, blocking and warning rules, artifact placeholder and file download, adapter
  repo loading (with a stubbed `transformers`), and the gateway endpoint and registration
  blocking.
- `ruff check`: clean on the Python packages and tests.
- UI: `tsc` and `vite build` succeed. ESLint is not installed in this project, so the
  lint script cannot run. I have not tested the UI in a browser.
