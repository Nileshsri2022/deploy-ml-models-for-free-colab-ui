# ColabDeploy demo video plan

**Target length:** 2–3 minutes. **Format:** one continuous screen recording with terminal and browser captures; no real API keys or personal data on screen.

## Before recording

1. Build the UI and start the local end-to-end demo using the commands in the README.
2. Keep the terminal visible with the generated demo API key, but use only this disposable local key in the recording.
3. Open the dashboard and Playground in a browser window at `http://127.0.0.1:8000/`.
4. Prepare a separate terminal for the sample `curl` request below.

## Shot list and narration

### 0:00–0:20 — The problem

**Show:** README architecture diagram, then the local demo terminal.

**Say:** “A Colab notebook can run a model, but it doesn’t give you a stable API or durable state. ColabDeploy connects a notebook runtime to a small gateway over an outbound tunnel.”

### 0:20–0:50 — One-command demo

**Show:** Run `uv run python scripts/e2e_local.py`; highlight the gateway URL, API key, agent attachment, and two deployed demo models.

**Say:** “This local demo runs the gateway and a real agent together, so we can try the end-to-end flow without setting up Colab first.”

### 0:50–1:25 — Predict and cache

**Show:** Submit a sentence for `demo-sentiment` in Playground, submit it again, then run:

```bash
curl -X POST http://127.0.0.1:8000/v1/models/demo-sentiment/predict \
  -H "Authorization: Bearer $CDP_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"inputs":{"text":"this is a great little demo"}}'
```

**Say:** “The first request reaches the runtime. Repeating it returns the cached answer from the gateway, with the same response shape.”

### 1:25–1:55 — Runtime resilience

**Show:** Runtime status in the UI. For a reliable screen demo, use the existing end-to-end failover test or disconnect one of two already-prepared runtimes and make a fresh prediction.

**Say:** “When a runtime drops, the gateway can retry a fresh request on another ready replica. If no runtime can answer, the endpoint follows its configured cache, queue, or fail policy.”

### 1:55–2:20 — The real Colab path

**Show:** Connect page, then the notebook launch cell with tokens hidden. Do not expose a live runtime token.

**Say:** “For a real model, deploy the gateway at a public URL, mint a runtime token, and run the notebook cell. Your callers use the same prediction API.”

### 2:20–2:35 — Close

**Show:** README links to setup, API reference, and deployment guide.

**Say:** “Start with the local demo, then connect a public gateway and your own Colab runtime. See the README and deployment guide for the setup.”

## Recording checklist

- Hide or replace API keys, runtime tokens, private URLs, and browser notifications.
- Show one successful prediction and one repeated cache hit.
- Avoid implying free Colab sessions are permanent or that keep-alive defeats the session cap.
- Caption the video and link the repository and deployment documentation in its description.
- Do not commit raw screen recordings or large exports to Git; publish the finished video externally and add its URL to the README.
