# Fork Operator Guide

This guide is for users running the `swaylenhayes/vllm-mlx` fork locally on Apple Silicon.

Use this as the practical runbook for install, startup profiles, health checks, and troubleshooting.

## 1) Install

### Option A: Global CLI tool (recommended)

```bash
uv tool install --upgrade git+https://github.com/swaylenhayes/vllm-mlx.git
```

### Option B: Project/venv install

```bash
uv pip install --upgrade git+https://github.com/swaylenhayes/vllm-mlx.git
```

## 2) Choose a startup profile

### Profile A: Local dev default (safe)

Good for daily local usage and local client integration.

```bash
vllm-mlx serve mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --localhost --port 8000 \
  --runtime-mode auto --cache-strategy auto \
  --batch-divergence-monitor \
  --batch-divergence-threshold 0.95 \
  --batch-divergence-action warn
```

### Profile B: Correctness-sensitive VLM runs

Use serialized fallback when live batch composition can affect output agreement.

```bash
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-4bit \
  --localhost --port 8000 --mllm \
  --runtime-mode auto --cache-strategy auto \
  --batch-divergence-monitor \
  --batch-divergence-threshold 0.95 \
  --batch-divergence-action serialize
```

### Profile C: Tool-calling workflow

```bash
vllm-mlx serve mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --localhost --port 8000 \
  --runtime-mode auto --cache-strategy auto \
  --enable-auto-tool-choice --tool-call-parser auto
```

### Profile D: Deterministic diagnostics (repro runs)

Use this for repeatable bug reproduction and validation; throughput will be lower.

```bash
vllm-mlx serve mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --localhost --port 8000 \
  --deterministic
```

## 3) Verify server contract quickly

```bash
curl -sS http://localhost:8000/health | jq
curl -sS http://localhost:8000/v1/models | jq
curl -sS http://localhost:8000/v1/capabilities | jq
```

If `--api-key` is enabled, include either `Authorization: Bearer ...` or `x-api-key: ...`.

`/health` is always unauthenticated liveness.  
`/health/diagnostics` follows server auth policy.

## 4) Smoke test completion path

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"default",
    "messages":[{"role":"user","content":"Reply with pong"}],
    "max_tokens":32
  }' | jq
```

## 5) Download reliability controls

The fork includes resumable download behavior and offline mode for startup/download paths.

Serve-time controls:
- `--download-timeout`
- `--download-retries`
- `--offline`

Pre-download without starting server:

```bash
vllm-mlx download mlx-community/Qwen3-4B-Instruct-2507-4bit
```

Offline check example:

```bash
vllm-mlx serve mlx-community/some-model --localhost --offline
```

## 6) Current fork behavior notes

- Repetition detector is active in batched scheduler paths and is intentionally conservative.
- Current detector does not expose runtime flags yet (policy/controls work is planned separately).
- `frequency_penalty` is supported in OpenAI-compatible requests and mapped to repetition-penalty behavior.

## 7) Troubleshooting

### Server starts but client requests fail

Check:
- `GET /v1/capabilities` for expected features/modes.
- auth mismatch (`--api-key` enabled but client missing header).

### Model load/download failures

Try:
- increasing `--download-timeout`
- increasing `--download-retries`
- pre-downloading with `vllm-mlx download ...`

### Throughput is good but outputs vary under concurrency

Use:
- `--batch-divergence-monitor`
- `--batch-divergence-action serialize` for correctness-sensitive runs.

### Memory pressure warnings

Tune:
- `--memory-warn-threshold`
- `--memory-limit-threshold`
- `--memory-action` (`warn`, `reduce-context`, `reject-new`)
- `--memory-monitor-interval`

## 8) Recommended docs to pair with this guide

- [OpenAI-Compatible Server](server.md)
- [Tool Calling](tool-calling.md)
- [Multimodal](multimodal.md)
- [Continuous Batching](continuous-batching.md)
- [Fork Benefits](../benchmarks/fork-benefits.md)
