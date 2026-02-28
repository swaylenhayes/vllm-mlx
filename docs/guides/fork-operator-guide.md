# Fork Operator Guide

This guide is the practical runbook for running the `swaylenhayes/vllm-mlx`
fork on Apple Silicon.

Use it for:
- choosing between the published install and a patched checkout
- managing multiple branches or project-specific test environments
- starting text and multimodal servers with the right profile
- understanding which request settings are controlled by the server vs the client
- validating frontend chat apps such as SemaChat against the backend

## 1) Pick the right runtime surface

There are three valid ways to run this fork. Do not treat them as interchangeable.

### A. Published tool install

Use this when you want the latest pushed GitHub state and do not need local
uncommitted patches.

```bash
uv tool install --upgrade git+https://github.com/swaylenhayes/vllm-mlx.git
```

This gives you a global `vllm-mlx` command.

### B. Patched checkout

Use this when testing local fixes that are not yet published, or when the
working tree is ahead of the installed global binary.

Canonical example:

```bash
cd /Users/swaylen/dev/vllm-mlx-fork/vllm-mlx
uv venv .venv
. .venv/bin/activate
uv pip install -e ".[dev]"
```

When running from this checkout, prefer the explicit module entrypoint:

```bash
/Users/swaylen/dev/vllm-mlx-fork/vllm-mlx/.venv/bin/python -m vllm_mlx.cli
```

This is the current canonical path for evaluating local backend patches.

### C. Isolated project worktree

Use this when you want a separate branch and venv for an experiment without
dirtying the main checkout.

Example:

```bash
cd /Users/swaylen/dev/vllm-mlx-fork/vllm-mlx
git fetch --all --prune
git worktree add ../vllm-mlx-qwen-vl-compare -b qwen-vl-compare origin/phase/p1
cd ../vllm-mlx-qwen-vl-compare
uv venv .venv
. .venv/bin/activate
uv pip install -e ".[dev]"
```

Recommended operator rule:
- published installs are for released or pushed states
- patched checkout is for current backend work
- separate worktrees are for branch-specific experiments

## 2) Update or switch code safely

### Refresh the main checkout

```bash
cd /Users/swaylen/dev/vllm-mlx-fork/vllm-mlx
git fetch --all --prune
git switch phase/p1
git pull --ff-only
```

### Switch to a project branch

```bash
cd /Users/swaylen/dev/vllm-mlx-fork/vllm-mlx
git fetch --all --prune
git switch <branch-name>
```

If the branch changes dependencies, reinstall into the local venv:

```bash
uv pip install -e ".[dev]"
```

### Verify which binary you are actually using

```bash
which vllm-mlx
python -c "import vllm_mlx, sys; print(sys.executable)"
```

Operator rule:
- if you are evaluating a local patch, do not trust the global `vllm-mlx`
  binary unless you have explicitly refreshed that install after the patch
- record the runner path in every benchmark or test note

## 3) Download models

### Download through the published tool

```bash
vllm-mlx download mlx-community/Qwen3-4B-Instruct-2507-4bit
```

### Download through the patched checkout

```bash
cd /Users/swaylen/dev/vllm-mlx-fork/vllm-mlx
/Users/swaylen/dev/vllm-mlx-fork/vllm-mlx/.venv/bin/python -m vllm_mlx.cli download \
  mlx-community/Qwen3-4B-Instruct-2507-4bit
```

Useful controls:
- `--download-timeout`
- `--download-retries`
- `--offline`

Offline check example:

```bash
vllm-mlx serve mlx-community/some-model --localhost --offline
```

## 4) Choose a serve profile

The most important distinction is:
- text-only vs multimodal
- simple runtime vs continuous batching
- thinking behavior vs reasoning parsing

Important:
- `--reasoning-parser` is not the same as turning thinking on or off
- `enable_thinking` is a request field, not a serve flag
- for fair benchmarks, keep the serve flags identical across models and vary
  only the request payload

Patched-checkout shortcut:

```bash
cd /Users/swaylen/dev/vllm-mlx-fork/vllm-mlx
scripts/serve_profile.sh text-default mlx-community/Qwen3-4B-Instruct-2507-4bit
```

The launcher always uses the checkout's `.venv` unless `VLLM_MLX_PYTHON` is
explicitly overridden.

### Profile A: Text local dev default

Good for daily text usage and straightforward local app integration.

```bash
cd /Users/swaylen/dev/vllm-mlx-fork/vllm-mlx
/Users/swaylen/dev/vllm-mlx-fork/vllm-mlx/.venv/bin/python -m vllm_mlx.cli serve \
  mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --localhost --port 8000 \
  --runtime-mode simple \
  --cache-strategy auto \
  --default-temperature 0.7 \
  --default-top-p 0.9
```

### Profile B: Text deterministic diagnostics

Use for bug repro, fairness checks, and response-shape comparisons.

```bash
cd /Users/swaylen/dev/vllm-mlx-fork/vllm-mlx
/Users/swaylen/dev/vllm-mlx-fork/vllm-mlx/.venv/bin/python -m vllm_mlx.cli serve \
  mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --localhost --port 8000 \
  --deterministic
```

### Profile C: Text tool-calling

```bash
cd /Users/swaylen/dev/vllm-mlx-fork/vllm-mlx
/Users/swaylen/dev/vllm-mlx-fork/vllm-mlx/.venv/bin/python -m vllm_mlx.cli serve \
  mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --localhost --port 8000 \
  --runtime-mode auto \
  --cache-strategy auto \
  --enable-auto-tool-choice \
  --tool-call-parser auto
```

If clients must send the exact model id instead of `"default"`, add:

```bash
  --strict-model-id
```

### Profile D: Multimodal default

Use this for image or mixed image+text chat.

```bash
cd /Users/swaylen/dev/vllm-mlx-fork/vllm-mlx
/Users/swaylen/dev/vllm-mlx-fork/vllm-mlx/.venv/bin/python -m vllm_mlx.cli serve \
  mlx-community/Qwen3-VL-4B-Instruct-4bit \
  --localhost --port 8000 \
  --runtime-mode simple \
  --cache-strategy auto \
  --mllm \
  --default-temperature 0.0 \
  --default-top-p 1.0
```

### Profile E: Correctness-sensitive multimodal runs under concurrency

If you need batching but care about output agreement, keep the divergence monitor
on and serialize when needed.

```bash
cd /Users/swaylen/dev/vllm-mlx-fork/vllm-mlx
/Users/swaylen/dev/vllm-mlx-fork/vllm-mlx/.venv/bin/python -m vllm_mlx.cli serve \
  mlx-community/Qwen3-VL-4B-Instruct-4bit \
  --localhost --port 8000 \
  --runtime-mode auto \
  --cache-strategy auto \
  --mllm \
  --batch-divergence-monitor \
  --batch-divergence-threshold 0.95 \
  --batch-divergence-action serialize
```

### Profile F: Extraction and JSON-mode text runs

Use this profile for JSON-mode and extraction-sensitive evaluations.

```bash
cd /Users/swaylen/dev/vllm-mlx-fork/vllm-mlx
/Users/swaylen/dev/vllm-mlx-fork/vllm-mlx/.venv/bin/python -m vllm_mlx.cli serve \
  mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit \
  --localhost --port 8000 \
  --runtime-mode simple \
  --cache-strategy auto \
  --timeout 900 \
  --default-temperature 0.7 \
  --default-top-p 0.9
```

### Profile G: Thinking and non-thinking request control

There is no server flag called `--enable-thinking`.

Instead:
- load the model normally
- set `enable_thinking` per request

Example request payload:

```json
{
  "model": "default",
  "messages": [
    {"role": "user", "content": "Reply with one sentence."}
  ],
  "enable_thinking": false,
  "max_tokens": 32,
  "temperature": 0.0,
  "top_p": 1.0
}
```

Important:
- `enable_thinking` only matters on chat-template paths that honor it
- `--reasoning-parser` only affects how emitted reasoning is separated in the
  response
- for side-by-side model tests, keep the server profile fixed and toggle
  `enable_thinking` in the request

## 5) What the important flags actually do

### Runtime selection

- `--runtime-mode simple`: single-request execution path; best for controlled
  evaluation, local single-user usage, and many multimodal runs
- `--continuous-batching`: enables batching for concurrent users
- `--deterministic`: forces simple runtime, greedy sampling, and serialized
  tracked routes for reproducibility

### Model family selection

- `--mllm`: force multimodal loading
- no `--mllm`: normal text path unless metadata-based autodetection classifies
  the repo as multimodal

### Sampling controls

- `--default-temperature`: default temperature when the request omits it
- `--default-top-p`: default `top_p` when the request omits it
- request `temperature` and `top_p` override these defaults

### Time and memory controls

- `--timeout`: server-side request timeout ceiling
- `--effective-context-tokens`: operator-facing effective context contract
- `--memory-warn-threshold`, `--memory-limit-threshold`, `--memory-action`:
  guardrails for memory pressure

### Correctness and observability

- `--batch-divergence-monitor`: periodic agreement probe for batched runs
- `--batch-divergence-action serialize`: correctness-first fallback for tracked
  routes when divergence exceeds threshold
- `include_diagnostics` and `diagnostics_level` are request-level additions for
  richer client-visible metadata

### Tooling and reasoning

- `--enable-auto-tool-choice` and `--tool-call-parser`: enable structured tool
  calling
- `--reasoning-parser`: separates visible reasoning into a dedicated response
  field when the model emits it
- `--max-thinking-tokens`: caps reasoning emission when a reasoning parser is
  active

## 6) Verify the server quickly

```bash
curl -sS http://localhost:8000/health | jq
curl -sS http://localhost:8000/v1/models | jq
curl -sS http://localhost:8000/v1/capabilities | jq
```

If `--api-key` is enabled, include either `Authorization: Bearer ...` or
`x-api-key: ...`.

`/health` is always unauthenticated liveness.  
`/health/diagnostics` follows server auth policy.

## 7) Smoke test requests

### Text

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"default",
    "messages":[{"role":"user","content":"Reply with pong"}],
    "max_tokens":32
  }' | jq
```

### Text with non-thinking request override

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"default",
    "messages":[{"role":"user","content":"Reply with exactly one sentence."}],
    "enable_thinking": false,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 32
  }' | jq
```

### Multimodal

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"default",
    "messages":[
      {
        "role":"user",
        "content":[
          {"type":"text","text":"Describe the image in one short sentence."},
          {"type":"image_url","image_url":{"url":"/ABS/PATH/image.png"}}
        ]
      }
    ],
    "enable_thinking": false,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 32
  }' | jq
```

## 8) SemaChat and client-app guidance

SemaChat or similar apps should target the OpenAI-compatible API:

- base URL: `http://localhost:8000/v1`
- model: `"default"` unless the client insists on the exact id

Recommended interpretation of common frontend settings:

- `temperature`: maps directly to request `temperature`
- `top_p`: maps directly to request `top_p`
- `max_tokens`: maps directly to request `max_tokens`
- `frequency_penalty`: maps directly to request `frequency_penalty`
- `streaming on/off`: maps directly to request `stream`
- `system prompt`: should be serialized as a `system` message at the front of
  `messages`
- `timeout`: usually client-side request timeout; it should be set high enough
  to tolerate the backend's expected latency and server `--timeout`

Likely frontend-only unless the app explicitly serializes it:

- `max messages`: usually local history truncation or UI memory behavior, not a
  backend inference parameter

Current operator caveats:
- if your GUI does not expose `enable_thinking`, you cannot toggle it from the
  app even though the backend supports it
- image upload only works when the backend model is loaded in multimodal mode
- if the app does not emit OpenAI-style multimodal content blocks, image tests
  may fail even when the backend is correct

Suggested SemaChat validation order:
1. text non-stream
2. text stream
3. text with system prompt
4. multimodal single image
5. multimodal with `enable_thinking: false`
6. `frequency_penalty` and `max_tokens` confirmation

## 9) How to handle side-track test results

Use a simple rule:
- code changes live in the repo checkout
- evidence lives in docs, benchmark artifacts, or test reports
- result-only experiments do not need their own branch unless they reveal a code
  issue that needs a fix

For every reported run, capture:
- model id
- exact serve command
- exact runner path
- whether the run used the published tool or the patched checkout
- prompt or request payload
- frontend app name if applicable
- latency, token counts, visible reasoning leakage, and output-format compliance

Recommended intake policy:
- migration or Graphiti findings stay on the side track unless they expose a
  general server defect
- operator-run model comparisons feed the public model/profile matrix and client
  compatibility notes
- if a result cannot be tied to a specific checkout or install, treat it as
  provisional only

## 10) Current fork behavior notes

- `frequency_penalty` is supported in OpenAI-compatible requests and mapped to
  repetition-penalty behavior.
- Repetition detector policy is additive and conservative by default.
- Batch divergence controls are designed for observability first, then
  correctness fallback.
- Metadata-aware multimodal detection is important for repos whose names do not
  include `-VL`.

## 11) Troubleshooting

### The server works from the checkout but not from `vllm-mlx`

You are probably hitting an older global install.

Check:

```bash
which vllm-mlx
vllm-mlx --help >/dev/null
```

Then either:
- rerun from the checkout module entrypoint, or
- refresh the tool install with `uv tool install --upgrade ...`

### Server starts but client requests fail

Check:
- `GET /v1/capabilities` for expected features and runtime metadata
- auth mismatch if `--api-key` is enabled
- frontend request format, especially for images and system prompts
- whether the client is timing out before the server does

### Model load or download failures

Try:
- increasing `--download-timeout`
- increasing `--download-retries`
- pre-downloading with `vllm-mlx download ...`
- adding `--mllm` if the repo is genuinely multimodal and you are using an
  older install without metadata-based autodetection

### Throughput is good but outputs vary under concurrency

Use:
- `--batch-divergence-monitor`
- `--batch-divergence-action serialize` for correctness-sensitive runs
- `--runtime-mode simple` for the cleanest side-by-side measurements

### Memory pressure warnings

Tune:
- `--memory-warn-threshold`
- `--memory-limit-threshold`
- `--memory-action` (`warn`, `reduce-context`, `reject-new`)
- `--memory-monitor-interval`

## 12) Recommended docs to pair with this guide

- [OpenAI-Compatible Server](server.md)
- [Tool Calling](tool-calling.md)
- [Multimodal](multimodal.md)
- [Continuous Batching](continuous-batching.md)
- [Fork Benefits](../benchmarks/fork-benefits.md)
