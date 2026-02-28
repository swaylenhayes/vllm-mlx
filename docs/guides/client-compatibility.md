# Client Compatibility

This page is the public operator-facing compatibility surface for external apps,
agents, and desktop clients that connect to `vllm-mlx`.

Use it for:
- choosing the right backend profile for a client
- seeing which targets are evidence-backed vs still in validation
- finding the shortest connection path for common OpenAI-compatible clients

## Core Rules

- Record the exact backend runner path, serve command, and model id for every claim.
- Treat `"default"` as safe only when `--strict-model-id` is off.
- Keep backend auth enabled for publishable examples unless the client only supports trusted local testing.
- Prefer the checkout launchers while validating new backend changes:
  - `scripts/serve_profile.sh`
  - `scripts/serve_client_profile.sh`

## Quick Start

For the shortest path to a client-ready backend, use the client launcher:

```bash
cd /Users/swaylen/dev/vllm-mlx-fork/vllm-mlx

scripts/serve_client_profile.sh goose-text \
  mlx-community/Qwen3-4B-Instruct-2507-4bit
```

The client launcher maps common client scenarios onto the validated serve profiles
and sets a matching default API key unless you override it with
`VLLM_MLX_CLIENT_API_KEY`.

## Client Profile Shortcuts

| Client profile | Backend profile | Default API key | Best use |
|---|---|---|---|
| `goose-text` | `text-default` | `goose-local` | Goose CLI text and streaming |
| `goose-tools` | `text-tools` | `goose-local` | Goose tool-calling |
| `open-webui-text` | `text-default` | `openwebui-local` | Open WebUI text setup |
| `open-webui-mllm` | `mllm-default` | `openwebui-local` | Open WebUI image chat |
| `jan` | `text-default` | `jan-local` | Jan remote engine |
| `anythingllm` | `text-default` | `anythingllm-local` | AnythingLLM generic OpenAI provider |
| `generic-openai` | `text-default` | `local-client` | Generic desktop/web OpenAI-compatible clients |
| `generic-mllm` | `mllm-default` | `local-client` | Generic multimodal OpenAI-compatible clients |

## Evidence-Backed Targets

Only rows with exact execution evidence are listed here as validated.

| Target | Type | Setup path | Text chat | Streaming | System prompt | Tool calling | Multimodal | Auth | Strict model id | Recommended backend path | Overall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Goose | terminal agent | built-in OpenAI provider first | pass | pass | pass | pass | planned | pass | pass | `goose-text` then `goose-tools` | conditional |
| Open WebUI | web app | OpenAI-compatible provider connection | pass | pass | pass | conditional | pass | pass | planned | `open-webui-text` then `open-webui-mllm` | conditional |

Why Goose is still overall `conditional`:

- multimodal has not been formally validated yet

Backing evidence:

- `_docs/exports/goose-vllm-mlx-validation-2026-02-28/artifacts/summary.md`
- `_docs/exports/open-webui-vllm-mlx-validation-2026-02-28/artifacts/summary.md`

## Validation Queue

These targets are actively queued but not yet published as validated:

| Target | Best path | Current state |
|---|---|---|
| Jan | `jan` | checklist + corpus + internal guide ready |
| AnythingLLM | `anythingllm` | checklist + corpus + internal guide ready |

Open WebUI notes:

- validated on local Open WebUI `v0.8.5`
- no manual model allowlist was required in the first local pass
- tool use remains `conditional` until the backend OpenAI tool-call request
  shape is independently confirmed

## Connection Notes

### OpenAI-compatible clients

Default local base URL:

```text
http://127.0.0.1:8000/v1
```

If the client runs in Docker and the backend runs on the host, switch to a host-reachable URL such as:

```text
http://host.docker.internal:8000/v1
```

For field-by-field setting behavior, use:

- [Client Settings Crosswalk](client-settings-crosswalk.md)

If you need the backend reachable off-host, run with:

```bash
LISTEN_MODE=public scripts/serve_client_profile.sh <client-profile> <model>
```

### Model id rule

- non-strict backend mode:
  - clients may often use `"default"`
- strict backend mode:
  - clients must send the exact served model id

### Client-safe default

For most first-pass client validation, start with:

```bash
scripts/serve_client_profile.sh generic-openai \
  mlx-community/Qwen3-4B-Instruct-2507-4bit
```

## Related Guides

- [Fork Operator Guide](fork-operator-guide.md)
- [Client Settings Crosswalk](client-settings-crosswalk.md)
- [Known-Good Model And Profile Matrix](model-profile-matrix.md)
- [OpenAI-Compatible Server](server.md)
