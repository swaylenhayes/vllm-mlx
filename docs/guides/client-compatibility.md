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
cd /path/to/vllm-mlx

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
| `cherry-studio` | `text-default` | `cherrystudio-local` | Cherry Studio desktop baseline |
| `chatbox` | `text-default` | `chatbox-local` | Chatbox desktop baseline |
| `librechat` | `text-default` | `librechat-local` | LibreChat text baseline |
| `witsy` | `text-default` | `witsy-local` | Witsy desktop baseline |
| `jan` | `text-default` | `jan-local` | Jan remote engine |
| `anythingllm` | `text-default` | `anythingllm-local` | AnythingLLM generic OpenAI provider |
| `boltai` | `text-default` | `boltai-local` | BoltAI desktop baseline |
| `generic-openai` | `text-default` | `local-client` | Generic desktop/web OpenAI-compatible clients |
| `generic-mllm` | `mllm-default` | `local-client` | Generic multimodal OpenAI-compatible clients |

## Evidence-Backed Targets

Only rows with exact execution evidence are listed here as validated.

| Target | Type | Setup path | Text chat | Streaming | System prompt | Tool calling | Multimodal | Auth | Strict model id | Recommended backend path | Overall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Goose | terminal agent | built-in OpenAI provider first | pass | pass | pass | pass | planned | pass | pass | `goose-text` then `goose-tools` | conditional |
| Open WebUI | web app | OpenAI-compatible provider connection | pass | pass | pass | conditional | pass | pass | planned | `open-webui-text` then `open-webui-mllm` | conditional |
| Cherry Studio | desktop app | custom service provider (`type=OpenAI`) | pass | pass | pass | planned | planned | pass | inferred pass | `cherry-studio` | conditional |

Why Goose is still overall `conditional`:

- multimodal has not been formally validated yet

Why Cherry Studio is still overall `conditional`:

- tool calling has not been formally validated yet
- multimodal has not been formally validated yet

Backing evidence:

- Validation runs are evidence-backed and archived in internal run artifacts used to populate this table.

## Validation Queue

These targets are actively queued but not yet published as validated:

| Target | Best path | Current state |
|---|---|---|
| Chatbox | `chatbox` | next external desktop client row after Cherry Studio |
| LibreChat | `librechat` | queued for capability-signaling and agent-style validation |
| Witsy | `witsy` | queued for desktop MCP and capability-signaling validation |
| Jan | `jan` | checklist + corpus + internal guide ready, but deferred after reprioritization |
| AnythingLLM | `anythingllm` | checklist + corpus + internal guide ready, but deferred after reprioritization |
| BoltAI | `boltai` | lower-priority desktop client row |

Open WebUI notes:

- validated on local Open WebUI `v0.8.5`
- no manual model allowlist was required in the first local pass
- tool use remains `conditional` until the backend OpenAI tool-call request
  shape is independently confirmed

Cherry Studio notes:

- validated on local Cherry Studio `v1.7.25`
- use a custom service provider for third-party backends; the built-in OpenAI
  provider warns that it no longer supports the older third-party calling path
- successful local path:
  - start `scripts/serve_client_profile.sh cherry-studio mlx-community/Qwen3-4B-Instruct-2507-4bit`
  - create a provider with type `OpenAI`
  - set API host to `http://127.0.0.1:8000/v1`
  - add the served model from `Manage`
  - run `Check`
- observed validation contract:
  - `GET /v1/models`
  - streamed `POST /v1/chat/completions`
  - exact model id: `mlx-community/Qwen3-4B-Instruct-2507-4bit`
  - `roles=['system', 'user']`
  - last user message: `hi`
- the provider check disconnects after the first streamed chunk and still reports
  success; this is expected for Cherry Studio's connectivity probe
- screenshot artifact: [Cherry Studio connection check success](../assets/client-compatibility/cherry-studio-check-success.png)

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
