# Client Settings Crosswalk

This guide maps common OpenAI-style client settings onto the current
`vllm-mlx` backend behavior.

Use it with:
- [Fork Operator Guide](fork-operator-guide.md)
- [Client Compatibility](client-compatibility.md)
- [Known-Good Model And Profile Matrix](model-profile-matrix.md)

## Core Rules

- Default local OpenAI base URL: `http://127.0.0.1:8000/v1`
- Use model `"default"` unless the backend is running with `--strict-model-id`
- Keep backend auth enabled for publishable examples when the client supports it
- Treat GUI controls as untrusted until you confirm the exact request they emit

## Client-Safe Starting Points

| Client scenario | Recommended launcher path | Recommended model family | First-pass request guidance | Why |
|---|---|---|---|---|
| Generic text client | `generic-openai` | Qwen3 text instruct | Leave app defaults in place, then verify `max_tokens` and `stream` | Safest low-friction baseline |
| Tool-capable agent client | `goose-tools` or `text-tools` | Validated Qwen3 text instruct | Send `tools` and `tool_choice` explicitly | Keeps tool parser setup explicit |
| Deterministic debugging | `text-deterministic` | Qwen3 text instruct | Force `temperature=0.0`, `top_p=1.0` | Best repro path |
| JSON or extraction client | `text-json` | Qwen3 text or MoE instruct | Prefer `response_format={"type":"json_object"}` or schema mode | Long timeout and stable text path |
| Multimodal client | `generic-mllm` or `mllm-default` | Qwen3-VL family | Start with `temperature=0.0`, `top_p=1.0`, `enable_thinking=false` if exposed | Simplest image-validation baseline |

## Common Setting Mapping

| Client control | Backend field or behavior | Current fork interpretation | Validation note |
|---|---|---|---|
| `temperature` | request `temperature` | Direct pass-through | Compare `0.0` vs `1.0` for visibly different output shape |
| `top_p` | request `top_p` | Direct pass-through | Compare `1.0` vs low `top_p` on an open-ended prompt |
| `max tokens` | request `max_tokens` | Direct pass-through | Confirm truncation with a very small value such as `4` |
| `frequency penalty` | request `frequency_penalty` | Accepted and mapped to repetition-penalty behavior | Verify on a prompt that naturally repeats |
| `streaming` | request `stream` | Direct pass-through | Confirm incremental UI rendering, not just a delayed final flush |
| `system prompt` | leading `system` message | Should serialize as the first message in `messages` | Check for a short visible instruction such as a required suffix |
| `timeout` | usually client-side only | Must be high enough to tolerate backend latency and server `--timeout` | A timeout miss does not prove backend failure |
| `model` selector | request `model` | `"default"` is safe only when strict model id is off | Under strict mode, client must send the exact served model id |
| API key | `Authorization` or `x-api-key` header | Both header styles are accepted when auth is enabled | Confirm the client is not silently omitting auth on retries |

## Settings That Often Do Not Map Cleanly

These controls are often UI-local unless the app documents otherwise:

- conversation length or "max messages"
- local history truncation
- retry count
- local prompt templates or hidden system wrappers

Treat these as client behavior until you confirm the emitted request payload.

## Backend-Specific Caveats

### `enable_thinking`

The backend supports request-level `enable_thinking`, but many GUI clients do
not expose it directly.

Practical rule:
- if the client does not serialize `enable_thinking`, you cannot rely on the UI
  to toggle thinking even though the backend supports it
- for fair comparisons, keep the serve profile fixed and only vary the request

### Multimodal uploads

Image or mixed image+text chat only works when:

- the backend is running with a multimodal-capable model
- the backend is using a multimodal profile
- the client emits OpenAI-style multimodal content blocks

If any of those are false, an image failure does not automatically indicate a
backend defect.

### Strict model id

Strict model id is a deployment choice, not a client-compatibility default.

Use:
- non-strict mode for the easiest first-pass validation
- strict mode when you need an explicit operator contract and the client can
  send the exact served id

## First Validation Order For A New Client

1. text non-stream
2. text stream
3. system prompt
4. `max_tokens` truncation
5. `frequency_penalty` or other exposed decode control
6. multimodal single-image flow if the client supports it
7. strict-model-id only after the basic path is stable

## Related Guides

- [Fork Operator Guide](fork-operator-guide.md)
- [Client Compatibility](client-compatibility.md)
- [Known-Good Model And Profile Matrix](model-profile-matrix.md)
- [OpenAI-Compatible Server](server.md)
