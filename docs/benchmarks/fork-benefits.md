# Fork Benefits and Compatibility Notes

This page summarizes what the fork changed beyond raw throughput numbers.

Scope date: 2026-02-25  
Fork: `swaylenhayes/vllm-mlx`  
Current head for these notes: `c25a85a` (I7 behavior validated on `219cfda`)

## Why this exists

The fork README tracks top-level phase throughput gains (upstream vs P0 vs P1).  
This document tracks the compatibility and reliability outcomes from those backend changes.

## Compatibility outcomes (re-evaluation summary)

From local model re-evaluation runs, the fork runtime policy and parser work improved practical usability for several model families by reducing runaway output, improving reasoning/output separation, and unlocking additional tool-call parsing formats.

| Model family | Re-eval issue before | Fork-side change used | Outcome direction |
|---|---|---|---|
| LiquidAI LFM / WaveCut | Unparsed proprietary tool-call format | `--tool-call-parser liquidai` | Tool-call extraction path now available |
| Thinking-heavy models | Large reasoning blocks reduced usable output budget | Reasoning parser improvements and `max_thinking_tokens` control | Better control over visible answer/tool budget |
| Qwen-family VLMs in MLLM mode | Tool metadata accepted but dropped in MLLM path (`0/9`) | I7 MLLM passthrough (`tools` + `tool_choice`) | Validated `9/9` tool-calling on two VLMs |
| Repetition-prone prompts | Limited request-level repetition tuning | `frequency_penalty` API support | Client-side tuning now possible |

## P1.10 validation outcomes (2026-02-25)

Thinking-model tool-calling re-evaluation (`d890ef6` -> `9c07636`):

| Model | `d890ef6` | `9c07636` | Best budget | Quality |
|---|---:|---:|---:|:---:|
| WaveCut LFM2.5-DWQ-4bit | 6/9 | **9/9** | 64 | A |
| LFM2.5-1.2B-Thinking-8bit | 6/9 | **9/9** | 128 | A |
| Nanbeige4.1-3B-8bit | 6/9 | **9/9** | 256 | B |

Key points:
- Engine-level think-exit forcing breaks the prior `6/9` ceiling for this probe set.
- Best `max_thinking_tokens` value is model-specific.
- A new failure mode appears at non-optimal budgets: redundant tool-call spray.

## I6 validation outcomes (2026-02-25, focused follow-up)

Focused re-check (`9c07636` baseline vs latest branch state `219cfda`) on spray-prone paths:

| Model | Budget | Baseline | Latest | Summary |
|---|---:|---|---|---|
| WaveCut DWQ | `128` | `3/3`, `13` calls | `3/3`, `7` calls | Exact dedupe confirmed |
| WaveCut DWQ | `256` | `0/3`, `0` calls | `3/3`, `2` calls | Strong quality gain in latest state |
| LFM-Thinking | `128` | `3/3`, `1` call | `0/3`, `0` calls | Variant run behavior |
| LFM-Thinking | `256` | untested | `3/3`, `1` call | Clean single-call output |
| Nanbeige4.1 | `256` | `3/3`, `15` calls | `0/3`, `0` calls | Variant run behavior |

Interpretation:
- I6 is safe to ship and does not show a clear regression signature tied to dedupe logic.
- I6 clearly helps exact duplicate spray.
- A universal budget is still not proven; repeated-run profiling remains required.

## I7 validation outcomes (2026-02-25)

MLLM tool-calling validation (`0/9` baseline -> `219cfda`):

| Model | Parser | Baseline | Latest | Calls/probe | Avg latency | finish_reason |
|---|---|---:|---:|---:|---:|---|
| Qwen3-VL-4B-Instruct-4bit | `auto` | `0/9` | `9/9` | `1` | `0.86s` | `tool_calls` |
| ZwZ-8B-VL-4bit | `auto` | `0/9` | `9/9` | `1` | `1.12s` | `tool_calls` |

Mixed image+tool request:
- `ZwZ-8B-VL-4bit` emitted structured `tool_calls` with `finish_reason=tool_calls` in `1.40s`.
- Engine behavior is validated; argument quality remains model-dependent.

Interpretation:
- I7 hypothesis is confirmed for the validated models/workload.
- Single-model VLM+tool operation is now viable for this tested set.
- Broader VLM-family coverage remains incremental follow-up.

## Fork runtime profile

Recommended baseline profile for local single-user testing:

```bash
vllm-mlx serve <model-id> \
  --localhost \
  --runtime-mode auto \
  --cache-strategy auto
```

Reasoning models:

```bash
vllm-mlx serve <model-id> \
  --localhost --runtime-mode auto --cache-strategy auto \
  --reasoning-parser qwen3
```

LiquidAI/WaveCut tool-calling models:

```bash
vllm-mlx serve <model-id> \
  --localhost --runtime-mode auto --cache-strategy auto \
  --enable-auto-tool-choice --tool-call-parser liquidai
```

## New controls added by the fork

1. `frequency_penalty` request support (mapped to backend repetition control)
2. LiquidAI parser aliases: `liquidai`, `liquid`, `lfm`
3. Thinking budget controls:
- CLI: `--max-thinking-tokens`
- Request field: `max_thinking_tokens`

## Caveat (current status)

Thinking budget behavior now has two layers:
- Engine layer (`SimpleEngine` LLM path): can force-close reasoning with injected `</think>` and continue decode.
- API layer (fallback): caps emitted reasoning tokens and routes overflow into content/tool parsing flow.

Operational caveats:
- Engine-level forcing depends on runtime path (`SimpleEngine` LLM route).
- Non-optimal budgets can degrade quality via redundant tool-call spray even when raw score is high.
- Small-model single-run outcomes can vary; treat one-shot probe results as directional.

## Related references

- Fork root summary and phase tables: [`README.md`](../../README.md)
- Tool-calling guide: [`docs/guides/tool-calling.md`](../guides/tool-calling.md)
- Server options: [`docs/guides/server.md`](../guides/server.md)
- CLI reference: [`docs/reference/cli.md`](../reference/cli.md)
