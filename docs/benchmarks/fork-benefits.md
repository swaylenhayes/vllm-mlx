# Fork Benefits and Compatibility Notes

This page summarizes what the fork changed beyond raw throughput numbers.

Scope date: 2026-02-25  
Fork: `swaylenhayes/vllm-mlx`  
Current head for these changes: `9c07636`

## Why this exists

The fork README tracks top-level phase throughput gains (upstream vs P0 vs P1).  
This document tracks the compatibility and reliability outcomes from those backend changes.

## Compatibility outcomes (re-evaluation summary)

From local model re-evaluation runs, the fork runtime policy and parser work improved practical usability for several model families by reducing runaway output, improving reasoning/output separation, and unlocking additional tool-call parsing formats.

| Model family | Re-eval issue before | Fork-side change used | Outcome direction |
|---|---|---|---|
| LiquidAI LFM / WaveCut | Unparsed proprietary tool-call format | `--tool-call-parser liquidai` | Tool-call extraction path now available |
| Thinking-heavy models | Large reasoning blocks reduced usable output budget | Reasoning parser improvements and `max_thinking_tokens` control | Better control over visible answer/tool budget |
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

## Related references

- Fork root summary and phase tables: [`README.md`](../../README.md)
- Tool-calling guide: [`docs/guides/tool-calling.md`](../guides/tool-calling.md)
- Server options: [`docs/guides/server.md`](../guides/server.md)
- CLI reference: [`docs/reference/cli.md`](../reference/cli.md)
