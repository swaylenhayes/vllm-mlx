# Fork Benefits and Compatibility Notes

This page summarizes what the fork changed beyond raw throughput numbers.

Scope date: 2026-02-26  
Fork: `swaylenhayes/vllm-mlx`  
Current head for these notes: `cdae83a` (includes U3-A upstream import closure)

## Why this exists

The fork README tracks top-level phase throughput gains (upstream vs P0 vs P1).  
This document tracks the compatibility and reliability outcomes from those backend changes.

## Highlights snapshot

- Throughput (phase benchmark): `+50.25%` token throughput vs upstream baseline.
- Thinking-model validation: `6/9 -> 9/9` after P1.10 + follow-up hardening.
- MLLM tool-calling validation: `0/9 -> 9/9` on two validated VLMs.
- R2C repeated-run divergence profile:
  - text model above threshold (`97.86%` token agreement),
  - tested VLM models well below threshold in batched mode.

## Compatibility outcomes (re-evaluation summary)

From local model re-evaluation runs, the fork runtime policy and parser work improved practical usability for several model families by reducing runaway output, improving reasoning/output separation, and unlocking additional tool-call parsing formats.

| Model family | Re-eval issue before | Fork-side change used | Outcome direction |
|---|---|---|---|
| LiquidAI LFM / WaveCut | Unparsed proprietary tool-call format | `--tool-call-parser liquidai` | Tool-call extraction path now available |
| Thinking-heavy models | Large reasoning blocks reduced usable output budget | Reasoning parser improvements and `max_thinking_tokens` control | Better control over visible answer/tool budget |
| Qwen-family VLMs in MLLM mode | Tool metadata accepted but dropped in MLLM path (`0/9`) | I7 MLLM passthrough (`tools` + `tool_choice`) | Validated `9/9` tool-calling on two VLMs |
| Repetition-prone prompts | Limited request-level repetition tuning | `frequency_penalty` API support | Client-side tuning now possible |

## R2C batch divergence outcomes (2026-02-25)

Repeated-run protocol (5 runs/model, 95% confidence, `concurrency=2`, `max_tokens=32`) under batched runtime path:

| Model | Token agreement mean | Token agreement 95% CI | Exact match mean | Interpretation |
|---|---:|---:|---:|---|
| Qwen3-4B-Instruct-2507-4bit | 97.86% | 97.86%-97.86% | 80.00% | Clears 95% token gate in this profile |
| ZwZ-8B-VL-MLX-4bit | 68.65% | 65.82%-71.49% | 34.00% | Persistent severe divergence |
| Qwen3-VL-30B-A3B-Instruct-4bit | 63.85% | 57.20%-70.50% | 32.00% | Persistent severe divergence |

Operational guidance from this dataset:

- Keep default monitor policy:
  - `--batch-divergence-threshold 0.95`
  - `--batch-divergence-action warn`
- For correctness-sensitive VLM production traffic:
  - use `--batch-divergence-action serialize` (or force simple-engine profile)
- Treat current VLM batched mode as throughput-optimized with known quality risk unless serialized fallback is active.

## Upstream sync impact (2026-02-26)

Integrated upstream change `#104` (commit `6d55631`, fork commit `f514235`):
- MLLM message/content-part serialization now excludes `None` fields.
- Prevents null keys (for example `image_url: null`) from triggering key-presence template logic and strict client schema failures.

U3-A upstream import closure:
- `#95` (`11e0bd7`): tool-call argument schema coercion hardening.
- `#109` (`2888fbf`): UTF-8-safe incremental decode via streaming detokenizer.
- `#105` (`1830be0`): MLLM prefill override CLI clarity/validation.
- Fork follow-on (`cdae83a`): wires MLLM prefill override into batched-engine startup.
- `#54` evaluated as already-present behavior in fork (empty cherry-pick after reconciliation).

Validation summary for U3-A closure:
- `#95` closure included full venv suite pass (`1031 passed, 5 skipped, 20 deselected`) after targeted coercion tests.
- `#109` validation suite (`streaming_detokenizer + batching + mllm_continuous_batching + server`): `137 passed, 8 deselected`.
- `#105` validation suite (`cli_runtime_policies + cli_localhost + docs_drift + mllm_continuous_batching + server`): `100 passed, 6 deselected`.

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
