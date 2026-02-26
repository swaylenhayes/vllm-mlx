# Fork Improvement Log (Measured, Append-Only)

This log is the ongoing record of measured backend improvements in this fork.

Rules:
- Append new entries at the top.
- Only include entries with reproducible measurement notes.
- Keep claims bounded to what was measured.

## Entries

### 2026-02-26 - U3-A upstream import closure (`11e0bd7`, `2888fbf`, `1830be0`, `cdae83a`)

- Change:
  - Completed U3-A upstream import pass on fork branch:
    - `#95` tool-call argument schema coercion hardening (`11e0bd7`)
    - `#109` UTF-8-safe incremental streaming detokenizer integration (`2888fbf`)
    - `#105` MLLM prefill-step CLI clarity/validation (`1830be0`)
  - Added fork follow-on patch:
    - `cdae83a` wires MLLM prefill-step override into batched-engine startup.
  - Evaluated `#54` as already present in fork behavior (empty cherry-pick after reconciliation).
- Measurement status: functional/regression validation
- Measurement setup:
  - Coercion closure suite:
    - targeted coercion tests in `tests/test_server.py`
    - full repo suite (`.venv/bin/python -m pytest -q`)
  - UTF-8 detokenizer suite:
    - `tests/test_streaming_detokenizer.py`
    - `tests/test_batching.py`
    - `tests/test_batching_deterministic.py`
    - `tests/test_continuous_batching.py`
    - `tests/test_mllm_continuous_batching.py`
    - `tests/test_server.py`
  - MLLM prefill override suite:
    - `tests/test_cli_runtime_policies.py`
    - `tests/test_cli_localhost.py`
    - `tests/test_docs_drift.py`
    - `tests/test_mllm_continuous_batching.py`
    - `tests/test_server.py`
  - `#54` no-op confirmation suite:
    - `tests/test_simple_engine.py`
    - `tests/test_mllm.py`
    - `tests/test_mllm_continuous_batching.py`
    - `tests/test_server.py`
- Result:
  - `#95` closure:
    - targeted coercion tests: `3 passed`
    - full suite: `1031 passed, 5 skipped, 20 deselected`
  - `#109` suite: `137 passed, 8 deselected`
  - `#105` suite: `100 passed, 6 deselected`
  - `#54` confirmation suite: `115 passed, 12 deselected`
- Practical impact:
  - Tool-call argument coercion now enforced across parsing/stream fallback paths with test lock-in.
  - Streaming decode now protects multi-byte UTF-8 boundaries in incremental output.
  - MLLM prefill-step override is documented, validated, and wired through batched startup.
  - U3-A import set is closed; next implementation track can pivot to Variant B patch work.

### 2026-02-26 - `f514235` (upstream MLLM null-field serialization fix sync)

- Change:
  - Cherry-picked upstream `#104` to MLLM paths:
    - use `exclude_none=True` for Pydantic serialization
    - apply `None` filtering in fallback dict conversion
- Measurement status: functional validation
- Measurement setup:
  - Targeted MLLM/server regression suite:
    - `tests/test_mllm.py`
    - `tests/test_mllm_continuous_batching.py`
    - `tests/test_server.py`
    - `tests/test_api_models.py`
- Result:
  - `177 passed, 12 deselected`
  - No regression observed in tested MLLM and API paths.
- Practical impact:
  - Prevents null-key artifacts (for example `image_url: null`) that can break key-presence template checks and strict client validation.

### 2026-02-25 - `d33283d` + R2C live dataset (batch divergence confidence pass)

- Change:
  - Upgraded batch invariance harness with repeated-run confidence support (`--runs`, `--confidence`, `--run-cooldown`) and connection retry handling.
  - Collected first full R2C live dataset (5 runs/model) for text + two VLM models under batched runtime conditions.
- Measurement status: benchmarked
- Measurement setup:
  - Models:
    - `mlx-community/Qwen3-4B-Instruct-2507-4bit`
    - `swaylenhayes/ZwZ-8B-VL-MLX-4bit`
    - `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`
  - Runtime profile:
    - `vllm-mlx serve <model> --localhost --port 8000 --runtime-mode auto --cache-strategy auto`
    - plus `--mllm` for VLM models
  - Workload:
    - 10 fixed prompts, deterministic decode
    - serial vs concurrent (`concurrency=2`)
    - repeated runs: `N=5`, confidence level `95%`
- Baseline:
  - R2A one-pass baseline at `38e13c8` showed all models below 95% token agreement.
- Result:
  - Qwen3-4B-Instruct-2507-4bit:
    - token agreement `97.86%` (95% CI `97.86-97.86`)
    - exact match `80.00%`
  - ZwZ-8B-VL-MLX-4bit:
    - token agreement `68.65%` (95% CI `65.82-71.49`)
    - exact match `34.00%`
  - Qwen3-VL-30B-A3B-Instruct-4bit:
    - token agreement `63.85%` (95% CI `57.20-70.50`)
    - exact match `32.00%`
  - Delta:
    - Text model moved from prior fail state to above-threshold under lower-concurrency repeated protocol.
    - Both VLM models remain well below threshold with confidence bands that do not approach 95%.
- Caveats:
  - This profile uses `concurrency=2`, not the earlier stress setting (`10`); values are not directly interchangeable.
  - Exact-match variability remains non-trivial even on the passing text model.
- Links:
  - Summary: `benchmarks/phase-results/batch-invariance-2026-02-25/summary.md`
  - Per-model JSON reports: `*_r2c.json`

### 2026-02-25 - `38e13c8` (R2A batch invariance baseline run)

- Change:
  - Ran first live 3-model batch invariance matrix using the shipped harness (`scripts/batch_invariance_harness.py`).
  - Captured serial-vs-concurrent agreement artifacts for text + VLM + MoE-VLM models.
- Measurement status: benchmarked
- Measurement setup:
  - Models:
    - `mlx-community/Qwen3-4B-Instruct-2507-4bit`
    - `swaylenhayes/ZwZ-8B-VL-MLX-4bit`
    - `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`
  - Runtime profile:
    - `vllm-mlx serve <model> --localhost --port 8000 --runtime-mode auto --cache-strategy auto`
    - plus `--mllm` for VLM models
  - Workload:
    - 10 fixed prompts, deterministic decode (`temperature=0`)
    - serial pass vs concurrent pass (`concurrency=10`)
- Baseline:
  - Commit: first live matrix using harness shipped at `38e13c8`
  - Threshold: token agreement `>=95%` interpreted as acceptable batch invariance
- Result:
  - Commit: `38e13c8`
  - Key metric(s):
    - Qwen3-4B-Instruct-2507-4bit: exact `60.00%`, token agreement `83.42%`
    - ZwZ-8B-VL-MLX-4bit: exact `30.00%`, token agreement `53.24%`
    - Qwen3-VL-30B-A3B-Instruct-4bit: exact `30.00%`, token agreement `48.64%`
  - Delta:
    - All models below the 95% target; baseline indicates batch-composition sensitivity.
- Caveats:
  - This is a first-pass baseline and not yet a multi-run confidence interval.
  - Deterministic settings reduce but do not eliminate nondeterministic effects.
- Links:
  - Full artifact bundle: `benchmarks/phase-results/batch-invariance-2026-02-25/`
  - Summary: `benchmarks/phase-results/batch-invariance-2026-02-25/summary.md`

### 2026-02-25 - `219cfda` (I7 MLLM tool-calling validation)

- Change:
  - Enabled MLLM tool-calling metadata passthrough (`tools` + `tool_choice`) across simple and batched chat paths.
  - Propagated `tool_choice` through server request plumbing.
- Measurement status: benchmarked (Tier C + mixed image/tool probe)
- Measurement setup:
  - Models: `Qwen3-VL-4B-Instruct-4bit`, `ZwZ-8B-VL-4bit`
  - Runtime profile:
    - `--localhost --mllm --runtime-mode auto --cache-strategy auto`
    - `--enable-auto-tool-choice --tool-call-parser auto`
  - Workload:
    - Tier C probes (`slack`, `file-search`, `weather`)
    - mixed image+tool request (single request with image + tool schema)
  - Compared states:
    - Pre-I7 baseline: VLM MLLM tool-calling `0/9`
    - Post-I7 validation target: `219cfda`
- Baseline:
  - Commit: pre-I7 MLLM path
  - Key metric(s):
    - Qwen3-VL-4B-Instruct-4bit: `0/9`
    - ZwZ-8B-VL-4bit: `0/9`
- Result:
  - Commit: `219cfda`
  - Key metric(s):
    - Qwen3-VL-4B-Instruct-4bit: `9/9`, avg latency `0.86s`, `finish_reason=tool_calls`
    - ZwZ-8B-VL-4bit: `9/9`, avg latency `1.12s`, `finish_reason=tool_calls`
    - Mixed image+tool request: structured tool call emitted (`finish_reason=tool_calls`, `1.40s`)
  - Delta:
    - `0/9 -> 9/9` on both validated VLMs
- Caveats:
  - Validation currently covers two Qwen-family VLMs; other model families may require parser/profile tuning.
  - Mixed image+tool engine behavior is correct; argument quality remains model-dependent.
- Links:
  - Validation source: local workspace spec `i7-validation-results-2026-02-25.md`
  - Compatibility detail: [`fork-benefits.md`](fork-benefits.md)

### 2026-02-25 - `05869bc` (I6 validation sweep on latest branch state `219cfda`)

- Change:
  - I6 server-side tool-call spray mitigation:
    - exact duplicate dedupe (`function + canonical args`)
    - large same-function burst collapse (threshold guard)
- Measurement status: benchmarked (tool-calling probes, focused follow-up)
- Measurement setup:
  - Models: `WaveCut LFM2.5-DWQ-4bit`, `LFM2.5-1.2B-Thinking-8bit`, `Nanbeige4.1-3B-8bit`
  - Runtime profile:
    - `--localhost --runtime-mode auto --cache-strategy auto`
    - WaveCut/LFM: `--enable-auto-tool-choice --tool-call-parser liquidai --reasoning-parser qwen3`
    - Nanbeige: `--enable-auto-tool-choice --tool-call-parser auto --reasoning-parser qwen3`
  - Workload: focused 3-probe subset plus sanity/full-probe checks
  - Compared states:
    - Pre-I6 baseline: `9c07636`
    - Post-I6 validation branch state: `219cfda` (includes I6 and later changes)
- Baseline:
  - Commit: `9c07636`
  - Key metric(s):
    - WaveCut@128: `3/3`, `13` calls
    - WaveCut@256: `0/3`, `0` calls
    - LFM@128: `3/3`, `1` call
    - Nanbeige@256: `3/3`, `15` calls
- Result:
  - Commit: `219cfda` (I6 validation target state)
  - Key metric(s):
    - WaveCut@128: `3/3`, `7` calls
    - WaveCut@256: `3/3`, `2` calls
    - LFM@128: `0/3`, `0` calls
    - LFM@256: `3/3`, `1` call
    - Nanbeige@256: `0/3`, `0` calls
  - Delta:
    - Exact dedupe confirmed for WaveCut@128: `13 -> 7`
    - No universal quality pattern at budget `256` across all models
- Caveats:
  - Some observed score shifts occur in "no tool-call emitted" mode and are likely run variance/model stochasticity, not direct dedupe side effects.
  - I6 mitigation runs after tool calls are emitted; it cannot by itself explain failures where no tool calls are produced.
  - Single-shot results on 1.2B-3B models are unstable; repeated-run methodology is recommended.
- Links:
  - Validation source: local workspace spec `i6-validation-results-2026-02-25.md`
  - Compatibility detail: [`fork-benefits.md`](fork-benefits.md)

### 2026-02-25 - `9c07636` (P1.10 validation sweep)

- Change:
  - Added engine-level forced think exit (`</think>`) in `SimpleEngine` LLM decode path.
  - Wired server-side parser-aware thinking-boundary kwargs for engine routing.
- Measurement status: benchmarked (tool-calling probes)
- Measurement setup:
  - Models: `WaveCut LFM2.5-DWQ-4bit`, `LFM2.5-1.2B-Thinking-8bit`, `Nanbeige4.1-3B-8bit`
  - Runtime profile:
    - `--localhost --runtime-mode auto --cache-strategy auto`
    - WaveCut/LFM: `--enable-auto-tool-choice --tool-call-parser liquidai --reasoning-parser qwen3`
    - Nanbeige: `--enable-auto-tool-choice --tool-call-parser auto --reasoning-parser qwen3`
  - Workload: 9 tool-calling probes per model (3 tools x 3 scenarios)
  - Budget sweep:
    - WaveCut: `64`, `128`, `256`
    - LFM-Thinking: `64`, `128`, `256`
    - Nanbeige: `64`, `128`, `256`
- Baseline:
  - Commit: `d890ef6`
  - Key metric(s):
    - WaveCut: `6/9`
    - LFM-Thinking: `6/9`
    - Nanbeige4.1: `6/9`
- Result:
  - Commit: `9c07636`
  - Key metric(s):
    - WaveCut: `9/9` (best at `64`)
    - LFM-Thinking: `9/9` (best at `128`)
    - Nanbeige4.1: `9/9` (best at `256`)
  - Delta:
    - WaveCut: `+3`
    - LFM-Thinking: `+3`
    - Nanbeige4.1: `+3`
- Caveats:
  - Optimal `max_thinking_tokens` is model-specific; no universal value.
  - Some budget/model combinations produce tool-call spray (13-15 redundant calls) despite passing score checks.
  - Engine-level forcing is on `SimpleEngine` LLM path; other paths keep API-layer budget behavior.
- Links:
  - Validation source: local workspace specs note (2026-02-25)
  - Compatibility detail: [`fork-benefits.md`](fork-benefits.md)

### 2026-02-24 - `d890ef6` (thinking-model validation pass)

- Change:
  - Validated LiquidAI parser + reasoning/tool parser ordering fix + thinking budget controls together.
- Measurement status: benchmarked (tool-calling probes)
- Measurement setup:
  - Models: `WaveCut LFM2.5-DWQ-4bit`, `LFM2.5-1.2B-Thinking-8bit`, `Nanbeige4.1-3B-8bit`
  - Runtime profile:
    - `--localhost --runtime-mode auto --cache-strategy auto`
    - WaveCut/LFM: `--enable-auto-tool-choice --tool-call-parser liquidai --reasoning-parser qwen3 --max-thinking-tokens 256`
    - Nanbeige: `--enable-auto-tool-choice --tool-call-parser auto --reasoning-parser qwen3 --max-thinking-tokens 256`
  - Workload: 9 tool-calling probes per model (3 tools x 3 scenarios)
- Baseline:
  - Commit: pre-`d890ef6`
  - Key metric(s):
    - WaveCut: `0/9`
    - LFM-Thinking: `0/9`
    - Nanbeige4.1: `6/9`
- Result:
  - Commit: `d890ef6`
  - Key metric(s):
    - WaveCut: `6/9`
    - LFM-Thinking: `6/9`
    - Nanbeige4.1: `6/9`
  - Delta:
    - WaveCut: `+6`
    - LFM-Thinking: `+6`
    - Nanbeige4.1: `0`
- Caveats:
  - All three models failed the same ambiguous file-search probe in this set (`6/9` ceiling).
  - Interpreted as probe/model-capacity limitation at this scale, not parser extraction failure.
- Links:
  - Validation summary source: local workspace specs note (2026-02-24)
  - Prior compatibility overview: [`fork-benefits.md`](fork-benefits.md)

### 2026-02-24 - `d890ef6`

- Change:
  - Added LiquidAI/LFM tool parser (`liquidai`, `liquid`, `lfm`)
  - Fixed reasoning/tool parser ordering in streaming and non-streaming paths
  - Added thinking budget controls (`--max-thinking-tokens`, request `max_thinking_tokens`)
- Measurement status: functional validation complete, throughput delta not yet benchmarked in phase table
- Validation:
  - Tool parser/ordering/tests passed in targeted suite
  - Reasoning budget controls validated at request/CLI contract level
- Notes:
  - Throughput impact should be recorded after next benchmark round

### 2026-02-24 - `140377c`

- Change:
  - Added `frequency_penalty` request support (mapped to backend repetition penalty)
- Measurement status: validated behavior
- Validation:
  - Request model, server mapping, and tests updated and passing
  - Confirmed usable as repetition tuning control from API clients

## Entry Template

Copy this block for each new measured improvement:

```md
### YYYY-MM-DD - `<commit>`

- Change:
  - <short summary of backend change>
  - <optional second bullet>
- Measurement status: <benchmarked|validated behavior|pending benchmark>
- Measurement setup:
  - Model(s): `<model id(s)>`
  - Command/profile: `<serve/bench command>`
  - Workload: <prompt/tool/request profile>
- Baseline:
  - Commit: `<baseline commit>`
  - Key metric(s): <value(s)>
- Result:
  - Commit: `<new commit>`
  - Key metric(s): <value(s)>
  - Delta: <percent/value delta>
- Caveats:
  - <what this does not prove>
- Links:
  - <docs/tests/benchmark artifact paths>
```
