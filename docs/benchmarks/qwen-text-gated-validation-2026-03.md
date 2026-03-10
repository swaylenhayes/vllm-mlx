# Qwen Text Models: Gated Validation Snapshot (March 2026)

This report summarizes three independent diagnostic-gated benchmark passes for text-only Qwen models on Apple Silicon.

## Validation Standard Used

- Baseline admission tier: minimum `2` gated passes.
- Release-grade tier: minimum `3` gated passes.
- Required gate outcomes on each pass:
  - Tier 0 `should_continue=true`
  - Tier 1 `should_continue=true`
  - Determinism check `T1.1=PASS`
  - Quant quality check `T1.3=PASS`
- Auto-escalation rule: run a 4th pass if either metric remains above threshold after pass 3:
  - throughput token/s coefficient of variation (CV) > `10%`, or
  - streaming TTFT CV > `15%`.
- Periodic qualifier: run reference-divergence check `T1.2` on a recurring basis for each model family/quantization class (not required on every individual pass).
- `T1.2` policy closure (`2026-03-10`) for current MLX quantized checkpoints:
  - `PASS`: qualifier satisfied.
  - `SKIP-KNOWN` (known reference-load incompatibility, `quantization_config` missing `quant_method`): non-blocking when required per-pass gates (`T0/T1/T1.1/T1.3`) pass and qualifier provenance is attached.
  - `SKIP-UNKNOWN` or `FAIL`: blocking until triaged/fixed or explicitly waived.
  - cadence: run before first public promotion per model family/quantization class, rerun after dependency/toolchain changes, and rerun at least every 30 days for active promoted families.

## Models Included

| Model | Passes | Gate Status | Throughput tok/s (mean ± sd) | Streaming TTFT ms (mean ± sd) | Streaming tok/s (mean ± sd) | Perplexity (mean ± sd) |
|---|---:|---|---:|---:|---:|---:|
| Qwen3-8B-4bit | 3 | PASS | 53.9902 ± 1.4382 | 243.6159 ± 29.6124 | 51.7054 ± 3.7159 | 2.2447 ± 0.0000 |
| Qwen3-30B-A3B-Instruct-2507-4bit | 3 | PASS | 51.5157 ± 1.9265 | 198.4372 ± 5.8670 | 49.8157 ± 1.4538 | 2.1927 ± 0.0000 |

## Pass-by-Pass Metrics

### Qwen3-8B-4bit

| Pass | Throughput tok/s | Prompts/s | TTFT ms mean | Streaming tok/s | Determinism | Perplexity |
|---|---:|---:|---:|---:|---|---:|
| 1 | 54.6986 | 0.8547 | 217.7264 | 54.8203 | PASS (1.0) | 2.2447 |
| 2 | 54.9368 | 0.8584 | 237.2167 | 52.7035 | PASS (1.0) | 2.2447 |
| 3 | 52.3353 | 0.8177 | 275.9047 | 47.5924 | PASS (1.0) | 2.2447 |

### Qwen3-30B-A3B-Instruct-2507-4bit

| Pass | Throughput tok/s | Prompts/s | TTFT ms mean | Streaming tok/s | Determinism | Perplexity |
|---|---:|---:|---:|---:|---|---:|
| 1 | 53.2716 | 1.6192 | 197.0926 | 50.7456 | PASS (1.0) | 2.1927 |
| 2 | 51.8206 | 1.5751 | 193.3592 | 50.5612 | PASS (1.0) | 2.1927 |
| 3 | 49.4550 | 1.5032 | 204.8598 | 48.1404 | PASS (1.0) | 2.1927 |

## Notes

- These are diagnostic-gated validation runs captured for reproducibility.
- Release-grade pass count is now met for both included models (3/3).
- Auto-escalation to pass 4 is not currently triggered: throughput CV and TTFT CV are below escalation thresholds for both models.
- Current `T1.2` classification for included MLX quantized checkpoints is `SKIP-KNOWN`, driven by a `transformers` reference-load quantization-config incompatibility (`missing quant_method`).
- Full provenance bundles (raw gate output, server logs, and raw benchmark JSON) are retained in internal artifacts.

Source data: [qwen-text-gated-validation-2026-03-source.json](qwen-text-gated-validation-2026-03-source.json)
