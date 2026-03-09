# Qwen Text Models: Gated Validation Snapshot (March 2026)

This report summarizes two independent pre-gated benchmark passes for text-only Qwen models on Apple Silicon.

## Validation Standard Used

- Minimum for model inclusion in the validated list: `2` gated passes.
- Required gate outcomes on each pass:
  - Tier 0 `should_continue=true`
  - Tier 1 `should_continue=true`
  - Determinism check `T1.1=PASS`
  - Quant quality check `T1.3=PASS`
- Escalation rule: run a 3rd pass if either:
  - throughput token/s coefficient of variation (CV) > `10%`, or
  - streaming TTFT CV > `15%`.

## Models Included

| Model | Passes | Gate Status | Throughput tok/s (mean ± sd) | Streaming TTFT ms (mean ± sd) | Streaming tok/s (mean ± sd) | Perplexity (mean ± sd) |
|---|---:|---|---:|---:|---:|---:|
| Qwen3-8B-4bit | 2 | PASS | 54.8177 ± 0.1684 | 227.4716 ± 13.7817 | 53.7619 ± 1.4968 | 2.2447 ± 0.0000 |
| Qwen3-30B-A3B-Instruct-2507-4bit | 2 | PASS | 52.5461 ± 1.0260 | 195.2259 ± 2.6399 | 50.6534 ± 0.1304 | 2.1927 ± 0.0000 |

## Pass-by-Pass Metrics

### Qwen3-8B-4bit

| Pass | Throughput tok/s | Prompts/s | TTFT ms mean | Streaming tok/s | Determinism | Perplexity |
|---|---:|---:|---:|---:|---|---:|
| 1 | 54.6986 | 0.8547 | 217.7264 | 54.8203 | PASS (1.0) | 2.2447 |
| 2 | 54.9368 | 0.8584 | 237.2167 | 52.7035 | PASS (1.0) | 2.2447 |

### Qwen3-30B-A3B-Instruct-2507-4bit

| Pass | Throughput tok/s | Prompts/s | TTFT ms mean | Streaming tok/s | Determinism | Perplexity |
|---|---:|---:|---:|---:|---|---:|
| 1 | 53.2716 | 1.6192 | 197.0926 | 50.7456 | PASS (1.0) | 2.1927 |
| 2 | 51.8206 | 1.5751 | 193.3592 | 50.5612 | PASS (1.0) | 2.1927 |

## Notes

- These are diagnostic-gated validation runs captured for reproducibility.
- Reference-divergence check `T1.2` is currently skipped in this environment because optional reference dependencies are not installed.
- Full provenance bundles (raw gate output, server logs, and raw benchmark JSON) are retained in internal artifacts.

Source data: [qwen-text-gated-validation-2026-03-source.json](qwen-text-gated-validation-2026-03-source.json)
