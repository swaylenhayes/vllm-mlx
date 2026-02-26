# VB1 Throughput Snapshot (Default vs Deterministic)

Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit`
Workload: 10 prompts, max_tokens=64, concurrency=10

| Profile | Total time (s) | Prompts/s | Tokens/s | Throughput (tok/s) |
|---|---:|---:|---:|---:|
| default | 6.95 | 1.44 | 92.04 | 113.62 |
| deterministic | 7.16 | 1.40 | 89.44 | 89.44 |

## Delta (deterministic vs default)

| Metric | Delta |
|---|---:|
| Total time | +2.91% |
| Prompts/s | -2.82% |
| Tokens/s | -2.82% |
| Throughput | -21.27% |

Notes:
- `Tokens/s` (completion tokens per second) is the primary comparable metric for this run pair.
- Deterministic run reported `prompt_tokens=0` in response usage on this profile, which makes `Throughput (tok/s)` (prompt + completion) artificially lower and not directly comparable for policy decisions.
