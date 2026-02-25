# Batch Invariance Baseline (R2A)

Date: 2026-02-25

Method: serial (batch_size=1) vs concurrent (10-way) runs on identical prompts at deterministic decode settings (`temperature=0`).

| Model | Exact Match | Token Agreement | Verdict |
|---|---:|---:|---|
| Qwen3-4B-Instruct-2507-4bit (text) | 60.00% | 83.42% | Potential violation (<95%) |
| ZwZ-8B-VL-MLX-4bit (vision) | 30.00% | 53.24% | Potential violation (<95%) |
| Qwen3-VL-30B-A3B-Instruct-4bit (MoE vision) | 30.00% | 48.64% | Potential violation (<95%) |

## Outcome

- All three planned roster models are below the 95% token-agreement threshold.
- This establishes a reproducible baseline indicating batch-composition sensitivity in current runtime behavior.
- Next step is `R2B`: add divergence monitoring in `/health/diagnostics` and evaluate a safety fallback mode for correctness-sensitive runs.

## Artifacts

- Run matrix: `model-matrix-corrected.txt`
- Run status: `run-summary-corrected.tsv`
- Per-model harness logs: `*.harness.corrected.txt`
- Per-model JSON reports: `*.corrected.json`
- Server logs: `logs/*.server.corrected.log`

## Supplemental Probe

- Additional live run on currently loaded model: `mlx-community/Qwen3-VL-4B-Instruct-4bit`
- Results: exact match `30.00%`, token agreement `47.20%` (`qwen3-vl-4b-instruct-current.{txt,json}`)
