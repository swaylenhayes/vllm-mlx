# Known-Good Model And Profile Matrix

This page is the quickest way to choose a practical serving profile for the
current fork.

Use it together with:
- [Fork Operator Guide](fork-operator-guide.md)
- [OpenAI-Compatible Server](server.md)

Important rules:
- treat this as an operator recommendation layer, not a universal benchmark law
- for fair comparisons, keep the serve profile fixed and vary only the request
  payload
- `enable_thinking` is a request field, not a serve flag
- multimodal `enable_thinking` behavior remains template-dependent even though
  the backend now forwards it

## Quick Picks

| Workload | Recommended model family | Recommended profile | Why |
|---|---|---|---|
| Daily local text chat | Qwen3 text instruct models | `text-default` | Simple, predictable, low-friction local serving |
| Repro or bug triage | Qwen3 text instruct models | `text-deterministic` | Greedy sampling and serialized tracked routes |
| Tool-calling | Qwen3 text instruct or validated tool-friendly models | `text-tools` | Keeps tool parser setup explicit |
| JSON extraction | Qwen3 text instruct / MoE instruct models | `text-json` | Long timeout and stable text-only extraction profile |
| Image or mixed image+text chat | Qwen3-VL family and similar MLLMs | `mllm-default` | Simple runtime avoids extra batching noise |
| Correctness-sensitive multimodal concurrency | Qwen3-VL family and similar MLLMs | `mllm-correctness` | Divergence monitor plus serialize fallback |

## Current Launcher Profiles

After creating the checkout venv, run:

```bash
scripts/serve_profile.sh <profile> <model>
```

Examples:

```bash
scripts/serve_profile.sh text-default mlx-community/Qwen3-4B-Instruct-2507-4bit
scripts/serve_profile.sh text-json mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
scripts/serve_profile.sh mllm-default mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit
```

## Detailed Matrix

| Profile | Best use | Baseline serve behavior | Request notes | Current guidance |
|---|---|---|---|---|
| `text-default` | General local text serving | `--runtime-mode simple --cache-strategy auto --default-temperature 0.7 --default-top-p 0.9` | Set `temperature`, `top_p`, and `max_tokens` per request when needed | Good default for single-user local serving |
| `text-deterministic` | Repro runs, fairness checks, output-shape debugging | `--deterministic` | Prefer `temperature=0`, `top_p=1`, fixed prompts | Use when repeatability matters more than throughput |
| `text-tools` | Function/tool calling | `--runtime-mode auto --cache-strategy auto --enable-auto-tool-choice --tool-call-parser auto` | Send `tools` and `tool_choice` explicitly | Use with validated tool-calling models and parser support |
| `text-json` | JSON mode, extraction, long-running structured output | `--runtime-mode simple --cache-strategy auto --timeout 900 --default-temperature 0.7 --default-top-p 0.9` | Prefer `response_format={"type":"json_object"}` or `json_schema` | Recommended starting point for strict extraction tasks |
| `mllm-default` | Single-user multimodal serving | `--runtime-mode simple --cache-strategy auto --mllm --default-temperature 0.0 --default-top-p 1.0` | Use OpenAI-style multimodal `messages[].content[]` blocks | Best first stop for image and mixed image+text use |
| `mllm-correctness` | Multimodal concurrency with correctness bias | `--runtime-mode auto --cache-strategy auto --mllm --batch-divergence-monitor --batch-divergence-threshold 0.95 --batch-divergence-action serialize` | Keep prompts controlled and compare outputs under repeated load | Use when concurrency matters but batch divergence is unacceptable |

## Thinking And Non-Thinking Guidance

The backend now accepts request-level `enable_thinking` on chat requests.

Use it like this:

```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "Return one short answer."}],
  "enable_thinking": false,
  "temperature": 0.0,
  "top_p": 1.0,
  "max_tokens": 32
}
```

Current practical interpretation:
- text chat paths now honor request-level `enable_thinking`
- multimodal chat paths now receive the same flag, but the model template must
  actually respect it
- if you are comparing thinking vs non-thinking behavior, do not change the
  serve profile between runs

## Multimodal Detection Guidance

Do not rely only on repo names.

This fork now detects many multimodal repos from metadata, including cached
models whose names do not include `-VL`.

Current practical rule:
- if the repo is truly multimodal, the backend should classify it correctly
- if you are using an older install and a model fails to load on the text path,
  retry from the latest checkout or force `--mllm`

## Evidence Notes

The current recommendations are based on shipped fork behavior and local
validation already recorded in:
- [Fork Benefits](../benchmarks/fork-benefits.md)
- [Fork Improvement Log](../benchmarks/fork-improvement-log.md)
- [Fork Operator Guide](fork-operator-guide.md)

This page should evolve as more public benchmark and compatibility data is
published.
