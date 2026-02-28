# vLLM-MLX

**Faster, more reliable local inference on Apple Silicon, with evidence-backed client compatibility and operator-first serving workflows.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)](https://support.apple.com/en-us/HT211814)
[![GitHub](https://img.shields.io/badge/GitHub-swaylenhayes%2Fvllm--mlx-blue?logo=github)](https://github.com/swaylenhayes/vllm-mlx)

This project builds on [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) and pushes it toward a more practical Apple Silicon backend for real local workflows. It grew out of systematic model evaluation work that repeatedly surfaced inference-engine defects and runtime-policy issues as practical blockers for otherwise promising local models.

## At A Glance

| | What | Why it matters |
|---|---|---|
| **Performance** | `+50.25%` token throughput vs upstream baseline | Measurably faster local serving on Apple Silicon |
| **Client compatibility** | Goose and Open WebUI evidence-backed; Jan and AnythingLLM queued next | Real tools connect today without guessing the backend contract |
| **Tool reliability** | Thinking-model tools: `6/9 -> 9/9`. MLLM tools: `0/9 -> 9/9` | Agent workflows that previously failed now work |
| **Serving ergonomics** | Validated profiles for text, deterministic, tools, JSON, and multimodal | Shorter path from clone to working backend |
| **Runtime controls** | Divergence monitoring, strict model id, `frequency_penalty`, `enable_thinking` | Better debugging and safer correctness-sensitive operation |
| **Upstream leverage** | Useful upstream fixes integrated; fork-side hardening ships faster | Faster iteration without cutting off upstream value |

Benchmarked on Apple Silicon using `mlx-community/Qwen3-0.6B-8bit`. Full configuration appears in [Detailed Benchmarks And Validation](#detailed-benchmarks-and-validation).

## Quick Start

Install and serve in two commands:

```bash
# Install
uv tool install git+https://github.com/swaylenhayes/vllm-mlx.git

# Serve (single user, max throughput)
vllm-mlx serve mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8000
```

Or with continuous batching for multiple users:

```bash
vllm-mlx serve mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8000 --continuous-batching
```

For the full operator workflow, see the [Fork Operator Guide](docs/guides/fork-operator-guide.md).

## Serving Profiles

For validated serving profiles with tuned defaults, use the launcher scripts from a git checkout:

```bash
# Daily text serving
scripts/serve_profile.sh text-default mlx-community/Qwen3-4B-Instruct-2507-4bit

# Connect Goose
scripts/serve_client_profile.sh goose-text mlx-community/Qwen3-4B-Instruct-2507-4bit

# Connect Open WebUI
scripts/serve_client_profile.sh open-webui-text mlx-community/Qwen3-4B-Instruct-2507-4bit
```

## Compatibility

| Target | Status | Validated capabilities |
|---|---|---|
| **Goose** | evidence-backed | text, streaming, system prompt, tools, auth, strict model id |
| **Open WebUI** | evidence-backed | text, streaming, system prompt, multimodal, auth |
| **Jan** | queued next | checklist, corpus, and guide ready |
| **AnythingLLM** | queued next | checklist, corpus, and guide ready |

Open WebUI note:
- tool use remains `conditional` until the backend OpenAI tool-call request shape is independently captured

Details: [Client Compatibility Guide](docs/guides/client-compatibility.md) · [Client Settings Crosswalk](docs/guides/client-settings-crosswalk.md)

## What This Project Does Differently

**Reliability hardening.** Clearer startup and runtime behavior, better defaults, explicit operator guidance, and deterministic serving paths for correctness-sensitive work.

**Evidence-backed compatibility.** Client validation against actual frontends and agent tools, not just raw endpoint claims. Each client status is backed by a documented test protocol.

**Known-good serving paths.** Profile and launcher helpers for common Mac-local workflows. Pick a profile, start the backend, connect a client.

**Observability and control.** Divergence monitoring with confidence intervals, strict model-id enforcement, request-level reasoning control, and `frequency_penalty` mapping.

**Stronger tool and multimodal support.** Tool-calling reliability improvements across thinking models and VLMs, metadata-based multimodal detection, and proprietary format parser support (`LiquidAI` / `WaveCut`).

**Selective upstream sync.** Upstream fixes are integrated where they help. Missing features and hardening ship faster on the fork side.

## Detailed Benchmarks And Validation

### Throughput

Benchmark configuration: `mlx-community/Qwen3-0.6B-8bit`, 10 prompts, `max_tokens=64`, `max_num_seqs=32`, `prefill_batch_size=8`, `completion_batch_size=16`. Measured 2026-02-24.

| Snapshot | Commit | Total time (s) | Prompts/s | Tokens/s | Throughput (tok/s) |
|---|---:|---:|---:|---:|---:|
| Upstream baseline | `1fd1c9a` | 1.94 | 5.16 | 330.11 | 366.22 |
| Early fork hardening | `a00ec35` | 1.31 | 7.62 | 487.72 | 541.06 |
| Current published fork | `26b143b` | 1.29 | 7.75 | 496.00 | 550.25 |

| Comparison | Total time | Prompts/s | Tokens/s | Throughput |
|---|---:|---:|---:|---:|
| Early fork hardening vs upstream | -32.47% | +47.67% | +47.74% | +47.74% |
| Current published fork vs upstream | -33.51% | +50.19% | +50.25% | +50.25% |
| Current published fork vs early hardening | -1.53% | +1.71% | +1.70% | +1.70% |

### Batch Divergence (Reliability)

Repeated-run protocol: 5 runs, 95% CI, concurrency=2, max_tokens=32.

| Model | Token agreement (mean, 95% CI) | Exact match (mean) | Verdict |
|---|---:|---:|---|
| Qwen3-4B-Instruct-2507-4bit | 97.86% (97.86-97.86) | 80.00% | Passes >=95% token gate |
| ZwZ-8B-VL-MLX-4bit | 68.65% (65.82-71.49) | 34.00% | Severe divergence |
| Qwen3-VL-30B-A3B-Instruct-4bit | 63.85% (57.20-70.50) | 32.00% | Severe divergence |

Recommended runtime policy:
- Default: `--batch-divergence-threshold 0.95 --batch-divergence-action warn`
- Correctness-sensitive VLM workloads: `--batch-divergence-action serialize`

### Deterministic Profile

Workload: 10 prompts, max_tokens=64, concurrency=10 (Qwen3-4B-Instruct-2507-4bit).

| Profile | Total time (s) | Prompts/s | Tokens/s (completion) |
|---|---:|---:|---:|
| Default | 6.95 | 1.44 | 92.04 |
| Deterministic (`--deterministic`) | 7.16 | 1.40 | 89.44 |

The deterministic path costs only -2.82% throughput vs default, which is a small price for reproducible output in debugging and correctness-sensitive workflows.

### Capability Deltas

| Area | Before | After |
|---|---|---|
| Thinking-model tool reliability | 6/9 ceiling | 9/9 in validated set |
| MLLM tool-calling | 0/9 on validated VLMs | 9/9 on validated VLMs |
| LiquidAI/WaveCut parsing | Unparsed proprietary format | Parser aliases `liquidai` / `liquid` / `lfm` |
| Decode controls | No frequency control | `frequency_penalty` mapped to repetition penalty |
| Model ID policy | Passthrough only | Optional strict enforcement via `--strict-model-id` |
| Reasoning control | Profile-level only | Request-level `enable_thinking` |
| Multimodal detection | Repo-name heuristics | Metadata-based detection |
| Client launch ergonomics | Manual flag mapping | Published profile and client launchers |

### Upstream Integration

Upstream fixes are integrated selectively. The latest upstream sync includes MLLM serialization hardening (`exclude_none=True`), tool-call argument coercion, UTF-8 streaming decode, and MLLM prefill override clarity. Full validation snapshots (1000+ tests passing) are recorded in the improvement log.

## Guides And References

**Operator:** [Fork Operator Guide](docs/guides/fork-operator-guide.md) · [Known-Good Model/Profile Matrix](docs/guides/model-profile-matrix.md) · [Client Compatibility](docs/guides/client-compatibility.md) · [Client Settings Crosswalk](docs/guides/client-settings-crosswalk.md)

**Evidence:** [Fork Benefits](docs/benchmarks/fork-benefits.md) · [Improvement Log](docs/benchmarks/fork-improvement-log.md) · Phase artifacts: `benchmarks/phase-results/`

**Platform:** [Installation](docs/getting-started/installation.md) · [Quick Start](docs/getting-started/quickstart.md) · [Server Guide](docs/guides/server.md) · [Anthropic Messages API](docs/guides/server.md#anthropic-messages-api) · [Multimodal](docs/guides/multimodal.md) · [Audio](docs/guides/audio.md) · [Embeddings](docs/guides/embeddings.md) · [Reasoning Models](docs/guides/reasoning.md) · [MCP & Tool Calling](docs/guides/mcp-tools.md) · [Continuous Batching](docs/guides/continuous-batching.md) · [CLI Reference](docs/reference/cli.md) · [Supported Models](docs/reference/models.md) · [Configuration](docs/reference/configuration.md) · [LLM Benchmarks](docs/benchmarks/llm.md) · [Image Benchmarks](docs/benchmarks/image.md) · [Video Benchmarks](docs/benchmarks/video.md) · [Audio Benchmarks](docs/benchmarks/audio.md)

---

## Platform

vLLM-MLX brings native Apple Silicon GPU acceleration to vLLM-style inference by integrating [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm), [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), [mlx-audio](https://github.com/Blaizzy/mlx-audio), and [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings). It supports text, image, video, and audio inference with OpenAI and Anthropic API compatibility, continuous batching, paged KV cache, MCP tool calling, reasoning model support, and native TTS in 10+ languages.

For full upstream platform documentation, see the original project: [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx).

### Use With OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Use With Anthropic SDK / Claude Code

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000", api_key="not-needed")

response = client.messages.create(
    model="default",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
```

```bash
# Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           vLLM API Layer                                │
│                    (OpenAI + Anthropic compatible)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            MLXPlatform                                  │
│               (vLLM platform plugin for Apple Silicon)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
        ┌─────────────┬────────────┴────────────┬─────────────┐
        ▼             ▼                         ▼             ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│    mlx-lm     │ │   mlx-vlm     │ │   mlx-audio   │ │mlx-embeddings │
│(LLM inference)│ │ (Vision+LLM)  │ │  (TTS + STT)  │ │ (Embeddings)  │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              MLX                                        │
│                (Apple ML Framework - Metal kernels)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Contributing

Contributions welcome. See the [Contributing Guide](docs/development/contributing.md) for details.

Areas where help is especially useful: performance benchmarks on different Apple Silicon chips, client compatibility validation (especially Jan and AnythingLLM), documentation improvements, and bug fixes.

Submit PRs to: [github.com/swaylenhayes/vllm-mlx](https://github.com/swaylenhayes/vllm-mlx)

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Citation

If you use this project in your research, please cite the original:

```bibtex
@software{vllm_mlx2025,
  author = {Barrios, Wayner},
  title = {vLLM-MLX: Apple Silicon MLX Backend for vLLM},
  year = {2025},
  url = {https://github.com/waybarrios/vllm-mlx},
  note = {Native GPU-accelerated LLM and vision-language model inference on Apple Silicon}
}
```

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — LLM inference library
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — Vision-language models
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — Text-to-Speech and Speech-to-Text
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) — Text embeddings
- [vLLM](https://github.com/vllm-project/vllm) — High-throughput LLM serving
- [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) — Original project this work builds on
