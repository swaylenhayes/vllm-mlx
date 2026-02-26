# vLLM-MLX Fork
(swaylenhayes/vllm-mlx)

**vLLM-like inference for Apple Silicon** - GPU-accelerated Text, Image, Video & Audio on Mac

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)](https://support.apple.com/en-us/HT211814)
[![GitHub](https://img.shields.io/badge/GitHub-swaylenhayes%2Fvllm--mlx-blue?logo=github)](https://github.com/swaylenhayes/vllm-mlx)

## Focus of this fork

This repository is a fork of [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx), focused on backend optimization and reliability improvements for local Apple Silicon inference.

Current fork scope:

- **P0**: API contract and reliability hardening
- **P1**: runtime mode policy, startup diagnostics, cache policy defaults, and capabilities contract helpers

## Highlights so far

- **Throughput improvement**: P1 shows **+50.25%** token throughput vs upstream baseline (`366.22 -> 550.25 tok/s`).
- **Tool-calling reliability**:
  - Thinking-model set improved from **6/9 -> 9/9** after P1.10 + follow-up hardening.
  - MLLM tool-calling improved from **0/9 -> 9/9** on two validated VLMs.
- **Batch-divergence observability**:
  - Repeated-run R2C protocol shipped and published with confidence intervals.
  - Text profile cleared token-agreement gate; tested VLM profiles remained well below threshold in batched mode.
- **Upstream sync**:
  - Pulled upstream MLLM serialization fix to exclude `None` fields and avoid null-key template/schema issues.

## Milestone: P0/P1 Performance

Benchmark configuration:
- Model: `mlx-community/Qwen3-0.6B-8bit`
- Command: `vllm-mlx bench ... --max-tokens 64 --max-num-seqs 32 --prefill-batch-size 8 --completion-batch-size 16`
- Prompts: 10
- Date: 2026-02-24

| Phase | Commit | Total time (s) | Prompts/s | Tokens/s | Throughput (tok/s) |
|---|---:|---:|---:|---:|---:|
| upstream baseline | `1fd1c9a` | 1.94 | 5.16 | 330.11 | 366.22 |
| P0 | `a00ec35` | 1.31 | 7.62 | 487.72 | 541.06 |
| P1 | `26b143b` | 1.29 | 7.75 | 496.00 | 550.25 |

| Comparison | Total time | Prompts/s | Tokens/s | Throughput |
|---|---:|---:|---:|---:|
| P0 vs upstream | -32.47% | +47.67% | +47.74% | +47.74% |
| P1 vs upstream | -33.51% | +50.19% | +50.25% | +50.25% |
| P1 vs P0 | -1.53% | +1.71% | +1.70% | +1.70% |

## Milestone: Reliability (R2A/R2B/R2C)

R2C repeated-run batch divergence results (`runs=5`, `95% CI`, `concurrency=2`, `max_tokens=32`):

| Model | Token agreement (mean, 95% CI) | Exact match (mean) | Verdict |
|---|---:|---:|---|
| Qwen3-4B-Instruct-2507-4bit | `97.86%` (`97.86-97.86`) | `80.00%` | Passes `>=95%` token gate |
| ZwZ-8B-VL-MLX-4bit | `68.65%` (`65.82-71.49`) | `34.00%` | Severe divergence |
| Qwen3-VL-30B-A3B-Instruct-4bit | `63.85%` (`57.20-70.50`) | `32.00%` | Severe divergence |

Recommended runtime policy:
- Default monitor profile: `--batch-divergence-threshold 0.95 --batch-divergence-action warn`
- Correctness-sensitive VLM workloads: `--batch-divergence-action serialize` (or simple runtime path)

## Milestone: Compatibility and Tool Calling

| Area | Before | Fork outcome |
|---|---|---|
| Thinking-model tool reliability | `6/9` ceiling in validation probes | Reached `9/9` in validated set after P1.10 + hardening |
| MLLM tool-calling | `0/9` on validated VLMs | `9/9` on validated VLMs after I7 |
| LiquidAI/WaveCut parsing | Unparsed proprietary tool-call format | Added parser aliases `liquidai` / `liquid` / `lfm` |
| Decode controls | No OpenAI-style frequency control | Added `frequency_penalty` mapping to repetition penalty |

## Milestone: Upstream Sync

Latest integrated upstream maintenance:
- `f514235` (cherry-pick of upstream `6d55631`)
- MLLM message/content-part serialization now excludes `None` fields (`exclude_none=True`) to prevent null-key template misinterpretation and strict-client schema failures.

## Detailed references

- Fork compatibility and reliability detail: [`docs/benchmarks/fork-benefits.md`](docs/benchmarks/fork-benefits.md)
- Append-only measured change log: [`docs/benchmarks/fork-improvement-log.md`](docs/benchmarks/fork-improvement-log.md)
- Phase artifacts: `benchmarks/phase-results/`

> [!NOTE]
> End of fork-specific summary. Sections below this point are from the upstream project overview and may not reflect fork-specific deltas documented above.

## Overview

vllm-mlx brings native Apple Silicon GPU acceleration to vLLM by integrating:

- **[MLX](https://github.com/ml-explore/mlx)**: Apple's ML framework with unified memory and Metal kernels
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)**: Optimized LLM inference with KV cache and quantization
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)**: Vision-language models for multimodal inference
- **[mlx-audio](https://github.com/Blaizzy/mlx-audio)**: Speech-to-Text and Text-to-Speech with native voices
- **[mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)**: Text embeddings for semantic search and RAG

## Features

- **Multimodal** - Text, Image, Video & Audio in one platform
- **Native GPU acceleration** on Apple Silicon (M1, M2, M3, M4)
- **Native TTS voices** - Spanish, French, Chinese, Japanese + 5 more languages
- **OpenAI API compatible** - drop-in replacement for OpenAI client
- **Anthropic Messages API** - native `/v1/messages` endpoint for Claude Code and OpenCode
- **Embeddings** - OpenAI-compatible `/v1/embeddings` endpoint with mlx-embeddings
- **Reasoning Models** - extract thinking process from Qwen3, DeepSeek-R1
- **MCP Tool Calling** - integrate external tools via Model Context Protocol
- **Paged KV Cache** - memory-efficient caching with prefix sharing
- **Continuous Batching** - high throughput for multiple concurrent users

## Quick Start

### Installation

**Using uv (recommended):**

```bash
# Install as CLI tool (system-wide)
uv tool install git+https://github.com/swaylenhayes/vllm-mlx.git

# Or install in a project/virtual environment
uv pip install git+https://github.com/swaylenhayes/vllm-mlx.git
```

**Using pip:**

```bash
# Install from GitHub
pip install git+https://github.com/swaylenhayes/vllm-mlx.git

# Or clone and install in development mode
git clone https://github.com/swaylenhayes/vllm-mlx.git
cd vllm-mlx
pip install -e .
```

### Start Server

```bash
# Simple mode (single user, max throughput)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# Continuous batching (multiple users)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching

# With API key authentication
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --api-key your-secret-key
```

### Use with OpenAI SDK

```python
from openai import OpenAI

# Without API key (local development)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# With API key (production)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-secret-key")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Use with Anthropic SDK

vllm-mlx exposes an Anthropic-compatible `/v1/messages` endpoint, so tools like Claude Code and OpenCode can connect directly.

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

To use with Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

See [Anthropic Messages API docs](docs/guides/server.md#anthropic-messages-api) for streaming, tool calling, system messages, and token counting.

### Multimodal (Images & Video)

```bash
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }]
)
```

### Audio (TTS/STT)

```bash
# Install audio dependencies
pip install vllm-mlx[audio]
python -m spacy download en_core_web_sm
brew install espeak-ng  # macOS, for non-English languages
```

```bash
# Text-to-Speech (English)
python examples/tts_example.py "Hello, how are you?" --play

# Text-to-Speech (Spanish)
python examples/tts_multilingual.py "Hola mundo" --lang es --play

# List available models and languages
python examples/tts_multilingual.py --list-models
python examples/tts_multilingual.py --list-languages
```

**Supported TTS Models:**
| Model | Languages | Description |
|-------|-----------|-------------|
| Kokoro | EN, ES, FR, JA, ZH, IT, PT, HI | Fast, 82M params, 11 voices |
| Chatterbox | 15+ languages | Expressive, voice cloning |
| VibeVoice | EN | Realtime, low latency |
| VoxCPM | ZH, EN | High quality Chinese/English |

### Reasoning Models

Extract the thinking process from reasoning models like Qwen3 and DeepSeek-R1:

```bash
# Start server with reasoning parser
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 17 × 23?"}]
)

# Access reasoning separately from the answer
print("Thinking:", response.choices[0].message.reasoning)
print("Answer:", response.choices[0].message.content)
```

**Supported Parsers:**
| Parser | Models | Description |
|--------|--------|-------------|
| `qwen3` | Qwen3 series | Requires both `<think>` and `</think>` tags |
| `deepseek_r1` | DeepSeek-R1 | Handles implicit `<think>` tag |

### Embeddings

Generate text embeddings for semantic search, RAG, and similarity:

```bash
# Start server with an embedding model pre-loaded
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

```python
# Generate embeddings using the OpenAI SDK
embeddings = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input=["Hello world", "How are you?"]
)
print(f"Dimensions: {len(embeddings.data[0].embedding)}")
```

See [Embeddings Guide](docs/guides/embeddings.md) for details on supported models and lazy loading.

## Documentation

For full documentation, see the [docs](docs/) directory:

- **Getting Started**
  - [Installation](docs/getting-started/installation.md)
  - [Quick Start](docs/getting-started/quickstart.md)

- **User Guides**
  - [OpenAI-Compatible Server](docs/guides/server.md)
  - [Anthropic Messages API](docs/guides/server.md#anthropic-messages-api)
  - [Python API](docs/guides/python-api.md)
  - [Multimodal (Images & Video)](docs/guides/multimodal.md)
  - [Audio (STT/TTS)](docs/guides/audio.md)
  - [Embeddings](docs/guides/embeddings.md)
  - [Reasoning Models](docs/guides/reasoning.md)
  - [MCP & Tool Calling](docs/guides/mcp-tools.md)
  - [Continuous Batching](docs/guides/continuous-batching.md)

- **Reference**
  - [CLI Commands](docs/reference/cli.md)
  - [Supported Models](docs/reference/models.md)
  - [Configuration](docs/reference/configuration.md)

- **Benchmarks**
  - [LLM Benchmarks](docs/benchmarks/llm.md)
  - [Image Benchmarks](docs/benchmarks/image.md)
  - [Video Benchmarks](docs/benchmarks/video.md)
  - [Audio Benchmarks](docs/benchmarks/audio.md)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           vLLM API Layer                                │
│                    (OpenAI-compatible interface)                         │
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
        │             │                         │             │
        └─────────────┴─────────────────────────┴─────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              MLX                                        │
│                (Apple ML Framework - Metal kernels)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Performance

**LLM Performance (M4 Max, 128GB):**

| Model | Speed | Memory |
|-------|-------|--------|
| Qwen3-0.6B-8bit | 402 tok/s | 0.7 GB |
| Llama-3.2-1B-4bit | 464 tok/s | 0.7 GB |
| Llama-3.2-3B-4bit | 200 tok/s | 1.8 GB |

**Continuous Batching (5 concurrent requests):**

| Model | Single | Batched | Speedup |
|-------|--------|---------|---------|
| Qwen3-0.6B-8bit | 328 tok/s | 1112 tok/s | **3.4x** |
| Llama-3.2-1B-4bit | 299 tok/s | 613 tok/s | **2.0x** |

**Audio - Speech-to-Text (M4 Max, 128GB):**

| Model | RTF* | Use Case |
|-------|------|----------|
| whisper-tiny | **197x** | Real-time, low latency |
| whisper-large-v3-turbo | **55x** | Best quality/speed balance |
| whisper-large-v3 | **24x** | Highest accuracy |

*RTF = Real-Time Factor. RTF of 100x means 1 minute transcribes in ~0.6 seconds.

See [benchmarks](docs/benchmarks/) for detailed results.

## Gemma 3 Support

vllm-mlx includes native support for Gemma 3 vision models. Gemma 3 is automatically detected as MLLM.

### Usage

```bash
# Start server with Gemma 3
vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000

# Verify it loaded as MLLM (not LLM)
curl http://localhost:8000/health
# Should show: "model_type": "mllm"
```

### Long Context Patch (mlx-vlm)

Gemma 3's default `sliding_window=1024` limits context to ~10K tokens on Apple Silicon (Metal GPU timeout at higher context). To enable longer context (up to ~50K tokens), patch mlx-vlm:

**Location:** `~/.../site-packages/mlx_vlm/models/gemma3/language.py`

Find the `make_cache` method and replace with:

```python
def make_cache(self):
    import os
    # Set GEMMA3_SLIDING_WINDOW=8192 for ~40K context
    # Set GEMMA3_SLIDING_WINDOW=0 for ~50K context (full KVCache)
    sliding_window = int(os.environ.get('GEMMA3_SLIDING_WINDOW', self.config.sliding_window))

    caches = []
    for i in range(self.config.num_hidden_layers):
        if (
            i % self.config.sliding_window_pattern
            == self.config.sliding_window_pattern - 1
        ):
            caches.append(KVCache())
        elif sliding_window == 0:
            caches.append(KVCache())  # Full context for all layers
        else:
            caches.append(RotatingKVCache(max_size=sliding_window, keep=0))
    return caches
```

**Usage:**

```bash
# Default (~10K max context)
vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000

# Extended context (~40K max)
GEMMA3_SLIDING_WINDOW=8192 vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000

# Maximum context (~50K max)
GEMMA3_SLIDING_WINDOW=0 vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000
```

**Benchmark Results (M4 Max 128GB):**

| Setting | Max Context | Memory |
|---------|-------------|--------|
| Default (1024) | ~10K tokens | ~16GB |
| `GEMMA3_SLIDING_WINDOW=8192` | ~40K tokens | ~25GB |
| `GEMMA3_SLIDING_WINDOW=0` | ~50K tokens | ~35GB |

## Contributing

We welcome contributions! See [Contributing Guide](docs/development/contributing.md) for details.

- Bug fixes and improvements
- Performance optimizations
- Documentation improvements
- Benchmarks on different Apple Silicon chips

Submit PRs to: [https://github.com/swaylenhayes/vllm-mlx](https://github.com/swaylenhayes/vllm-mlx)

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Citation

If you use vLLM-MLX in your research or project, please cite:

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

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - LLM inference library
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision-language models
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) - Text-to-Speech and Speech-to-Text
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) - Text embeddings
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
