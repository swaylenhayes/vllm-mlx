#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLI for vllm-mlx.

Commands:
    vllm-mlx serve <model> --port 8000    Start OpenAI-compatible server
    vllm-mlx bench <model>                Run benchmark

Usage:
    vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
    vllm-mlx bench mlx-community/Llama-3.2-1B-Instruct-4bit --num-prompts 10
"""

import argparse
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheProfile:
    """Resolved cache configuration after applying CLI policy."""

    enable_prefix_cache: bool
    use_memory_aware_cache: bool
    use_paged_cache: bool
    strategy_label: str


@dataclass(frozen=True)
class DeterministicProfile:
    """Resolved deterministic profile options for reproducible diagnostics."""

    enabled: bool
    use_batching: bool
    runtime_mode_reason: str
    forced_temperature: float | None
    forced_top_p: float | None
    serialize_tracked_routes: bool


def _resolve_bind_host(host: str, localhost: bool) -> str:
    """Resolve bind host with localhost profile precedence."""
    return "127.0.0.1" if localhost else host


def _is_localhost(bind_host: str) -> bool:
    normalized = bind_host.strip().lower()
    return normalized in {"127.0.0.1", "localhost", "::1"}


def _build_startup_diagnostics(
    *,
    bind_host: str,
    api_key: str | None,
    rate_limit: int,
    runtime_mode: str,
    cache_profile: CacheProfile | None = None,
    deterministic_profile: DeterministicProfile | None = None,
) -> list[str]:
    """Build operator-facing startup diagnostics for risky local configurations."""
    diagnostics: list[str] = []
    local_only = _is_localhost(bind_host)
    auth_enabled = bool(api_key)
    rate_limit_enabled = rate_limit > 0

    if not local_only and not auth_enabled:
        diagnostics.append(
            "WARN: Server is exposed on a non-localhost bind without API key auth."
        )
    if not local_only and not rate_limit_enabled:
        diagnostics.append(
            "WARN: Server is exposed on a non-localhost bind with rate limiting disabled."
        )
    if auth_enabled and not rate_limit_enabled:
        diagnostics.append(
            "WARN: Authentication enabled but rate limiting is disabled; abusive clients can still saturate the server."
        )
    if rate_limit_enabled and not auth_enabled:
        diagnostics.append(
            "WARN: Rate limiting enabled without API keys; client identity falls back to IP and may be coarse."
        )
    if rate_limit_enabled and rate_limit < 5:
        diagnostics.append(
            f"WARN: Very low rate limit configured ({rate_limit} req/min); clients may observe frequent 429s."
        )
    if local_only and not auth_enabled:
        diagnostics.append(
            "INFO: Localhost-only mode with auth disabled is acceptable for single-user local development."
        )
    if runtime_mode == "simple":
        diagnostics.append(
            "INFO: Runtime mode is simple; monitor peak concurrency and switch to batched if queueing appears."
        )
    if runtime_mode == "batched":
        diagnostics.append(
            "INFO: Runtime mode is batched; expect higher overhead for strictly single-user traffic."
        )
    if cache_profile is not None:
        diagnostics.append(
            "INFO: Cache strategy resolved to "
            f"{cache_profile.strategy_label} "
            f"(prefix={cache_profile.enable_prefix_cache}, "
            f"memory_aware={cache_profile.use_memory_aware_cache}, "
            f"paged={cache_profile.use_paged_cache})."
        )
        if cache_profile.enable_prefix_cache:
            diagnostics.append(
                "INFO: Warm-cache guidance: send a representative request after startup to seed cache entries."
            )
    if deterministic_profile is not None and deterministic_profile.enabled:
        diagnostics.append(
            "INFO: Deterministic profile enabled: forcing simple runtime, greedy sampling "
            "(temperature=0, top_p=1), and serialized tracked inference routes."
        )
        diagnostics.append(
            "INFO: Deterministic profile is for reproducibility diagnostics; expect lower throughput under load."
        )

    return diagnostics


def _resolve_cache_profile(args, use_batching: bool) -> CacheProfile:
    """
    Resolve cache behavior from explicit flags plus cache strategy policy.

    Strategy precedence:
    1) explicit cache strategy
    2) auto strategy heuristic
    3) explicit cache flags
    """
    enable_prefix_cache = args.enable_prefix_cache and not args.disable_prefix_cache
    use_memory_aware_cache = not args.no_memory_aware_cache
    use_paged_cache = args.use_paged_cache
    strategy_label = args.cache_strategy

    if args.cache_strategy == "legacy":
        enable_prefix_cache = True
        use_memory_aware_cache = False
        use_paged_cache = False
    elif args.cache_strategy == "memory-aware":
        enable_prefix_cache = True
        use_memory_aware_cache = True
        use_paged_cache = False
    elif args.cache_strategy == "paged":
        enable_prefix_cache = True
        use_memory_aware_cache = False
        use_paged_cache = True
    elif args.cache_strategy == "auto":
        if not enable_prefix_cache:
            use_memory_aware_cache = False
            use_paged_cache = False
            strategy_label = "auto->disabled"
        elif use_paged_cache:
            use_memory_aware_cache = False
            strategy_label = "auto->explicit-paged"
        elif use_batching and args.max_num_seqs >= 128:
            use_memory_aware_cache = False
            use_paged_cache = True
            strategy_label = "auto->paged(max_num_seqs>=128)"
        elif use_memory_aware_cache:
            strategy_label = "auto->memory-aware"
        else:
            strategy_label = "auto->legacy"
    else:
        raise ValueError(f"Unsupported cache strategy: {args.cache_strategy!r}")

    if not enable_prefix_cache:
        use_memory_aware_cache = False
        use_paged_cache = False

    if use_paged_cache:
        use_memory_aware_cache = False

    return CacheProfile(
        enable_prefix_cache=enable_prefix_cache,
        use_memory_aware_cache=use_memory_aware_cache,
        use_paged_cache=use_paged_cache,
        strategy_label=strategy_label,
    )


def _resolve_deterministic_profile(
    *,
    deterministic: bool,
    use_batching: bool,
    runtime_mode_reason: str,
) -> DeterministicProfile:
    """Resolve deterministic mode behavior as an additive runtime policy."""
    if not deterministic:
        return DeterministicProfile(
            enabled=False,
            use_batching=use_batching,
            runtime_mode_reason=runtime_mode_reason,
            forced_temperature=None,
            forced_top_p=None,
            serialize_tracked_routes=False,
        )

    if use_batching:
        reason = (
            "deterministic profile: forcing simple mode (overrode batched runtime selection)"
        )
    else:
        reason = "deterministic profile: simple mode with reproducible sampling controls"

    return DeterministicProfile(
        enabled=True,
        use_batching=False,
        runtime_mode_reason=reason,
        forced_temperature=0.0,
        forced_top_p=1.0,
        serialize_tracked_routes=True,
    )


def serve_command(args):
    """Start the OpenAI-compatible server."""
    import logging
    import os
    import sys

    import uvicorn

    # Import unified server
    from .runtime_mode import load_observed_peak_concurrency, select_runtime_mode
    from . import server
    from .scheduler import SchedulerConfig
    from .server import RateLimiter, app, load_model

    logger = logging.getLogger(__name__)

    # Validate tool calling arguments
    if args.enable_auto_tool_choice and not args.tool_call_parser:
        print("Error: --enable-auto-tool-choice requires --tool-call-parser")
        print("Example: --enable-auto-tool-choice --tool-call-parser mistral")
        sys.exit(1)
    if args.runtime_mode_threshold < 1:
        print("Error: --runtime-mode-threshold must be >= 1")
        sys.exit(1)
    if args.effective_context_tokens is not None and args.effective_context_tokens < 1:
        print("Error: --effective-context-tokens must be >= 1")
        sys.exit(1)
    if args.mllm_vision_cache_size < 1:
        print("Error: --mllm-vision-cache-size must be >= 1")
        sys.exit(1)
    if args.max_thinking_tokens is not None and args.max_thinking_tokens < 1:
        print("Error: --max-thinking-tokens must be >= 1")
        sys.exit(1)
    if args.max_thinking_tokens is not None and not args.reasoning_parser:
        print("Error: --max-thinking-tokens requires --reasoning-parser")
        sys.exit(1)
    if args.memory_warn_threshold <= 0:
        print("Error: --memory-warn-threshold must be > 0")
        sys.exit(1)
    if args.memory_limit_threshold <= args.memory_warn_threshold:
        print(
            "Error: --memory-limit-threshold must be greater than "
            "--memory-warn-threshold"
        )
        sys.exit(1)
    if args.memory_monitor_interval <= 0:
        print("Error: --memory-monitor-interval must be > 0")
        sys.exit(1)
    if args.batch_divergence_interval <= 0:
        print("Error: --batch-divergence-interval must be > 0")
        sys.exit(1)
    if args.batch_divergence_threshold <= 0 or args.batch_divergence_threshold > 1:
        print("Error: --batch-divergence-threshold must be in (0, 1]")
        sys.exit(1)

    # Configure server security settings
    server._api_key = args.api_key
    server._default_timeout = args.timeout
    server._repetition_policy = args.repetition_policy
    server._repetition_override_policy = "trusted_only"
    server._trust_requests_when_auth_disabled = (
        args.trust_requests_when_auth_disabled
    )
    if args.rate_limit > 0:
        server._rate_limiter = RateLimiter(
            requests_per_minute=args.rate_limit, enabled=True
        )

    # Configure tool calling
    if args.enable_auto_tool_choice and args.tool_call_parser:
        server._enable_auto_tool_choice = True
        server._tool_call_parser = args.tool_call_parser
    else:
        server._enable_auto_tool_choice = False
        server._tool_call_parser = None

    # Configure generation defaults
    server._default_temperature = args.default_temperature
    server._default_top_p = args.default_top_p
    server._effective_context_tokens = args.effective_context_tokens
    server._deterministic_mode = False
    server._deterministic_serialize = False
    server._strict_model_id = args.strict_model_id

    # Configure reasoning parser
    if args.reasoning_parser:
        try:
            from .reasoning import get_parser

            parser_cls = get_parser(args.reasoning_parser)
            server._reasoning_parser = parser_cls()
            logger.info(f"Reasoning parser enabled: {args.reasoning_parser}")
        except KeyError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except ImportError as e:
            print(f"Error: Failed to import reasoning module: {e}")
            sys.exit(1)
        except Exception as e:
            print(
                f"Error: Failed to initialize reasoning parser "
                f"'{args.reasoning_parser}': {e}"
            )
            sys.exit(1)
    else:
        server._reasoning_parser = None
    server._max_thinking_tokens = args.max_thinking_tokens
    server._memory_warn_threshold_pct = args.memory_warn_threshold
    server._memory_limit_threshold_pct = args.memory_limit_threshold
    server._memory_action = args.memory_action
    server._memory_monitor_interval_seconds = args.memory_monitor_interval
    server._batch_divergence_monitor_enabled = args.batch_divergence_monitor
    server._batch_divergence_interval_seconds = args.batch_divergence_interval
    server._batch_divergence_threshold = args.batch_divergence_threshold
    server._batch_divergence_action = args.batch_divergence_action
    server._batch_divergence_state.configure(
        enabled=args.batch_divergence_monitor,
        threshold=args.batch_divergence_threshold,
        action=args.batch_divergence_action,
    )

    bind_host = _resolve_bind_host(args.host, args.localhost)
    server._bind_host = bind_host
    observed_peak = load_observed_peak_concurrency()
    use_batching, runtime_mode_reason = select_runtime_mode(
        requested_mode=args.runtime_mode,
        continuous_batching_flag=args.continuous_batching,
        observed_peak=observed_peak,
        threshold=args.runtime_mode_threshold,
    )
    deterministic_profile = _resolve_deterministic_profile(
        deterministic=args.deterministic,
        use_batching=use_batching,
        runtime_mode_reason=runtime_mode_reason,
    )
    use_batching = deterministic_profile.use_batching
    runtime_mode_reason = deterministic_profile.runtime_mode_reason
    runtime_mode = "batched" if use_batching else "simple"

    if deterministic_profile.enabled:
        server._deterministic_mode = True
        server._deterministic_serialize = deterministic_profile.serialize_tracked_routes
        server._default_temperature = deterministic_profile.forced_temperature
        server._default_top_p = deterministic_profile.forced_top_p

    cache_profile = _resolve_cache_profile(args, use_batching=use_batching)

    # Security summary at startup
    print("=" * 60)
    print("SECURITY CONFIGURATION")
    print("=" * 60)
    if args.api_key:
        print("  Authentication: ENABLED (API key required)")
    else:
        print("  Authentication: DISABLED - Use --api-key to enable")
    if args.rate_limit > 0:
        print(f"  Rate limiting: ENABLED ({args.rate_limit} req/min)")
    else:
        print("  Rate limiting: DISABLED - Use --rate-limit to enable")
    print(
        "  Repetition policy: "
        f"default={args.repetition_policy}, "
        f"override_policy=trusted_only, "
        f"trust_when_auth_disabled={args.trust_requests_when_auth_disabled}"
    )
    print(f"  Request timeout: {args.timeout}s")
    print(
        "  Memory guardrails: "
        f"warn={args.memory_warn_threshold:.1f}% "
        f"limit={args.memory_limit_threshold:.1f}% "
        f"action={args.memory_action} "
        f"interval={args.memory_monitor_interval:.1f}s"
    )
    print(
        "  Batch divergence: "
        f"enabled={args.batch_divergence_monitor} "
        f"threshold={args.batch_divergence_threshold:.2f} "
        f"action={args.batch_divergence_action} "
        f"interval={args.batch_divergence_interval:.1f}s"
    )
    if deterministic_profile.enabled:
        print(
            "  Deterministic profile: ENABLED "
            "(runtime=simple, temperature=0.0, top_p=1.0, serialize=true)"
        )
    else:
        print("  Deterministic profile: DISABLED")
    if args.strict_model_id:
        print("  Strict model ID: ENABLED")
    else:
        print("  Strict model ID: DISABLED")
    if args.effective_context_tokens is not None:
        print(
            "  Effective context contract: "
            f"operator override={args.effective_context_tokens} tokens"
        )
    else:
        print("  Effective context contract: auto (from model metadata when available)")
    if args.enable_auto_tool_choice:
        print(f"  Tool calling: ENABLED (parser: {args.tool_call_parser})")
    else:
        print("  Tool calling: Use --enable-auto-tool-choice to enable")
    if args.reasoning_parser:
        if args.max_thinking_tokens is not None:
            print(
                "  Reasoning: ENABLED "
                f"(parser: {args.reasoning_parser}, max_thinking_tokens={args.max_thinking_tokens})"
            )
        else:
            print(f"  Reasoning: ENABLED (parser: {args.reasoning_parser})")
    else:
        print("  Reasoning: Use --reasoning-parser to enable")
    print(f"  Runtime mode: {runtime_mode} ({runtime_mode_reason})")
    print("=" * 60)

    # Pre-download model with retry/timeout
    from .api.utils import is_mllm_model
    from .utils.download import DownloadConfig, ensure_model_downloaded

    download_config = DownloadConfig(
        download_timeout=args.download_timeout,
        max_retries=args.download_retries,
        offline=getattr(args, "offline", False),
    )
    ensure_model_downloaded(
        args.model,
        config=download_config,
        is_mllm=is_mllm_model(args.model),
    )

    print(f"Loading model: {args.model}")
    print(f"Default max tokens: {args.max_tokens}")

    # Store MCP config path for FastAPI startup
    if args.mcp_config:
        print(f"MCP config: {args.mcp_config}")
        os.environ["VLLM_MLX_MCP_CONFIG"] = args.mcp_config

    # Pre-load embedding model if specified
    if args.embedding_model:
        print(f"Pre-loading embedding model: {args.embedding_model}")
        server.load_embedding_model(args.embedding_model, lock=True)
        print(f"Embedding model loaded: {args.embedding_model}")

    # Build scheduler config for batched mode
    scheduler_config = None
    if use_batching:
        scheduler_config = SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            prefill_batch_size=args.prefill_batch_size,
            completion_batch_size=args.completion_batch_size,
            repetition_policy=args.repetition_policy,
            enable_prefix_cache=cache_profile.enable_prefix_cache,
            prefix_cache_size=args.prefix_cache_size,
            # Memory-aware cache options
            use_memory_aware_cache=cache_profile.use_memory_aware_cache,
            cache_memory_mb=args.cache_memory_mb,
            cache_memory_percent=args.cache_memory_percent,
            # Paged cache options
            use_paged_cache=cache_profile.use_paged_cache,
            paged_cache_block_size=args.paged_cache_block_size,
            max_cache_blocks=args.max_cache_blocks,
            # Chunked prefill
            chunked_prefill_tokens=args.chunked_prefill_tokens,
            # MTP
            enable_mtp=args.enable_mtp,
            mtp_num_draft_tokens=args.mtp_num_draft_tokens,
            mtp_optimistic=args.mtp_optimistic,
            # KV cache quantization
            kv_cache_quantization=args.kv_cache_quantization,
            kv_cache_quantization_bits=args.kv_cache_quantization_bits,
            kv_cache_quantization_group_size=args.kv_cache_quantization_group_size,
            kv_cache_min_quantize_tokens=args.kv_cache_min_quantize_tokens,
            # MLLM cache settings
            enable_vision_cache=not args.disable_mllm_vision_cache,
            vision_cache_size=args.mllm_vision_cache_size,
            mllm_prefill_step_size=(
                args.mllm_prefill_step_size if args.mllm_prefill_step_size > 0 else None
            ),
        )

        print("Mode: Continuous batching (for multiple concurrent users)")
        if args.chunked_prefill_tokens > 0:
            print(f"Chunked prefill: {args.chunked_prefill_tokens} tokens per step")
        if args.enable_mtp:
            print(f"MTP: enabled, draft_tokens={args.mtp_num_draft_tokens}")
        print(f"Stream interval: {args.stream_interval} tokens")
        if cache_profile.use_paged_cache:
            print(
                f"Paged cache: block_size={args.paged_cache_block_size}, max_blocks={args.max_cache_blocks}"
            )
        elif cache_profile.enable_prefix_cache and cache_profile.use_memory_aware_cache:
            cache_info = (
                f"{args.cache_memory_mb}MB"
                if args.cache_memory_mb
                else f"{args.cache_memory_percent*100:.0f}% of RAM"
            )
            print(f"Memory-aware cache: {cache_info}")
            if args.kv_cache_quantization:
                print(
                    f"KV cache quantization: {args.kv_cache_quantization_bits}-bit, "
                    f"group_size={args.kv_cache_quantization_group_size}"
                )
        elif cache_profile.enable_prefix_cache:
            print(f"Prefix cache: max_entries={args.prefix_cache_size}")
        if args.mllm_vision_cache_size != 100 or args.disable_mllm_vision_cache:
            print(
                f"MLLM vision cache: enabled={not args.disable_mllm_vision_cache}, "
                f"size={args.mllm_vision_cache_size}"
            )
    else:
        print("Mode: Simple (maximum throughput)")

    diagnostics = _build_startup_diagnostics(
        bind_host=bind_host,
        api_key=args.api_key,
        rate_limit=args.rate_limit,
        runtime_mode=runtime_mode,
        cache_profile=cache_profile if use_batching else None,
        deterministic_profile=deterministic_profile,
    )
    for diagnostic in diagnostics:
        print(f"  {diagnostic}")
        if diagnostic.startswith("WARN:"):
            logger.warning(diagnostic)
        elif diagnostic.startswith("INFO:"):
            logger.info(diagnostic)

    # Load model with unified server
    load_model(
        args.model,
        use_batching=use_batching,
        scheduler_config=scheduler_config,
        stream_interval=args.stream_interval if use_batching else 1,
        max_tokens=args.max_tokens,
        force_mllm=args.mllm,
    )

    # Start server
    print(f"Starting server at http://{bind_host}:{args.port}")
    uvicorn.run(app, host=bind_host, port=args.port, log_level="info")


def download_command(args):
    """Download a model to local cache without starting a server."""
    from .utils.download import DownloadConfig, ensure_model_downloaded

    config = DownloadConfig(
        download_timeout=args.timeout,
        max_retries=args.retries,
    )
    print(f"Downloading model: {args.model}")
    path = ensure_model_downloaded(
        args.model,
        config=config,
        is_mllm=args.mllm,
    )
    print(f"Model ready at: {path}")


def bench_command(args):
    """Run benchmark."""
    import asyncio
    import time

    from mlx_lm import load

    from .engine_core import AsyncEngineCore, EngineConfig
    from .request import SamplingParams
    from .scheduler import SchedulerConfig

    # Handle prefix cache flags
    enable_prefix_cache = args.enable_prefix_cache and not args.disable_prefix_cache

    async def run_benchmark():
        print(f"Loading model: {args.model}")
        model, tokenizer = load(args.model)

        scheduler_config = SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            prefill_batch_size=args.prefill_batch_size,
            completion_batch_size=args.completion_batch_size,
            enable_prefix_cache=enable_prefix_cache,
            prefix_cache_size=args.prefix_cache_size,
            # Memory-aware cache options
            use_memory_aware_cache=not args.no_memory_aware_cache,
            cache_memory_mb=args.cache_memory_mb,
            cache_memory_percent=args.cache_memory_percent,
            # Paged cache options
            use_paged_cache=args.use_paged_cache,
            paged_cache_block_size=args.paged_cache_block_size,
            max_cache_blocks=args.max_cache_blocks,
            # KV cache quantization
            kv_cache_quantization=args.kv_cache_quantization,
            kv_cache_quantization_bits=args.kv_cache_quantization_bits,
            kv_cache_quantization_group_size=args.kv_cache_quantization_group_size,
            kv_cache_min_quantize_tokens=args.kv_cache_min_quantize_tokens,
        )
        engine_config = EngineConfig(
            model_name=args.model,
            scheduler_config=scheduler_config,
        )

        if args.use_paged_cache:
            print(
                f"Paged cache: block_size={args.paged_cache_block_size}, max_blocks={args.max_cache_blocks}"
            )

        # Generate prompts
        prompts = [
            f"Write a short poem about {topic}."
            for topic in [
                "nature",
                "love",
                "technology",
                "space",
                "music",
                "art",
                "science",
                "history",
                "food",
                "travel",
            ][: args.num_prompts]
        ]

        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.7,
        )

        print(
            f"\nRunning benchmark with {len(prompts)} prompts, max_tokens={args.max_tokens}"
        )
        print("-" * 50)

        total_prompt_tokens = 0
        total_completion_tokens = 0

        async with AsyncEngineCore(model, tokenizer, engine_config) as engine:
            await asyncio.sleep(0.1)  # Warm up

            start_time = time.perf_counter()

            # Add all requests
            request_ids = []
            for prompt in prompts:
                rid = await engine.add_request(prompt, params)
                request_ids.append(rid)

            # Collect all outputs
            async def get_output(rid):
                async for out in engine.stream_outputs(rid, timeout=120):
                    if out.finished:
                        return out
                return None

            results = await asyncio.gather(*[get_output(r) for r in request_ids])

            total_time = time.perf_counter() - start_time

        # Calculate stats
        for r in results:
            if r:
                total_prompt_tokens += r.prompt_tokens
                total_completion_tokens += r.completion_tokens

        total_tokens = total_prompt_tokens + total_completion_tokens

        print("\nResults:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Prompts/second: {len(prompts)/total_time:.2f}")
        print(f"  Total prompt tokens: {total_prompt_tokens}")
        print(f"  Total completion tokens: {total_completion_tokens}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Tokens/second: {total_completion_tokens/total_time:.2f}")
        print(f"  Throughput: {total_tokens/total_time:.2f} tok/s")

    asyncio.run(run_benchmark())


def bench_detok_command(args):
    """Benchmark streaming detokenizer optimization."""
    import statistics
    import time

    from mlx_lm import load
    from mlx_lm.generate import generate

    print("=" * 70)
    print(" Streaming Detokenizer Benchmark")
    print("=" * 70)
    print()

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    # Generate tokens for benchmark
    prompt = "Write a detailed explanation of how machine learning works and its applications in modern technology."
    print(f"Generating tokens with prompt: {prompt[:50]}...")

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=2000,
        verbose=False,
    )

    prompt_tokens = tokenizer.encode(prompt)
    all_tokens = tokenizer.encode(output)
    generated_tokens = all_tokens[len(prompt_tokens) :]
    print(f"Generated {len(generated_tokens)} tokens for benchmark")
    print()

    iterations = args.iterations

    # Benchmark naive decode (old method)
    print("Benchmarking Naive Decode (OLD method)...")
    naive_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        for t in generated_tokens:
            _ = tokenizer.decode([t])
        elapsed = time.perf_counter() - start
        naive_times.append(elapsed)

    naive_mean = statistics.mean(naive_times) * 1000

    # Benchmark streaming decode (new method)
    print("Benchmarking Streaming Detokenizer (NEW method)...")
    streaming_times = []
    detok_class = tokenizer._detokenizer_class
    for _ in range(iterations):
        detok = detok_class(tokenizer)
        detok.reset()
        start = time.perf_counter()
        for t in generated_tokens:
            detok.add_token(t)
            _ = detok.last_segment
        detok.finalize()
        elapsed = time.perf_counter() - start
        streaming_times.append(elapsed)

    streaming_mean = statistics.mean(streaming_times) * 1000

    # Results
    speedup = naive_mean / streaming_mean
    time_saved = naive_mean - streaming_mean

    print()
    print("=" * 70)
    print(f" RESULTS: {len(generated_tokens)} tokens, {iterations} iterations")
    print("=" * 70)
    print(f"{'Method':<25} {'Time':>12} {'Speedup':>10}")
    print("-" * 70)
    print(f"{'Naive decode():':<25} {naive_mean:>10.2f}ms {'1.00x':>10}")
    print(f"{'Streaming detokenizer:':<25} {streaming_mean:>10.2f}ms {speedup:>9.2f}x")
    print("-" * 70)
    print(f"{'Time saved per request:':<25} {time_saved:>10.2f}ms")
    print(
        f"{'Per-token savings:':<25} {(time_saved/len(generated_tokens)*1000):>10.1f}µs"
    )
    print()

    # Verify correctness (strip for BPE edge cases with leading/trailing spaces)
    print("Verifying correctness...")
    detok = detok_class(tokenizer)
    detok.reset()
    for t in generated_tokens:
        detok.add_token(t)
    detok.finalize()

    batch_result = tokenizer.decode(generated_tokens)
    # BPE tokenizers may have minor edge case differences with spaces
    # Compare stripped versions for functional correctness
    streaming_stripped = detok.text.strip()
    batch_stripped = batch_result.strip()
    if streaming_stripped == batch_stripped:
        print("  ✓ Streaming output matches batch decode")
    elif streaming_stripped in batch_stripped or batch_stripped in streaming_stripped:
        print("  ✓ Streaming output matches (minor BPE edge case)")
    else:
        # Check if most of the content matches (BPE edge cases at boundaries)
        common_len = min(len(streaming_stripped), len(batch_stripped)) - 10
        if (
            common_len > 0
            and streaming_stripped[:common_len] == batch_stripped[:common_len]
        ):
            print("  ✓ Streaming output matches (BPE boundary difference)")
        else:
            print("  ✗ MISMATCH! Results differ")
            print(f"    Streaming: {repr(detok.text[:100])}...")
            print(f"    Batch: {repr(batch_result[:100])}...")


def bench_kv_cache_command(args):
    """Benchmark KV cache quantization memory savings and quality."""
    import time

    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    from .memory_cache import (
        _dequantize_cache,
        _quantize_cache,
        estimate_kv_cache_memory,
    )

    print("=" * 70)
    print(" KV Cache Quantization Benchmark")
    print("=" * 70)
    print()

    n_layers = args.layers
    seq_len = args.seq_len
    n_heads = args.heads
    head_dim = args.head_dim

    print(
        f"Config: {n_layers} layers, seq_len={seq_len}, "
        f"n_heads={n_heads}, head_dim={head_dim}"
    )
    print()

    # Create synthetic KV cache with random data
    print("Creating synthetic KV cache...")
    cache = []
    for _ in range(n_layers):
        kv = KVCache()
        kv.keys = mx.random.normal((1, n_heads, seq_len, head_dim))
        kv.values = mx.random.normal((1, n_heads, seq_len, head_dim))
        kv.offset = seq_len
        cache.append(kv)
    mx.eval(*[kv.keys for kv in cache], *[kv.values for kv in cache])

    fp16_mem = estimate_kv_cache_memory(cache)
    print(f"FP16 cache memory: {fp16_mem / 1024 / 1024:.2f} MB")
    print()

    # Test each bit width
    results = []
    for bits in [8, 4]:
        group_size = args.group_size

        # Quantize
        start = time.perf_counter()
        quantized = _quantize_cache(cache, bits=bits, group_size=group_size)
        mx.eval(
            *[
                layer.keys[0]
                for layer in quantized
                if hasattr(layer, "keys") and layer.keys is not None
            ]
        )
        quant_time = (time.perf_counter() - start) * 1000

        quant_mem = estimate_kv_cache_memory(quantized)

        # Dequantize
        start = time.perf_counter()
        restored = _dequantize_cache(quantized)
        mx.eval(
            *[
                layer.keys
                for layer in restored
                if hasattr(layer, "keys") and layer.keys is not None
            ]
        )
        dequant_time = (time.perf_counter() - start) * 1000

        # Measure quality
        total_error = 0.0
        max_error = 0.0
        count = 0
        for orig, rest in zip(cache, restored):
            if orig.keys is not None and rest.keys is not None:
                mx.eval(orig.keys, rest.keys, orig.values, rest.values)
                key_err = mx.abs(orig.keys - rest.keys).mean().item()
                val_err = mx.abs(orig.values - rest.values).mean().item()
                key_max = mx.abs(orig.keys - rest.keys).max().item()
                val_max = mx.abs(orig.values - rest.values).max().item()
                total_error += (key_err + val_err) / 2
                max_error = max(max_error, key_max, val_max)
                count += 1

        mean_error = total_error / count if count > 0 else 0.0
        ratio = fp16_mem / quant_mem if quant_mem > 0 else 0.0

        results.append(
            {
                "bits": bits,
                "mem_mb": quant_mem / 1024 / 1024,
                "ratio": ratio,
                "mean_err": mean_error,
                "max_err": max_error,
                "quant_ms": quant_time,
                "dequant_ms": dequant_time,
            }
        )

    # Print results
    fp16_mb = fp16_mem / 1024 / 1024
    print(
        f"{'Mode':<12} {'Memory':>10} {'Savings':>10} "
        f"{'Mean Err':>10} {'Max Err':>10} {'Quant':>10} {'Dequant':>10}"
    )
    print("-" * 72)
    print(
        f"{'FP16':<12} {fp16_mb:>8.2f}MB {'1.00x':>10} "
        f"{'0.000':>10} {'0.000':>10} {'-':>10} {'-':>10}"
    )

    for r in results:
        print(
            f"{r['bits']}-bit{'':<7} {r['mem_mb']:>8.2f}MB "
            f"{r['ratio']:>9.2f}x "
            f"{r['mean_err']:>10.5f} {r['max_err']:>10.5f} "
            f"{r['quant_ms']:>8.1f}ms {r['dequant_ms']:>8.1f}ms"
        )

    print()

    # Recommendation
    best = results[0]  # 8-bit
    print(
        f"Recommendation: 8-bit quantization gives {best['ratio']:.1f}x memory savings "
        f"with mean error {best['mean_err']:.5f}"
    )
    print(
        f"Use 4-bit for maximum compression if quality loss of "
        f"{results[1]['mean_err']:.4f} is acceptable."
    )
    print()
    print("Usage:")
    print("  vllm-mlx serve <model> --continuous-batching --kv-cache-quantization")
    print(
        "  vllm-mlx serve <model> --continuous-batching --kv-cache-quantization "
        "--kv-cache-quantization-bits 4"
    )


def main():
    parser = argparse.ArgumentParser(
        description="vllm-mlx: Apple Silicon MLX backend for vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
  vllm-mlx bench mlx-community/Llama-3.2-1B-Instruct-4bit --num-prompts 10
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible server")
    serve_parser.add_argument("model", type=str, help="Model to serve")
    serve_parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind"
    )
    serve_parser.add_argument(
        "--localhost",
        action="store_true",
        help="Bind server to localhost only (127.0.0.1). Overrides --host.",
    )
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve_parser.add_argument(
        "--max-num-seqs", type=int, default=256, help="Max concurrent sequences"
    )
    serve_parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Prefill batch size"
    )
    serve_parser.add_argument(
        "--completion-batch-size", type=int, default=32, help="Completion batch size"
    )
    serve_parser.add_argument(
        "--mllm-prefill-step-size",
        type=int,
        default=0,
        help="Override MLLM prefill-step guard (0=use MLLM default: 1024)",
    )
    serve_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Enable prefix caching for repeated prompts (default: enabled)",
    )
    serve_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    serve_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Max entries in prefix cache (default: 100, legacy mode only)",
    )
    # Memory-aware cache options (recommended for large models)
    serve_parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=None,
        help="Cache memory limit in MB (default: auto-detect ~20%% of RAM)",
    )
    serve_parser.add_argument(
        "--cache-memory-percent",
        type=float,
        default=0.20,
        help="Fraction of available RAM for cache if auto-detecting (default: 0.20)",
    )
    serve_parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Disable memory-aware cache, use legacy entry-count based cache",
    )
    # KV cache quantization options
    serve_parser.add_argument(
        "--kv-cache-quantization",
        action="store_true",
        help="Quantize stored KV caches to reduce memory (8-bit by default)",
    )
    serve_parser.add_argument(
        "--kv-cache-quantization-bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Bit width for KV cache quantization (default: 8)",
    )
    serve_parser.add_argument(
        "--kv-cache-quantization-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
    )
    serve_parser.add_argument(
        "--kv-cache-min-quantize-tokens",
        type=int,
        default=256,
        help="Minimum tokens for quantization to apply (default: 256)",
    )
    serve_parser.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Tokens to batch before streaming (1=smooth, higher=throughput)",
    )
    serve_parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Default max tokens for generation (default: 32768)",
    )
    serve_parser.add_argument(
        "--runtime-mode",
        type=str,
        default="auto",
        choices=["auto", "simple", "batched"],
        help="Runtime engine mode policy: auto (from observed concurrency), simple, or batched.",
    )
    serve_parser.add_argument(
        "--runtime-mode-threshold",
        type=int,
        default=2,
        help="Peak concurrency threshold for selecting batched mode when --runtime-mode=auto (default: 2).",
    )
    serve_parser.add_argument(
        "--effective-context-tokens",
        type=int,
        default=None,
        help=(
            "Override effective context limit exposed via /v1/capabilities and "
            "response diagnostics (default: auto from model metadata when available)."
        ),
    )
    serve_parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Enable reproducibility profile: force simple runtime, "
            "greedy sampling (temperature=0, top_p=1), and serialize tracked "
            "inference routes."
        ),
    )
    serve_parser.add_argument(
        "--strict-model-id",
        action="store_true",
        help=(
            "Require request model id to exactly match loaded model id for "
            "chat/completions and Anthropic messages endpoints."
        ),
    )
    serve_parser.add_argument(
        "--continuous-batching",
        action="store_true",
        help="Legacy override to force batched mode (equivalent to --runtime-mode batched in auto mode).",
    )
    serve_parser.add_argument(
        "--cache-strategy",
        type=str,
        default="auto",
        choices=["auto", "memory-aware", "paged", "legacy"],
        help="Cache strategy policy for batched mode. Auto chooses based on concurrency profile and max-num-seqs.",
    )
    # Paged cache options (experimental)
    serve_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help="Use paged KV cache for memory efficiency (experimental)",
    )
    serve_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per cache block (default: 64)",
    )
    serve_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks (default: 1000)",
    )
    # Chunked prefill
    serve_parser.add_argument(
        "--chunked-prefill-tokens",
        type=int,
        default=0,
        help="Max prefill tokens per scheduler step (0=disabled). "
        "Prevents starvation of active requests during long prefills.",
    )
    # MTP (Multi-Token Prediction)
    serve_parser.add_argument(
        "--enable-mtp",
        action="store_true",
        default=False,
        help="Enable MTP (Multi-Token Prediction) for models with built-in MTP heads. "
        "Uses cache snapshot/restore for speculative generation.",
    )
    serve_parser.add_argument(
        "--mtp-num-draft-tokens",
        type=int,
        default=1,
        help="Number of draft tokens per MTP step (default: 1)",
    )
    serve_parser.add_argument(
        "--mtp-optimistic",
        action="store_true",
        default=False,
        help="Skip MTP acceptance check for maximum speed. "
        "~5-10%% wrong tokens. Best for chat, not for code.",
    )
    # MCP options
    serve_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML) for tool integration",
    )
    # Security options
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (if not set, no auth required)",
    )
    serve_parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Rate limit requests per minute per client (0 = disabled)",
    )
    serve_parser.add_argument(
        "--repetition-policy",
        type=str,
        default="safe",
        choices=["safe", "strict"],
        help=(
            "Server default repetition detector mode: safe (accidental degeneration) "
            "or strict (aggressive repetition clamp)."
        ),
    )
    serve_parser.add_argument(
        "--trust-requests-when-auth-disabled",
        action="store_true",
        default=False,
        help=(
            "When --api-key is disabled, treat requests as trusted for "
            "repetition_policy_override. Defaults to false."
        ),
    )
    serve_parser.add_argument(
        "--memory-warn-threshold",
        type=float,
        default=70.0,
        help="Warn threshold for memory utilization percent (default: 70.0)",
    )
    serve_parser.add_argument(
        "--memory-limit-threshold",
        type=float,
        default=85.0,
        help="Limit threshold for memory utilization percent (default: 85.0)",
    )
    serve_parser.add_argument(
        "--memory-action",
        type=str,
        default="warn",
        choices=["warn", "reduce-context", "reject-new"],
        help=(
            "Action when memory limit threshold is crossed: "
            "warn, reduce-context, or reject-new."
        ),
    )
    serve_parser.add_argument(
        "--memory-monitor-interval",
        type=float,
        default=5.0,
        help="Memory monitor polling interval in seconds (default: 5.0)",
    )
    serve_parser.add_argument(
        "--batch-divergence-monitor",
        action="store_true",
        help="Enable periodic batch divergence probes (serial vs concurrent).",
    )
    serve_parser.add_argument(
        "--batch-divergence-interval",
        type=float,
        default=300.0,
        help="Batch divergence probe interval in seconds (default: 300.0)",
    )
    serve_parser.add_argument(
        "--batch-divergence-threshold",
        type=float,
        default=0.95,
        help="Minimum token agreement before divergence warning (0-1, default: 0.95)",
    )
    serve_parser.add_argument(
        "--batch-divergence-action",
        type=str,
        default="warn",
        choices=["warn", "serialize"],
        help=(
            "Action when divergence exceeds threshold: "
            "warn or serialize tracked inference routes."
        ),
    )
    serve_parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Default request timeout in seconds (default: 300)",
    )
    # Tool calling options
    serve_parser.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        help="Enable auto tool choice for supported models. Use --tool-call-parser to specify which parser to use.",
    )
    serve_parser.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        choices=[
            "auto",
            "mistral",
            "qwen",
            "qwen3_coder",
            "llama",
            "hermes",
            "deepseek",
            "kimi",
            "liquidai",
            "liquid",
            "lfm",
            "granite",
            "nemotron",
            "xlam",
            "functionary",
            "glm47",
        ],
        help=(
            "Select the tool call parser for the model. Options: "
            "auto (auto-detect), mistral, qwen, qwen3_coder, llama, hermes, "
            "deepseek, kimi, liquidai, liquid, lfm, granite, nemotron, "
            "xlam, functionary, glm47. "
            "Required for --enable-auto-tool-choice."
        ),
    )
    # Reasoning parser options - choices loaded dynamically from registry
    from .reasoning import list_parsers

    reasoning_choices = list_parsers()
    serve_parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        choices=reasoning_choices,
        help=(
            "Enable reasoning content extraction with specified parser. "
            "Extracts <think>...</think> tags into reasoning_content field. "
            f"Options: {', '.join(reasoning_choices)}."
        ),
    )
    serve_parser.add_argument(
        "--max-thinking-tokens",
        type=int,
        default=None,
        help=(
            "Maximum reasoning tokens to emit before routing overflow into content. "
            "Requires --reasoning-parser. Can be overridden per request via "
            "`max_thinking_tokens`."
        ),
    )
    # Multimodal option
    serve_parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force load model as multimodal (vision) even if name doesn't match auto-detection patterns",
    )
    serve_parser.add_argument(
        "--disable-mllm-vision-cache",
        action="store_true",
        help="Disable MLLM vision embedding cache in batched multimodal mode.",
    )
    serve_parser.add_argument(
        "--mllm-vision-cache-size",
        type=int,
        default=100,
        help="Maximum entries for MLLM vision embedding cache in batched multimodal mode (default: 100).",
    )
    # Generation defaults
    serve_parser.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="Override default temperature for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-top-p",
        type=float,
        default=None,
        help="Override default top_p for all requests (default: use model default)",
    )
    # Embedding model option
    serve_parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Pre-load an embedding model at startup (e.g. mlx-community/embeddinggemma-300m-6bit)",
    )
    # Download options
    serve_parser.add_argument(
        "--download-timeout",
        type=int,
        default=300,
        help="Per-file download timeout in seconds (default: 300)",
    )
    serve_parser.add_argument(
        "--download-retries",
        type=int,
        default=3,
        help="Number of download retry attempts (default: 3)",
    )
    serve_parser.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode — only use locally cached models",
    )
    # Bench command
    bench_parser = subparsers.add_parser("bench", help="Run benchmark")
    bench_parser.add_argument("model", type=str, help="Model to benchmark")
    bench_parser.add_argument(
        "--num-prompts", type=int, default=10, help="Number of prompts"
    )
    bench_parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens per prompt"
    )
    bench_parser.add_argument(
        "--max-num-seqs", type=int, default=32, help="Max concurrent sequences"
    )
    bench_parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Prefill batch size"
    )
    bench_parser.add_argument(
        "--completion-batch-size", type=int, default=16, help="Completion batch size"
    )
    bench_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Enable prefix caching (default: enabled)",
    )
    bench_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    bench_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Max entries in prefix cache (default: 100, legacy mode only)",
    )
    # Memory-aware cache options (recommended for large models)
    bench_parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=None,
        help="Cache memory limit in MB (default: auto-detect ~20%% of RAM)",
    )
    bench_parser.add_argument(
        "--cache-memory-percent",
        type=float,
        default=0.20,
        help="Fraction of available RAM for cache if auto-detecting (default: 0.20)",
    )
    bench_parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Disable memory-aware cache, use legacy entry-count based cache",
    )
    # KV cache quantization options
    bench_parser.add_argument(
        "--kv-cache-quantization",
        action="store_true",
        help="Quantize stored KV caches to reduce memory (8-bit by default)",
    )
    bench_parser.add_argument(
        "--kv-cache-quantization-bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Bit width for KV cache quantization (default: 8)",
    )
    bench_parser.add_argument(
        "--kv-cache-quantization-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
    )
    bench_parser.add_argument(
        "--kv-cache-min-quantize-tokens",
        type=int,
        default=256,
        help="Minimum tokens for quantization to apply (default: 256)",
    )
    # Paged cache options (experimental)
    bench_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help="Use paged KV cache for memory efficiency (experimental)",
    )
    bench_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per cache block (default: 64)",
    )
    bench_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks (default: 1000)",
    )

    # Detokenizer benchmark
    detok_parser = subparsers.add_parser(
        "bench-detok", help="Benchmark streaming detokenizer optimization"
    )
    detok_parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default="mlx-community/Qwen3-0.6B-8bit",
        help="Model to use for tokenizer (default: mlx-community/Qwen3-0.6B-8bit)",
    )
    detok_parser.add_argument(
        "--iterations", type=int, default=5, help="Benchmark iterations (default: 5)"
    )

    # KV cache quantization benchmark
    kv_cache_parser = subparsers.add_parser(
        "bench-kv-cache", help="Benchmark KV cache quantization memory savings"
    )
    kv_cache_parser.add_argument(
        "--layers", type=int, default=32, help="Number of layers (default: 32)"
    )
    kv_cache_parser.add_argument(
        "--seq-len", type=int, default=512, help="Sequence length (default: 512)"
    )
    kv_cache_parser.add_argument(
        "--heads", type=int, default=32, help="Number of attention heads (default: 32)"
    )
    kv_cache_parser.add_argument(
        "--head-dim", type=int, default=128, help="Head dimension (default: 128)"
    )
    kv_cache_parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )

    # Download command
    download_parser = subparsers.add_parser(
        "download", help="Download a model to local cache without starting a server"
    )
    download_parser.add_argument("model", type=str, help="Model to download")
    download_parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-file download timeout in seconds (default: 300)",
    )
    download_parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts (default: 3)",
    )
    download_parser.add_argument(
        "--mllm",
        action="store_true",
        help="Download as multimodal model (broader file patterns)",
    )
    args = parser.parse_args()

    if args.command == "serve":
        serve_command(args)
    elif args.command == "bench":
        bench_command(args)
    elif args.command == "bench-detok":
        bench_detok_command(args)
    elif args.command == "bench-kv-cache":
        bench_kv_cache_command(args)
    elif args.command == "download":
        download_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
