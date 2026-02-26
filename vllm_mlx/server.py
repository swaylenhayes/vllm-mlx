# SPDX-License-Identifier: Apache-2.0
"""
Unified OpenAI-compatible API server for vllm-mlx.

This module provides a FastAPI server that exposes an OpenAI-compatible
API for LLM and MLLM (Multimodal Language Model) inference using MLX on Apple Silicon.

Supports two modes:
- Simple mode (default): Maximum throughput for single-user scenarios
- Batched mode: Continuous batching for multiple concurrent users

Features:
- Text-only LLM inference (mlx-lm)
- Multimodal MLLM inference with images and video (mlx-vlm)
- OpenAI-compatible chat/completions API
- Streaming responses
- MCP (Model Context Protocol) tool integration
- Tool calling (Qwen/Llama formats)

Usage:
    # Simple mode (maximum throughput)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Batched mode (for multiple concurrent users)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

    # With MCP tools
    python -m vllm_mlx.server --model mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

The server provides:
    - POST /v1/completions - Text completions
    - POST /v1/chat/completions - Chat completions (with multimodal support)
    - GET /v1/models - List available models
    - GET /health - Health check
    - GET /v1/mcp/tools - List MCP tools
    - GET /v1/mcp/servers - MCP server status
    - POST /v1/mcp/execute - Execute MCP tool
"""

import argparse
import asyncio
import contextlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import secrets
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
import psutil
import yaml

# Import from new modular API
# Re-export for backwards compatibility with tests
from .api.anthropic_adapter import anthropic_to_openai, openai_to_anthropic
from .api.anthropic_models import AnthropicRequest
from .api.models import (
    AssistantMessage,  # noqa: F401
    CapabilitiesResponse,
    CapabilityAuth,
    CapabilityFeatures,
    CapabilityLimits,
    CapabilityModalities,
    CapabilityRateLimit,
    ChatCompletionChoice,  # noqa: F401
    ChatCompletionChunk,  # noqa: F401
    ChatCompletionChunkChoice,  # noqa: F401
    ChatCompletionChunkDelta,  # noqa: F401
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,  # noqa: F401
    CompletionRequest,
    CompletionResponse,
    ContentPart,  # noqa: F401
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    FunctionCall,
    ImageUrl,  # noqa: F401
    DiagnosticsHealthResponse,
    DiagnosticCheck,
    DiagnosticMemory,
    MCPExecuteRequest,
    MCPExecuteResponse,
    MCPServerInfo,  # noqa: F401
    MCPServersResponse,
    MCPToolInfo,  # noqa: F401
    MCPToolsResponse,
    Message,  # noqa: F401
    ModelInfo,  # noqa: F401
    ModelsResponse,
    ToolCall,
    Usage,  # noqa: F401
    VideoUrl,  # noqa: F401
)
from .api.tool_calling import (
    build_json_system_prompt,
    convert_tools_for_template,
    parse_json_output,
    parse_tool_calls,
)
from .api.utils import (
    SPECIAL_TOKENS_PATTERN,
    clean_output_text,
    extract_multimodal_content,
    is_mllm_model,  # noqa: F401
)
from .engine import BaseEngine, BatchedEngine, GenerationOutput, SimpleEngine
from .runtime_mode import save_observed_peak_concurrency
from .tool_parsers import ToolParserManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
_engine: BaseEngine | None = None
_model_name: str | None = None
_default_max_tokens: int = 32768
_default_timeout: float = 300.0  # Default request timeout in seconds (5 minutes)
_default_temperature: float | None = None  # Set via --default-temperature
_default_top_p: float | None = None  # Set via --default-top-p
_max_thinking_tokens: int | None = None  # Set via --max-thinking-tokens
_deterministic_mode: bool = False  # Set via --deterministic
_deterministic_serialize: bool = False  # Serialize tracked routes when deterministic

_FALLBACK_TEMPERATURE = 0.7
_FALLBACK_TOP_P = 0.9


class ConcurrencyTracker:
    """Track active and peak concurrent inference requests."""

    def __init__(self) -> None:
        self._active = 0
        self._peak = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self._active += 1
            if self._active > self._peak:
                self._peak = self._active

    def exit(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self._active, self._peak

    @property
    def peak(self) -> int:
        with self._lock:
            return self._peak


class MemoryPressureState:
    """Thread-safe memory pressure state used by diagnostics and request policies."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_bytes: int | None = None
        self._peak_bytes: int | None = None
        self._system_bytes: int | None = None
        self._utilization_pct: float | None = None
        self._trend: str = "unknown"
        self._pressure: str = "unknown"
        self._max_tokens_factor: float = 1.0
        self._reject_new: bool = False
        self._last_updated_epoch: float | None = None
        self._prev_active_bytes: int | None = None

    def update(
        self,
        *,
        active_bytes: int | None,
        peak_bytes: int | None,
        system_bytes: int | None,
        warn_threshold_pct: float,
        limit_threshold_pct: float,
        action: str,
    ) -> None:
        with self._lock:
            self._active_bytes = active_bytes
            self._peak_bytes = peak_bytes
            self._system_bytes = system_bytes
            self._last_updated_epoch = time.time()

            utilization_pct = None
            if (
                active_bytes is not None
                and system_bytes is not None
                and system_bytes > 0
            ):
                utilization_pct = (active_bytes / system_bytes) * 100.0
            self._utilization_pct = utilization_pct

            if active_bytes is not None and self._prev_active_bytes is not None:
                delta = active_bytes - self._prev_active_bytes
                drift_floor = 256 * 1024 * 1024  # 256MB
                if abs(delta) <= drift_floor:
                    trend = "stable"
                else:
                    trend = "growing" if delta > 0 else "declining"
            else:
                trend = "unknown"
            self._trend = trend
            self._prev_active_bytes = active_bytes

            if utilization_pct is None:
                pressure = "unknown"
            elif utilization_pct >= limit_threshold_pct:
                pressure = "critical"
            elif utilization_pct >= warn_threshold_pct:
                pressure = "elevated"
            else:
                pressure = "normal"
            self._pressure = pressure

            # Enforce configured action policy for new requests.
            self._max_tokens_factor = (
                0.5 if action == "reduce-context" and pressure == "critical" else 1.0
            )
            self._reject_new = action == "reject-new" and pressure == "critical"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_bytes": self._active_bytes,
                "peak_bytes": self._peak_bytes,
                "system_bytes": self._system_bytes,
                "utilization_pct": self._utilization_pct,
                "trend": self._trend,
                "pressure": self._pressure,
                "max_tokens_factor": self._max_tokens_factor,
                "reject_new": self._reject_new,
                "last_updated_epoch": self._last_updated_epoch,
            }

    def max_tokens_factor(self) -> float:
        with self._lock:
            return self._max_tokens_factor

    def should_reject_new(self) -> bool:
        with self._lock:
            return self._reject_new


class BatchDivergenceState:
    """Thread-safe state for batch divergence probes and mitigation policy."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled: bool = False
        self._threshold: float = 0.95
        self._action: str = "warn"  # warn | serialize
        self._model_name: str | None = None
        self._engine_type: str = "unknown"
        self._sample_count: int = 0
        self._last_checked_epoch: float | None = None
        self._last_token_agreement: float | None = None
        self._last_exact_match: bool | None = None
        self._last_serial_latency_s: float | None = None
        self._last_concurrent_latency_s: float | None = None
        self._status: str = "unknown"  # pass | warning | unknown
        self._detail: str = "No probe samples collected yet."
        self._error: str | None = None

    def configure(self, *, enabled: bool, threshold: float, action: str) -> None:
        with self._lock:
            self._enabled = enabled
            self._threshold = threshold
            self._action = action
            if not enabled:
                self._status = "pass"
                self._detail = "Batch divergence monitor disabled."
                self._error = None

    def reset(self, *, model_name: str | None, engine_type: str) -> None:
        with self._lock:
            self._model_name = model_name
            self._engine_type = engine_type
            self._sample_count = 0
            self._last_checked_epoch = None
            self._last_token_agreement = None
            self._last_exact_match = None
            self._last_serial_latency_s = None
            self._last_concurrent_latency_s = None
            self._status = "unknown"
            self._detail = "No probe samples collected yet."
            self._error = None

    def update_probe(
        self,
        *,
        token_agreement: float,
        exact_match: bool,
        serial_latency_s: float,
        concurrent_latency_s: float,
    ) -> None:
        with self._lock:
            self._sample_count += 1
            self._last_checked_epoch = time.time()
            self._last_token_agreement = token_agreement
            self._last_exact_match = exact_match
            self._last_serial_latency_s = serial_latency_s
            self._last_concurrent_latency_s = concurrent_latency_s
            self._error = None

            if token_agreement < self._threshold:
                self._status = "warning"
                self._detail = (
                    f"Token agreement {token_agreement * 100:.2f}% is below "
                    f"threshold {self._threshold * 100:.2f}%."
                )
            else:
                self._status = "pass"
                self._detail = (
                    f"Token agreement {token_agreement * 100:.2f}% meets "
                    f"threshold {self._threshold * 100:.2f}%."
                )

    def update_info(self, *, status: str, detail: str, error: str | None = None) -> None:
        with self._lock:
            self._last_checked_epoch = time.time()
            self._status = status
            self._detail = detail
            self._error = error

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            serialize_active = (
                self._enabled
                and self._action == "serialize"
                and self._status == "warning"
                and self._engine_type == "batched"
            )
            return {
                "enabled": self._enabled,
                "threshold": self._threshold,
                "action": self._action,
                "model_name": self._model_name,
                "engine_type": self._engine_type,
                "sample_count": self._sample_count,
                "last_checked_epoch": self._last_checked_epoch,
                "last_token_agreement": self._last_token_agreement,
                "last_exact_match": self._last_exact_match,
                "last_serial_latency_s": self._last_serial_latency_s,
                "last_concurrent_latency_s": self._last_concurrent_latency_s,
                "status": self._status,
                "detail": self._detail,
                "error": self._error,
                "serialize_active": serialize_active,
            }

    def should_serialize(self) -> bool:
        with self._lock:
            return (
                self._enabled
                and self._action == "serialize"
                and self._status == "warning"
                and self._engine_type == "batched"
            )


_KNOWN_BUGS_PATH = Path(__file__).resolve().parent / "data" / "known_bugs.yaml"
_memory_state = MemoryPressureState()
_memory_monitor_task: asyncio.Task | None = None
_memory_warn_threshold_pct: float = 70.0
_memory_limit_threshold_pct: float = 85.0
_memory_action: str = "warn"  # warn | reduce-context | reject-new
_memory_monitor_interval_seconds: float = 5.0
_batch_divergence_state = BatchDivergenceState()
_batch_divergence_monitor_task: asyncio.Task | None = None
_batch_divergence_monitor_enabled: bool = False
_batch_divergence_interval_seconds: float = 300.0
_batch_divergence_threshold: float = 0.95
_batch_divergence_action: str = "warn"  # warn | serialize
_batch_divergence_probe_primary_prompt: str = "Return exactly: READY"
_batch_divergence_probe_distractor_prompt: str = (
    "Give a one-line definition of deterministic decoding."
)
_batch_serialize_lock: asyncio.Lock | None = None


def _resolve_temperature(request_value: float | None) -> float:
    """Resolve temperature: request > CLI default > fallback."""
    if _deterministic_mode:
        return 0.0
    if request_value is not None:
        return request_value
    if _default_temperature is not None:
        return _default_temperature
    return _FALLBACK_TEMPERATURE


def _resolve_top_p(request_value: float | None) -> float:
    """Resolve top_p: request > CLI default > fallback."""
    if _deterministic_mode:
        return 1.0
    if request_value is not None:
        return request_value
    if _default_top_p is not None:
        return _default_top_p
    return _FALLBACK_TOP_P


def _resolve_repetition_penalty(
    repetition_penalty: float | None,
    frequency_penalty: float | None,
) -> float | None:
    """
    Resolve repetition penalty from explicit or OpenAI-style frequency penalty.

    Priority:
    1) explicit repetition_penalty (passthrough)
    2) mapped from frequency_penalty (1.0 + frequency_penalty)
    3) None (backend default)
    """
    if repetition_penalty is not None:
        return repetition_penalty
    if frequency_penalty is None:
        return None

    mapped = 1.0 + frequency_penalty
    # Decoding backends require strictly positive repetition penalty.
    return max(0.01, mapped)


def _resolve_max_thinking_tokens(request_value: int | None) -> int | None:
    """Resolve max thinking budget: request override > server default."""
    if request_value is not None:
        return request_value
    return _max_thinking_tokens


def _get_reasoning_boundary_tokens() -> tuple[str, str] | None:
    """
    Return reasoning boundary tokens when parser exposes think-style tags.

    Non-think parsers (e.g., harmony/gpt-oss) do not expose start/end tags and
    cannot support forced think-exit steering.
    """
    if _reasoning_parser is None:
        return None

    start_token = getattr(_reasoning_parser, "start_token", None)
    end_token = getattr(_reasoning_parser, "end_token", None)
    if not isinstance(start_token, str) or not start_token:
        return None
    if not isinstance(end_token, str) or not end_token:
        return None
    return start_token, end_token


def _build_engine_thinking_kwargs(request_value: int | None) -> dict[str, Any]:
    """
    Build optional engine-level thinking controls for decode steering.

    This is currently effective on SimpleEngine LLM decode paths. Other engines
    safely ignore unknown kwargs and continue using API-layer budget fallback.
    """
    max_thinking_tokens = _resolve_max_thinking_tokens(request_value)
    boundaries = _get_reasoning_boundary_tokens()
    if max_thinking_tokens is None or boundaries is None:
        return {}

    start_token, end_token = boundaries
    return {
        "thinking_budget_tokens": max_thinking_tokens,
        "thinking_start_token": start_token,
        "thinking_end_token": end_token,
    }


def _get_text_tokenizer() -> Any | None:
    """Return a tokenizer with encode/decode if available."""
    if _engine is None:
        return None
    tokenizer = getattr(_engine, "tokenizer", None)
    if tokenizer is None:
        return None
    if hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
        return tokenizer
    nested = getattr(tokenizer, "tokenizer", None)
    if nested is not None and hasattr(nested, "encode") and hasattr(nested, "decode"):
        return nested
    return None


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    """Best-effort tokenizer encoding wrapper."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    """Best-effort tokenizer decode wrapper."""
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(token_ids)


def _split_text_by_token_budget(
    text: str,
    token_budget: int,
    tokenizer: Any | None,
) -> tuple[str, str]:
    """
    Split text into (within_budget, overflow) by token count.

    Falls back to a whitespace split if tokenizer is unavailable.
    """
    if not text or token_budget <= 0:
        return "", text

    if tokenizer is not None:
        try:
            token_ids = _encode_text(tokenizer, text)
            if len(token_ids) <= token_budget:
                return text, ""
            in_budget = _decode_tokens(tokenizer, token_ids[:token_budget])
            overflow = _decode_tokens(tokenizer, token_ids[token_budget:])
            return in_budget, overflow
        except Exception:
            pass

    words = text.split()
    if len(words) <= token_budget:
        return text, ""
    in_budget = " ".join(words[:token_budget])
    overflow = " ".join(words[token_budget:])
    return in_budget, overflow


def _count_text_tokens(text: str, tokenizer: Any | None) -> int:
    """Best-effort token count for text deltas."""
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(_encode_text(tokenizer, text))
        except Exception:
            pass
    return len(text.split())


# Global MCP manager
_mcp_manager = None
_mcp_executor = None

# Global embedding engine (lazy loaded)
_embedding_engine = None
_embedding_model_locked: str | None = None  # Set when --embedding-model is used

# API key authentication
_api_key: str | None = None
_auth_warning_logged: bool = False

# Reasoning parser (for models like Qwen3, DeepSeek-R1)
_reasoning_parser = None  # ReasoningParser instance when enabled

# Tool calling configuration
_enable_auto_tool_choice: bool = False
_tool_call_parser: str | None = None  # Parser name: auto, mistral, qwen, llama, hermes
_tool_parser_instance = None  # Instantiated parser
# Anti-spray guard for pathological "many calls to same function" outputs.
# Threshold is intentionally high to avoid impacting legitimate multi-tool flows.
_tool_call_spray_threshold: int = 8
_concurrency_tracker = ConcurrencyTracker()

_CONCURRENCY_TRACKED_PATHS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/embeddings",
)


def _should_track_concurrency(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in _CONCURRENCY_TRACKED_PATHS)


def _load_prefix_cache_from_disk() -> None:
    """Load prefix cache from disk during startup."""
    try:
        d = _get_cache_dir()
        logger.info(f"[lifespan] Loading prefix cache from {d}")
        loaded = _engine.load_cache_from_disk(d)
        if loaded > 0:
            logger.info(f"[lifespan] Loaded {loaded} prefix cache entries")
        else:
            logger.info("[lifespan] No prefix cache entries found on disk")
    except Exception as e:
        logger.warning(f"[lifespan] Failed to load cache from disk: {e}", exc_info=True)


def _save_prefix_cache_to_disk() -> None:
    """Save prefix cache to disk during shutdown."""
    try:
        d = _get_cache_dir()
        logger.info(f"[lifespan] Saving prefix cache to {d}")
        saved = _engine.save_cache_to_disk(d)
        if saved:
            logger.info(f"[lifespan] Saved prefix cache to {d}")
        else:
            logger.info("[lifespan] No cache to save")
    except Exception as e:
        logger.warning(f"[lifespan] Failed to save cache to disk: {e}", exc_info=True)


def _get_cache_dir() -> str:
    """Get cache persistence directory based on model name."""
    # Use global _model_name which is always a string, set during load_model()
    model_name = _model_name if _model_name else "default"
    logger.info(
        f"[_get_cache_dir] _model_name={_model_name!r} type={type(_model_name)}"
    )
    # Sanitize model name for filesystem
    safe_name = str(model_name).replace("/", "--").replace("\\", "--")
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "vllm-mlx", "prefix_cache", safe_name
    )
    logger.info(f"[_get_cache_dir] cache_dir={cache_dir!r}")
    return cache_dir


async def lifespan(app: FastAPI):
    """FastAPI lifespan for startup/shutdown events."""
    global _engine, _mcp_manager, _memory_monitor_task
    global _batch_divergence_monitor_task, _batch_serialize_lock

    # Startup: Start engine if loaded (needed for BatchedEngine in uvicorn's event loop)
    if _engine is not None and hasattr(_engine, "_loaded") and not _engine._loaded:
        await _engine.start()

    # Load persisted cache from disk (AFTER engine start — AsyncEngineCore must exist)
    if _engine is not None and hasattr(_engine, "load_cache_from_disk"):
        _load_prefix_cache_from_disk()

    # Initialize MCP if config provided
    mcp_config = os.environ.get("VLLM_MLX_MCP_CONFIG")
    if mcp_config:
        await init_mcp(mcp_config)

    # Start background memory monitor for pressure guardrails.
    if _memory_monitor_task is None or _memory_monitor_task.done():
        _memory_monitor_task = asyncio.create_task(_memory_monitor_loop())
    if _batch_serialize_lock is None:
        _batch_serialize_lock = asyncio.Lock()
    if _batch_divergence_monitor_task is None or _batch_divergence_monitor_task.done():
        _batch_divergence_monitor_task = asyncio.create_task(
            _batch_divergence_monitor_loop()
        )

    yield

    # Shutdown: Save cache to disk BEFORE stopping engine
    if _engine is not None and hasattr(_engine, "save_cache_to_disk"):
        _save_prefix_cache_to_disk()

    # Persist observed concurrency profile for next startup mode selection.
    peak_concurrency = _concurrency_tracker.peak
    if peak_concurrency > 0:
        try:
            profile = save_observed_peak_concurrency(peak_concurrency)
            logger.info(
                "[lifespan] Persisted concurrency profile: last_run_peak=%s observed_peak=%s",
                profile.get("last_run_peak_concurrency"),
                profile.get("observed_peak_concurrency"),
            )
        except Exception as e:
            logger.warning(
                "[lifespan] Failed to persist concurrency profile: %s",
                e,
                exc_info=True,
            )

    # Shutdown: Close MCP connections and stop engine
    if _mcp_manager is not None:
        await _mcp_manager.stop()
        logger.info("MCP manager stopped")
    if _engine is not None:
        await _engine.stop()
        logger.info("Engine stopped")
    if _memory_monitor_task is not None:
        _memory_monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _memory_monitor_task
        _memory_monitor_task = None
    if _batch_divergence_monitor_task is not None:
        _batch_divergence_monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _batch_divergence_monitor_task
        _batch_divergence_monitor_task = None
    _batch_serialize_lock = None


app = FastAPI(
    title="vllm-mlx API",
    description="OpenAI-compatible API for MLX LLM/MLLM inference on Apple Silicon",
    version="0.2.1",
    lifespan=lifespan,
)


@app.middleware("http")
async def track_runtime_concurrency(request: Request, call_next):
    """Track concurrency for high-impact inference endpoints."""
    should_track = _should_track_concurrency(request.url.path)
    if should_track and _memory_state.should_reject_new():
        snapshot = _memory_state.snapshot()
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Memory pressure is critical; rejecting new requests.",
                "memory_pressure": snapshot.get("pressure"),
            },
        )

    async def _invoke() -> Response:
        if should_track:
            _concurrency_tracker.enter()
        try:
            return await call_next(request)
        finally:
            if should_track:
                _concurrency_tracker.exit()

    if should_track and _batch_serialize_lock is not None and (
        _deterministic_serialize or _batch_divergence_state.should_serialize()
    ):
        async with _batch_serialize_lock:
            return await _invoke()

    return await _invoke()


security = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def _normalize_api_key(value: str | None) -> str | None:
    """Normalize API key value from headers/credentials."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _extract_bearer_token(authorization_header: str | None) -> str | None:
    """Extract bearer token from Authorization header."""
    if not isinstance(authorization_header, str):
        return None

    parts = authorization_header.strip().split(maxsplit=1)
    if len(parts) != 2:
        return None

    scheme, token = parts
    if scheme.lower() != "bearer":
        return None

    return _normalize_api_key(token)


def _extract_api_key_from_headers(
    authorization_header: str | None,
    x_api_key: str | None,
) -> str | None:
    """Extract API key from supported auth headers."""
    bearer_token = _extract_bearer_token(authorization_header)
    if bearer_token:
        return bearer_token

    return _normalize_api_key(x_api_key)


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, requests_per_minute: int = 60, enabled: bool = False):
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled
        self.window_size = 60.0  # 1 minute window
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Check if request is allowed for client.

        Returns:
            (is_allowed, retry_after_seconds)
        """
        if not self.enabled:
            return True, 0

        current_time = time.time()
        window_start = current_time - self.window_size

        with self._lock:
            # Clean old requests outside window
            self._requests[client_id] = [
                t for t in self._requests[client_id] if t > window_start
            ]

            # Check rate limit
            if len(self._requests[client_id]) >= self.requests_per_minute:
                # Calculate retry-after
                oldest = min(self._requests[client_id])
                retry_after = int(oldest + self.window_size - current_time) + 1
                return False, max(1, retry_after)

            # Record this request
            self._requests[client_id].append(current_time)
            return True, 0


# Global rate limiter (disabled by default)
_rate_limiter = RateLimiter(requests_per_minute=60, enabled=False)


async def check_rate_limit(request: Request):
    """Rate limiting dependency."""
    # Use normalized API key as client ID when available, otherwise use IP.
    request_api_key = _extract_api_key_from_headers(
        request.headers.get("Authorization"),
        request.headers.get("x-api-key"),
    )
    client_id = (
        f"api-key:{request_api_key}"
        if request_api_key
        else (request.client.host if request.client else "unknown")
    )

    allowed, retry_after = _rate_limiter.is_allowed(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


async def verify_api_key(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(security)
    ] = None,
    x_api_key: Annotated[str | None, Depends(api_key_header)] = None,
):
    """Verify API key if authentication is enabled."""
    global _auth_warning_logged

    if _api_key is None:
        # Log warning once about running without authentication
        if not _auth_warning_logged:
            logger.warning(
                "SECURITY WARNING: Server running without API key authentication. "
                "Anyone can access the API. Use --api-key to enable authentication."
            )
            _auth_warning_logged = True
        return True  # No auth required

    provided_api_key = _normalize_api_key(
        credentials.credentials
        if isinstance(credentials, HTTPAuthorizationCredentials)
        else None
    )
    if provided_api_key is None:
        provided_api_key = _normalize_api_key(x_api_key)

    if provided_api_key is None:
        raise HTTPException(status_code=401, detail="API key required")
    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(provided_api_key, _api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def get_engine() -> BaseEngine:
    """Get the loaded engine, raising error if not loaded."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _engine


def _canonicalize_tool_arguments(arguments: Any) -> str:
    """Canonical string form for tool call argument comparison."""
    if arguments is None:
        return ""

    if isinstance(arguments, (dict, list)):
        try:
            return json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        except Exception:
            return str(arguments).strip()

    if isinstance(arguments, str):
        value = arguments.strip()
        if not value:
            return ""
        try:
            parsed = json.loads(value)
            return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        except Exception:
            return value

    return str(arguments).strip()


def _tool_call_name(call: ToolCall | dict[str, Any]) -> str:
    """Extract tool function name from ToolCall object or stream/tool dict."""
    if isinstance(call, ToolCall):
        return call.function.name
    if isinstance(call, dict):
        fn = call.get("function")
        if isinstance(fn, dict):
            return str(fn.get("name", ""))
        return str(call.get("name", ""))
    return ""


def _tool_call_arguments(call: ToolCall | dict[str, Any]) -> Any:
    """Extract raw tool arguments from ToolCall object or stream/tool dict."""
    if isinstance(call, ToolCall):
        return call.function.arguments
    if isinstance(call, dict):
        fn = call.get("function")
        if isinstance(fn, dict):
            return fn.get("arguments")
        return call.get("arguments")
    return None


def _dedupe_tool_calls(
    tool_calls: list[ToolCall] | list[dict[str, Any]],
) -> list[ToolCall] | list[dict[str, Any]]:
    """Remove exact duplicate calls (same function + same canonical arguments)."""
    seen: set[tuple[str, str]] = set()
    deduped: list[ToolCall] | list[dict[str, Any]] = []

    for call in tool_calls:
        name = _tool_call_name(call)
        args_key = _canonicalize_tool_arguments(_tool_call_arguments(call))
        key = (name, args_key)

        if name and key in seen:
            continue
        if name:
            seen.add(key)
        deduped.append(call)

    return deduped


def _apply_tool_call_spray_policy(
    tool_calls: list[ToolCall],
    *,
    source: str,
) -> list[ToolCall]:
    """
    Apply anti-spray policy for tool calls.

    Policy:
    1. Remove exact duplicates by (function, canonical arguments).
    2. If a large burst remains and every call targets the same function,
       collapse to the first call.
    """
    if not tool_calls:
        return tool_calls

    original_count = len(tool_calls)
    deduped = _dedupe_tool_calls(tool_calls)
    deduped_count = len(deduped)
    if deduped_count < original_count:
        logger.info(
            "Tool-call dedupe (%s): %s -> %s",
            source,
            original_count,
            deduped_count,
        )

    if deduped_count >= _tool_call_spray_threshold:
        names = {_tool_call_name(call) for call in deduped if _tool_call_name(call)}
        if len(names) == 1:
            fn_name = next(iter(names))
            logger.warning(
                "Tool-call spray mitigation (%s): collapsing %s '%s' calls to first call",
                source,
                deduped_count,
                fn_name,
            )
            return [deduped[0]]

    return deduped


def _normalize_tool_call_delta(
    call: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    """Normalize stream delta tool-call shape to OpenAI-compatible dict."""
    fn = call.get("function")
    if isinstance(fn, dict):
        name = str(fn.get("name", ""))
        arguments = fn.get("arguments", "")
    else:
        name = str(call.get("name", ""))
        arguments = call.get("arguments", "")

    return {
        "index": index,
        "id": call.get("id", f"call_{uuid.uuid4().hex[:8]}"),
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments if isinstance(arguments, str) else str(arguments),
        },
    }


def _apply_tool_call_spray_policy_to_deltas(
    tool_calls: list[dict[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    """Apply anti-spray policy to streaming tool-call deltas."""
    if not tool_calls:
        return tool_calls

    normalized = [_normalize_tool_call_delta(tc, index=i) for i, tc in enumerate(tool_calls)]
    deduped = _dedupe_tool_calls(normalized)
    deduped_count = len(deduped)

    if len(normalized) > deduped_count:
        logger.info(
            "Tool-call delta dedupe (%s): %s -> %s",
            source,
            len(normalized),
            deduped_count,
        )

    if deduped_count >= _tool_call_spray_threshold:
        names = {_tool_call_name(call) for call in deduped if _tool_call_name(call)}
        if len(names) == 1:
            fn_name = next(iter(names))
            logger.warning(
                "Tool-call spray mitigation (%s): collapsing %s '%s' calls to first call",
                source,
                deduped_count,
                fn_name,
            )
            deduped = [deduped[0]]

    return [_normalize_tool_call_delta(tc, index=i) for i, tc in enumerate(deduped)]


def _parse_tool_calls_with_mitigation(
    output_text: str,
    request_dict: dict[str, Any] | None,
    *,
    source: str,
) -> tuple[str, list[ToolCall] | None]:
    """Parse tool calls with anti-spray post-processing."""
    cleaned_text, tool_calls = parse_tool_calls(output_text, request_dict)
    if not tool_calls:
        return cleaned_text, None
    tools = request_dict.get("tools") if request_dict else None
    if tools:
        tool_calls = [
            ToolCall(
                id=tc.id,
                type=tc.type,
                function=FunctionCall(
                    name=tc.function.name,
                    arguments=_coerce_tool_arguments(
                        tc.function.arguments, tc.function.name, tools
                    ),
                ),
            )
            for tc in tool_calls
        ]
    return cleaned_text, _apply_tool_call_spray_policy(tool_calls, source=source)


def _coerce_tool_arguments(
    arguments_json: str, tool_name: str, tools: list[dict] | None
) -> str:
    """
    Coerce tool call arguments to match the tool schema.

    If a schema field expects "string" but the model produced an object/array,
    JSON-stringify the value. This fixes a common LLM failure mode where models
    output raw JSON objects instead of JSON strings for file content, etc.
    """
    if not tools:
        return arguments_json

    # Find the schema for this tool
    schema = None
    for tool in tools:
        if isinstance(tool, dict) and tool.get("function", {}).get("name") == tool_name:
            schema = tool["function"].get("parameters", {})
            break

    if not schema or "properties" not in schema:
        return arguments_json

    try:
        arguments = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError):
        return arguments_json

    if not isinstance(arguments, dict):
        return arguments_json

    properties = schema.get("properties", {})
    changed = False

    for key, value in arguments.items():
        if key in properties:
            expected_type = properties[key].get("type")
            if expected_type == "string" and isinstance(value, (dict, list)):
                arguments[key] = json.dumps(value, ensure_ascii=False, indent=2)
                changed = True

    if changed:
        return json.dumps(arguments, ensure_ascii=False)

    return arguments_json


def _coerce_tool_call_delta_arguments(
    tool_call: dict[str, Any], tools: list[dict] | None
) -> dict[str, Any]:
    """Coerce streaming tool-call arguments to match schema types when possible."""
    if not tools:
        return tool_call

    call = dict(tool_call)
    fn = call.get("function")
    if isinstance(fn, dict):
        fn_copy = dict(fn)
        name = fn_copy.get("name")
        arguments = fn_copy.get("arguments")
        if isinstance(name, str) and isinstance(arguments, str):
            fn_copy["arguments"] = _coerce_tool_arguments(arguments, name, tools)
            call["function"] = fn_copy
        return call

    name = call.get("name")
    arguments = call.get("arguments")
    if isinstance(name, str) and isinstance(arguments, str):
        call["arguments"] = _coerce_tool_arguments(arguments, name, tools)
    return call


def _parse_tool_calls_with_parser(
    output_text: str, request: ChatCompletionRequest | None = None
) -> tuple[str, list | None]:
    """
    Parse tool calls from model output using the configured parser.

    If --enable-auto-tool-choice is set with --tool-call-parser, uses the
    selected parser. Otherwise falls back to the generic parse_tool_calls.

    Args:
        output_text: The model output text
        request: The original request (for context)

    Returns:
        Tuple of (cleaned_text, tool_calls)
    """
    global _tool_parser_instance

    request_dict = request.model_dump() if request else None

    # If auto tool choice is not enabled, use the generic parser
    if not _enable_auto_tool_choice or not _tool_call_parser:
        return _parse_tool_calls_with_mitigation(
            output_text,
            request_dict,
            source="generic",
        )

    # Initialize parser if needed
    if _tool_parser_instance is None:
        try:
            parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
            # Get tokenizer from engine if available
            tokenizer = None
            if _engine is not None and hasattr(_engine, "_tokenizer"):
                tokenizer = _engine._tokenizer
            _tool_parser_instance = parser_cls(tokenizer)
            logger.info(f"Initialized tool call parser: {_tool_call_parser}")
        except Exception as e:
            logger.warning(
                f"Failed to initialize tool parser '{_tool_call_parser}': {e}"
            )
            logger.warning("Falling back to generic parser")
            return _parse_tool_calls_with_mitigation(
                output_text,
                request_dict,
                source="fallback-init",
            )

    # Use the configured parser
    try:
        # Reset parser state between requests
        _tool_parser_instance.reset()
        result = _tool_parser_instance.extract_tool_calls(output_text, request_dict)
        if result.tools_called:
            tools = request_dict.get("tools") if request_dict else None
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    type="function",
                    function=FunctionCall(
                        name=tc["name"],
                        arguments=_coerce_tool_arguments(
                            tc["arguments"], tc["name"], tools
                        ),
                    ),
                )
                for tc in result.tool_calls
            ]
            return (
                result.content or "",
                _apply_tool_call_spray_policy(tool_calls, source="parser"),
            )
        else:
            # Fallback: specific parser didn't find tool calls,
            # try generic parser which handles more formats (e.g. Nemotron XML)
            return _parse_tool_calls_with_mitigation(
                output_text,
                request_dict,
                source="fallback-empty",
            )
    except Exception as e:
        logger.warning(f"Tool parser error: {e}")
        return _parse_tool_calls_with_mitigation(
            output_text,
            request_dict,
            source="fallback-error",
        )


def _detect_native_tool_support() -> bool:
    """
    Detect if the active tool parser supports native tool format.

    Native format means role="tool" messages and tool_calls fields
    are preserved instead of being converted to text.

    Returns:
        True if native format should be preserved
    """
    if not _enable_auto_tool_choice or not _tool_call_parser:
        return False

    try:
        parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
        return parser_cls.supports_native_format()
    except KeyError:
        # Parser not found - this is a configuration error, log as error
        logger.error(
            f"Tool parser '{_tool_call_parser}' not found. "
            f"Available parsers: {ToolParserManager.list_registered()}"
        )
        return False
    except Exception as e:
        # Unexpected error during detection
        logger.warning(f"Failed to detect native tool support: {e}")
        return False


def load_embedding_model(
    model_name: str | None,
    *,
    lock: bool = False,
    reuse_existing: bool = True,
) -> None:
    """Load or reuse the embedding model engine when configured."""
    global _embedding_engine, _embedding_model_locked

    if not model_name:
        return

    if lock:
        _embedding_model_locked = model_name

    if (
        reuse_existing
        and _embedding_engine is not None
        and _embedding_engine.model_name == model_name
    ):
        return

    from .embedding import EmbeddingEngine

    _embedding_engine = EmbeddingEngine(model_name)
    _embedding_engine.load()


def load_model(
    model_name: str,
    use_batching: bool = False,
    scheduler_config=None,
    stream_interval: int = 1,
    max_tokens: int = 32768,
    force_mllm: bool = False,
):
    """
    Load a model (auto-detects MLLM vs LLM).

    Args:
        model_name: HuggingFace model name or local path
        use_batching: Use continuous batching (BatchedEngine) vs simple mode (SimpleEngine)
        scheduler_config: Scheduler config for batched mode
        stream_interval: Tokens to batch before streaming (batched mode only)
        max_tokens: Default max tokens for generation
        force_mllm: Force loading as MLLM even if not auto-detected
    """
    global _engine, _model_name, _default_max_tokens, _tool_parser_instance
    global _batch_divergence_state

    _default_max_tokens = max_tokens
    _model_name = model_name
    # Reset tool parser instance when model is reloaded (tokenizer may change)
    _tool_parser_instance = None

    if force_mllm:
        logger.info("Force MLLM mode enabled via --mllm flag")

    if use_batching:
        logger.info(f"Loading model with BatchedEngine: {model_name}")
        _engine = BatchedEngine(
            model_name=model_name,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval,
            force_mllm=force_mllm,
        )
        # BatchedEngine will be started in lifespan (uvicorn's event loop)
        # Just log for now
        logger.info(f"Model loaded (batched mode): {model_name}")
        _batch_divergence_state.reset(model_name=model_name, engine_type="batched")
    else:
        logger.info(f"Loading model with SimpleEngine: {model_name}")
        _engine = SimpleEngine(model_name=model_name, force_mllm=force_mllm)
        # Start SimpleEngine synchronously (no background loop)
        # Use new_event_loop() for Python 3.10+ compatibility (get_event_loop() is deprecated)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_engine.start())
        model_type = "MLLM" if _engine.is_mllm else "LLM"
        logger.info(f"{model_type} model loaded (simple mode): {model_name}")
        _batch_divergence_state.reset(model_name=model_name, engine_type="simple")

    # Set native tool format support on the engine (thread-safe via instance property)
    _engine.preserve_native_tool_format = _detect_native_tool_support()
    if _engine.preserve_native_tool_format:
        logger.info(f"Native tool format enabled for parser: {_tool_call_parser}")

    logger.info(f"Default max tokens: {_default_max_tokens}")


def get_usage(output: GenerationOutput) -> Usage:
    """Extract usage metrics from GenerationOutput."""
    total_prompt_tokens = (
        output.prompt_tokens if hasattr(output, "prompt_tokens") else 0
    )
    total_completion_tokens = (
        output.completion_tokens if hasattr(output, "completion_tokens") else 0
    )
    return Usage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
    )


def _module_available(module_name: str) -> bool:
    """Check whether a Python module can be imported in this runtime."""
    return importlib.util.find_spec(module_name) is not None


def _parse_version_tuple(version_str: str | None) -> tuple[int, ...] | None:
    """Parse a dotted version string into a comparable integer tuple."""
    if not isinstance(version_str, str) or not version_str.strip():
        return None
    parts: list[int] = []
    for raw in version_str.strip().split("."):
        digits = "".join(ch for ch in raw if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else None


def _version_satisfies_constraint(version_str: str | None, constraint: str) -> bool:
    """
    Evaluate simple semantic-version constraints.

    Supports comma-separated constraints using operators:
    `<`, `<=`, `>`, `>=`, `==`, `!=`.
    """
    version_tuple = _parse_version_tuple(version_str)
    if version_tuple is None:
        return False

    for raw_part in constraint.split(","):
        part = raw_part.strip()
        if not part:
            continue
        operator = None
        rhs = None
        for candidate in ("<=", ">=", "==", "!=", "<", ">"):
            if part.startswith(candidate):
                operator = candidate
                rhs = part[len(candidate) :].strip()
                break
        if operator is None or not rhs:
            return False

        rhs_tuple = _parse_version_tuple(rhs)
        if rhs_tuple is None:
            return False

        if operator == "<" and not (version_tuple < rhs_tuple):
            return False
        if operator == "<=" and not (version_tuple <= rhs_tuple):
            return False
        if operator == ">" and not (version_tuple > rhs_tuple):
            return False
        if operator == ">=" and not (version_tuple >= rhs_tuple):
            return False
        if operator == "==" and not (version_tuple == rhs_tuple):
            return False
        if operator == "!=" and not (version_tuple != rhs_tuple):
            return False

    return True


def _collect_runtime_versions() -> dict[str, str]:
    """Collect runtime package versions for diagnostics."""
    versions: dict[str, str] = {}
    for package_name in ("mlx", "mlx-lm", "vllm-mlx"):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "unknown"
    return versions


def _load_known_bug_entries() -> list[dict[str, Any]]:
    """Load known MLX reliability bugs from YAML database."""
    try:
        if not _KNOWN_BUGS_PATH.exists():
            return []
        with _KNOWN_BUGS_PATH.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        bugs = payload.get("bugs", [])
        return bugs if isinstance(bugs, list) else []
    except Exception as e:
        logger.warning("Failed to load known bugs DB: %s", e, exc_info=True)
        return []


def _infer_model_architecture() -> str:
    """Best-effort architecture key used by known-bug matching."""
    model_name = (_model_name or "").lower()
    if "gemma-3" in model_name or "gemma3" in model_name:
        return "gemma3"
    if "qwen3-vl-30b-a3b" in model_name or "a3b" in model_name or "moe" in model_name:
        return "qwen3_vl_moe"
    if "qwen3-vl" in model_name:
        return "qwen3_vl"
    if "qwen3" in model_name:
        return "qwen3"
    if "zwz" in model_name:
        return "zwz_vl"
    if "deepseek" in model_name:
        return "deepseek"
    return "unknown"


def _read_metal_memory_bytes() -> tuple[int | None, int | None]:
    """Read active/peak Metal memory from MLX runtime if available."""
    try:
        import mlx.core as mx
    except Exception:
        return None, None

    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "is_available") and not metal.is_available():
        return None, None

    active_getter = None
    peak_getter = None
    if metal is not None:
        active_getter = getattr(metal, "get_active_memory", None)
        peak_getter = getattr(metal, "get_peak_memory", None)
    if active_getter is None:
        active_getter = getattr(mx, "get_active_memory", None)
    if peak_getter is None:
        peak_getter = getattr(mx, "get_peak_memory", None)

    if not callable(active_getter) or not callable(peak_getter):
        return None, None
    return int(active_getter()), int(peak_getter())


def _system_memory_bytes() -> int | None:
    """Return total system memory bytes."""
    try:
        total = int(psutil.virtual_memory().total)
        if total > 0:
            return total
    except Exception:
        pass
    return None


def _poll_memory_state() -> dict[str, Any]:
    """Poll memory and update shared pressure state."""
    active_bytes, peak_bytes = _read_metal_memory_bytes()
    system_bytes = _system_memory_bytes()
    _memory_state.update(
        active_bytes=active_bytes,
        peak_bytes=peak_bytes,
        system_bytes=system_bytes,
        warn_threshold_pct=_memory_warn_threshold_pct,
        limit_threshold_pct=_memory_limit_threshold_pct,
        action=_memory_action,
    )
    return _memory_state.snapshot()


async def _memory_monitor_loop() -> None:
    """Background memory monitoring loop for pressure-based guardrails."""
    while True:
        snapshot = _poll_memory_state()
        pressure = snapshot.get("pressure")
        utilization = snapshot.get("utilization_pct")
        if pressure in {"elevated", "critical"} and utilization is not None:
            logger.warning(
                "[memory] pressure=%s utilization=%.1f%% action=%s",
                pressure,
                utilization,
                _memory_action,
            )
        await asyncio.sleep(max(1.0, _memory_monitor_interval_seconds))


def _token_agreement_rate(a: str, b: str) -> float:
    """Token-level agreement between two decoded texts."""
    ta = a.split()
    tb = b.split()
    max_len = max(len(ta), len(tb), 1)
    matches = 0
    for idx in range(max_len):
        tok_a = ta[idx] if idx < len(ta) else None
        tok_b = tb[idx] if idx < len(tb) else None
        if tok_a == tok_b:
            matches += 1
    return matches / max_len


async def _run_batch_divergence_probe_once() -> dict[str, Any]:
    """Run one serial-vs-concurrent probe and return metrics."""
    if _engine is None:
        raise RuntimeError("Model not loaded.")

    probe_kwargs = {
        "max_tokens": min(64, _default_max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
    }
    primary_messages = [{"role": "user", "content": _batch_divergence_probe_primary_prompt}]
    distractor_messages = [
        {"role": "user", "content": _batch_divergence_probe_distractor_prompt}
    ]

    serial_start = time.perf_counter()
    serial_output = await _engine.chat(messages=primary_messages, **probe_kwargs)
    serial_latency_s = time.perf_counter() - serial_start

    concurrent_start = time.perf_counter()
    concurrent_primary_output, _ = await asyncio.gather(
        _engine.chat(messages=primary_messages, **probe_kwargs),
        _engine.chat(messages=distractor_messages, **probe_kwargs),
    )
    concurrent_latency_s = time.perf_counter() - concurrent_start

    serial_text = clean_output_text(getattr(serial_output, "text", "") or "")
    concurrent_text = clean_output_text(
        getattr(concurrent_primary_output, "text", "") or ""
    )
    token_agreement = _token_agreement_rate(serial_text, concurrent_text)

    return {
        "token_agreement": token_agreement,
        "exact_match": serial_text == concurrent_text,
        "serial_latency_s": serial_latency_s,
        "concurrent_latency_s": concurrent_latency_s,
    }


async def _batch_divergence_monitor_loop() -> None:
    """Background loop that tracks batch divergence in batched runtime mode."""
    while True:
        sleep_s = max(5.0, _batch_divergence_interval_seconds)
        try:
            if not _batch_divergence_monitor_enabled:
                _batch_divergence_state.update_info(
                    status="pass",
                    detail="Batch divergence monitor disabled.",
                )
                await asyncio.sleep(sleep_s)
                continue

            if _engine is None:
                _batch_divergence_state.update_info(
                    status="warning",
                    detail="Model not loaded; cannot run batch divergence probe.",
                )
                await asyncio.sleep(sleep_s)
                continue

            engine_stats = _engine.get_stats()
            engine_type = str(engine_stats.get("engine_type", "unknown"))
            if engine_type != "batched":
                _batch_divergence_state.update_info(
                    status="pass",
                    detail=(
                        f"Runtime mode is {engine_type}; batch divergence probing "
                        "is only applicable in batched mode."
                    ),
                )
                await asyncio.sleep(sleep_s)
                continue

            probe = await _run_batch_divergence_probe_once()
            _batch_divergence_state.update_probe(
                token_agreement=float(probe["token_agreement"]),
                exact_match=bool(probe["exact_match"]),
                serial_latency_s=float(probe["serial_latency_s"]),
                concurrent_latency_s=float(probe["concurrent_latency_s"]),
            )

            if probe["token_agreement"] < _batch_divergence_threshold:
                logger.warning(
                    "[batch-divergence] agreement=%.2f%% threshold=%.2f%% action=%s",
                    probe["token_agreement"] * 100.0,
                    _batch_divergence_threshold * 100.0,
                    _batch_divergence_action,
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _batch_divergence_state.update_info(
                status="warning",
                detail=f"Batch divergence probe failed: {e}",
                error=str(e),
            )
            logger.warning("[batch-divergence] probe failed: %s", e, exc_info=True)
        await asyncio.sleep(sleep_s)


def _resolve_effective_max_tokens(request_value: int | None) -> int:
    """Apply memory-pressure policy to request max_tokens."""
    base = request_value or _default_max_tokens
    factor = _memory_state.max_tokens_factor()
    if factor >= 0.999:
        return base
    return max(1, int(base * factor))


def _collect_dtype_strings() -> tuple[str | None, str | None]:
    """Best-effort (loaded_dtype, expected_dtype) extraction from runtime state."""
    candidates: list[Any] = [_engine]
    if _engine is not None:
        candidates.extend(
            [
                getattr(_engine, "_model", None),
                getattr(getattr(_engine, "_model", None), "model", None),
                getattr(getattr(_engine, "_model", None), "config", None),
                getattr(getattr(getattr(_engine, "_model", None), "model", None), "config", None),
            ]
        )

    loaded_dtype = None
    expected_dtype = None
    for obj in candidates:
        if obj is None:
            continue
        for attr_name in ("dtype", "torch_dtype"):
            value = getattr(obj, attr_name, None)
            if value is not None and loaded_dtype is None:
                loaded_dtype = str(value)
        config = getattr(obj, "config", None)
        if config is not None:
            value = getattr(config, "torch_dtype", None)
            if value is not None and expected_dtype is None:
                expected_dtype = str(value)

    if loaded_dtype is None and _model_name:
        lowered = _model_name.lower()
        if "4bit" in lowered:
            loaded_dtype = "int4/quantized"
        elif "8bit" in lowered:
            loaded_dtype = "int8/quantized"

    return loaded_dtype, expected_dtype


def _check_dtype_status() -> DiagnosticCheck:
    """Run dtype consistency diagnostics."""
    loaded_dtype, expected_dtype = _collect_dtype_strings()
    architecture = _infer_model_architecture()

    if loaded_dtype is None and expected_dtype is None:
        return DiagnosticCheck(
            status="warning",
            detail="Could not determine loaded or expected dtype from runtime state.",
            metadata={"architecture": architecture},
        )

    loaded_lower = (loaded_dtype or "").lower()
    expected_lower = (expected_dtype or "").lower()
    if "float16" in loaded_lower and "bfloat16" in expected_lower:
        return DiagnosticCheck(
            status="warning",
            detail=(
                f"Loaded dtype {loaded_dtype} may be incompatible with expected "
                f"{expected_dtype}."
            ),
            metadata={"architecture": architecture},
        )

    if "float16" in loaded_lower and architecture in {"gemma3", "qwen3_vl_moe"}:
        return DiagnosticCheck(
            status="warning",
            detail=(
                "float16 detected on architecture with known sensitivity; "
                "prefer bfloat16 or quantized runtime path."
            ),
            metadata={"architecture": architecture},
        )

    detail_parts = []
    if loaded_dtype:
        detail_parts.append(f"loaded={loaded_dtype}")
    if expected_dtype:
        detail_parts.append(f"expected={expected_dtype}")
    if not detail_parts:
        detail_parts.append("dtype unavailable")

    return DiagnosticCheck(
        status="pass",
        detail=f"Dtype check OK ({', '.join(detail_parts)}).",
        metadata={"architecture": architecture},
    )


def _collect_stop_token_ids() -> tuple[set[int], set[int]]:
    """
    Collect EOS/stop token IDs from tokenizer and generation configuration.

    Returns:
        (configured_ids, generation_ids)
    """
    configured_ids: set[int] = set()
    generation_ids: set[int] = set()

    tokenizer = _get_text_tokenizer()
    if tokenizer is not None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_id, int):
            configured_ids.add(eos_id)
        eos_ids = getattr(tokenizer, "eos_token_ids", None)
        if isinstance(eos_ids, (list, tuple)):
            configured_ids.update(i for i in eos_ids if isinstance(i, int))

        gen_cfg = getattr(tokenizer, "generation_config", None)
        if gen_cfg is not None:
            gen_eos_id = getattr(gen_cfg, "eos_token_id", None)
            if isinstance(gen_eos_id, int):
                generation_ids.add(gen_eos_id)
            elif isinstance(gen_eos_id, (list, tuple)):
                generation_ids.update(i for i in gen_eos_id if isinstance(i, int))

    for obj in (
        getattr(_engine, "_model", None),
        getattr(getattr(_engine, "_model", None), "model", None),
        getattr(getattr(_engine, "_model", None), "config", None),
    ):
        if obj is None:
            continue
        gen_cfg = getattr(obj, "generation_config", None)
        if gen_cfg is None:
            continue
        gen_eos_id = getattr(gen_cfg, "eos_token_id", None)
        if isinstance(gen_eos_id, int):
            generation_ids.add(gen_eos_id)
        elif isinstance(gen_eos_id, (list, tuple)):
            generation_ids.update(i for i in gen_eos_id if isinstance(i, int))

    return configured_ids, generation_ids


def _check_eos_status() -> DiagnosticCheck:
    """Run EOS/stop-token and template readiness diagnostics."""
    configured_ids, generation_ids = _collect_stop_token_ids()
    tokenizer = _get_text_tokenizer()

    template_warning = None
    if tokenizer is None:
        template_warning = "Tokenizer unavailable for chat template verification."
    elif hasattr(tokenizer, "apply_chat_template"):
        try:
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "health-check"}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            # Some tokenizers don't accept add_generation_prompt/tools kwargs.
            try:
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": "health-check"}],
                    tokenize=False,
                )
            except Exception as e:
                template_warning = f"Chat template rendering failed: {e}"
        except Exception as e:
            template_warning = f"Chat template rendering failed: {e}"
    else:
        template_warning = "Tokenizer has no apply_chat_template; fallback formatting may be used."

    if configured_ids and generation_ids and not generation_ids.issuperset(configured_ids):
        return DiagnosticCheck(
            status="warning",
            detail=(
                f"EOS mismatch: configured={sorted(configured_ids)} "
                f"generation={sorted(generation_ids)}"
            ),
        )

    if template_warning:
        return DiagnosticCheck(status="warning", detail=template_warning)

    if configured_ids and generation_ids:
        detail = (
            f"EOS tokens aligned: configured={sorted(configured_ids)} "
            f"generation={sorted(generation_ids)}."
        )
    elif configured_ids:
        detail = f"EOS tokens present in tokenizer: {sorted(configured_ids)}."
    else:
        detail = "No explicit EOS IDs detected; relying on model defaults."
    return DiagnosticCheck(status="pass", detail=detail)


def _check_memory_status() -> tuple[DiagnosticCheck, DiagnosticMemory]:
    """Run memory utilization diagnostics and return check + structured memory payload."""
    snapshot = _poll_memory_state()
    active_bytes = snapshot.get("active_bytes")
    peak_bytes = snapshot.get("peak_bytes")
    system_bytes = snapshot.get("system_bytes")
    utilization_pct = snapshot.get("utilization_pct")
    pressure = snapshot.get("pressure", "unknown")
    trend = snapshot.get("trend", "unknown")

    memory_payload = DiagnosticMemory(
        active_gb=round(active_bytes / 1e9, 2) if isinstance(active_bytes, int) else None,
        peak_gb=round(peak_bytes / 1e9, 2) if isinstance(peak_bytes, int) else None,
        system_gb=round(system_bytes / 1e9, 2) if isinstance(system_bytes, int) else None,
        utilization_pct=round(utilization_pct, 2)
        if isinstance(utilization_pct, (float, int))
        else None,
        trend=trend,
        pressure=pressure,
    )

    if utilization_pct is None:
        return (
            DiagnosticCheck(
                status="warning",
                detail="Metal memory metrics unavailable in this runtime.",
            ),
            memory_payload,
        )

    detail = (
        f"Active: {memory_payload.active_gb}GB / {memory_payload.system_gb}GB "
        f"({memory_payload.utilization_pct}%), trend={trend}, pressure={pressure}"
    )

    if pressure == "critical":
        return DiagnosticCheck(status="fail", detail=detail), memory_payload
    if pressure == "elevated":
        return DiagnosticCheck(status="warning", detail=detail), memory_payload
    return DiagnosticCheck(status="pass", detail=detail), memory_payload


def _check_version_status() -> DiagnosticCheck:
    """Run runtime version diagnostics and known-bug matching."""
    versions = _collect_runtime_versions()
    architecture = _infer_model_architecture()
    known_bugs = _load_known_bug_entries()

    matching_bug_ids: list[str] = []
    for entry in known_bugs:
        if not isinstance(entry, dict):
            continue
        bug_id = str(entry.get("id", "unknown"))
        applies = entry.get("applies_to_architectures", [])
        if applies and architecture not in applies and "all" not in applies:
            continue

        constraints = entry.get("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}

        violated = True
        for package_name, constraint in constraints.items():
            if not isinstance(constraint, str):
                continue
            pkg_version = versions.get(package_name, "unknown")
            if not _version_satisfies_constraint(pkg_version, constraint):
                violated = False
                break
        if violated:
            matching_bug_ids.append(bug_id)

    detail = (
        f"mlx {versions.get('mlx')}, mlx-lm {versions.get('mlx-lm')}, "
        f"vllm-mlx {versions.get('vllm-mlx')}"
    )
    if matching_bug_ids:
        return DiagnosticCheck(
            status="warning",
            detail=f"{detail}; matched known issues: {', '.join(sorted(matching_bug_ids))}",
            metadata={"architecture": architecture, "known_bug_ids": matching_bug_ids},
        )
    return DiagnosticCheck(
        status="pass",
        detail=f"{detail}; no matching known-issue signatures for {architecture}.",
        metadata={"architecture": architecture},
    )


def _check_batch_invariance_status() -> DiagnosticCheck:
    """Run diagnostics check for batch divergence monitor state."""
    snapshot = _batch_divergence_state.snapshot()
    metadata = {
        "enabled": snapshot.get("enabled"),
        "threshold": snapshot.get("threshold"),
        "action": snapshot.get("action"),
        "deterministic_mode": _deterministic_mode,
        "deterministic_serialize": _deterministic_serialize,
        "engine_type": snapshot.get("engine_type"),
        "sample_count": snapshot.get("sample_count"),
        "last_checked_epoch": snapshot.get("last_checked_epoch"),
        "last_token_agreement": snapshot.get("last_token_agreement"),
        "last_exact_match": snapshot.get("last_exact_match"),
        "last_serial_latency_s": snapshot.get("last_serial_latency_s"),
        "last_concurrent_latency_s": snapshot.get("last_concurrent_latency_s"),
        "serialize_active": snapshot.get("serialize_active"),
    }
    if snapshot.get("error"):
        metadata["error"] = snapshot.get("error")

    if not snapshot.get("enabled"):
        return DiagnosticCheck(
            status="pass",
            detail="Batch divergence monitor disabled.",
            metadata=metadata,
        )

    if snapshot.get("engine_type") != "batched":
        return DiagnosticCheck(
            status="pass",
            detail=(
                f"Runtime mode is {snapshot.get('engine_type')}; "
                "batch divergence check is not applicable."
            ),
            metadata=metadata,
        )

    token_agreement = snapshot.get("last_token_agreement")
    sample_count = int(snapshot.get("sample_count") or 0)
    threshold = float(snapshot.get("threshold") or _batch_divergence_threshold)
    if sample_count <= 0 or token_agreement is None:
        return DiagnosticCheck(
            status="warning",
            detail="Batch divergence monitor enabled, awaiting first probe sample.",
            metadata=metadata,
        )

    if float(token_agreement) < threshold:
        detail = (
            f"Batch divergence detected: agreement {float(token_agreement) * 100.0:.2f}% "
            f"below threshold {threshold * 100.0:.2f}%."
        )
        if snapshot.get("serialize_active"):
            detail += " Serialize mitigation is active for tracked inference routes."
        return DiagnosticCheck(status="warning", detail=detail, metadata=metadata)

    return DiagnosticCheck(
        status="pass",
        detail=(
            f"Batch divergence within threshold: agreement "
            f"{float(token_agreement) * 100.0:.2f}%."
        ),
        metadata=metadata,
    )


def _aggregate_diagnostic_status(checks: dict[str, DiagnosticCheck]) -> str:
    """Reduce per-check states to top-level health status."""
    states = {c.status for c in checks.values()}
    if "fail" in states:
        return "unhealthy"
    if "warning" in states:
        return "degraded"
    return "healthy"


@app.get("/health")
async def health():
    """Health check endpoint."""
    mcp_info = None
    if _mcp_manager is not None:
        connected = sum(
            1 for s in _mcp_manager.get_server_status() if s.state.value == "connected"
        )
        total = len(_mcp_manager.get_server_status())
        mcp_info = {
            "enabled": True,
            "servers_connected": connected,
            "servers_total": total,
            "tools_available": len(_mcp_manager.get_all_tools()),
        }

    engine_stats = _engine.get_stats() if _engine else {}

    return {
        "status": "healthy",
        "model_loaded": _engine is not None,
        "model_name": _model_name,
        "model_type": "mllm" if (_engine and _engine.is_mllm) else "llm",
        "engine_type": engine_stats.get("engine_type", "unknown"),
        "mcp": mcp_info,
    }


@app.get("/health/diagnostics", dependencies=[Depends(verify_api_key)])
async def health_diagnostics() -> DiagnosticsHealthResponse:
    """
    Diagnostic health endpoint with lightweight quality checks.

    Auth behavior respects the global API key policy:
    - when auth is disabled, endpoint is effectively public
    - when auth is enabled, API key is required
    """
    checks: dict[str, DiagnosticCheck] = {}

    if _engine is None:
        checks["dtype"] = DiagnosticCheck(
            status="fail",
            detail="Model not loaded; cannot run dtype diagnostics.",
        )
        checks["eos"] = DiagnosticCheck(
            status="fail",
            detail="Model not loaded; cannot run EOS diagnostics.",
        )
    else:
        checks["dtype"] = _check_dtype_status()
        checks["eos"] = _check_eos_status()

    memory_check, memory_payload = _check_memory_status()
    checks["memory"] = memory_check
    checks["version"] = _check_version_status()
    checks["batch_invariance"] = _check_batch_invariance_status()

    return DiagnosticsHealthResponse(
        status=_aggregate_diagnostic_status(checks),
        model=_model_name,
        checks=checks,
        memory=memory_payload,
        timestamp=datetime.now(UTC).isoformat(),
    )


@app.get("/v1/status")
async def status():
    """Real-time status with per-request details for debugging and monitoring."""
    if _engine is None:
        return {"status": "not_loaded", "model": None, "requests": []}

    stats = _engine.get_stats()
    active_concurrency, peak_concurrency = _concurrency_tracker.snapshot()

    return {
        "status": "running" if stats.get("running") else "stopped",
        "model": _model_name,
        "uptime_s": round(stats.get("uptime_seconds", 0), 1),
        "steps_executed": stats.get("steps_executed", 0),
        "num_running": stats.get("num_running", 0),
        "num_waiting": stats.get("num_waiting", 0),
        "total_requests_processed": stats.get("num_requests_processed", 0),
        "total_prompt_tokens": stats.get("total_prompt_tokens", 0),
        "total_completion_tokens": stats.get("total_completion_tokens", 0),
        "metal": {
            "active_memory_gb": stats.get("metal_active_memory_gb"),
            "peak_memory_gb": stats.get("metal_peak_memory_gb"),
            "cache_memory_gb": stats.get("metal_cache_memory_gb"),
        },
        "cache": stats.get("memory_aware_cache")
        or stats.get("paged_cache")
        or stats.get("prefix_cache"),
        "runtime": {
            "active_concurrency": active_concurrency,
            "peak_concurrency": peak_concurrency,
        },
        "requests": stats.get("requests", []),
    }


@app.get("/v1/cache/stats")
async def cache_stats():
    """Get cache statistics for debugging and monitoring."""
    try:
        from mlx_vlm.utils import (
            get_multimodal_kv_cache_stats,
            get_pil_cache_stats,
            get_pixel_values_cache_stats,
        )

        return {
            "multimodal_kv_cache": get_multimodal_kv_cache_stats(),
            "pixel_values_cache": get_pixel_values_cache_stats(),
            "pil_image_cache": get_pil_cache_stats(),
        }
    except ImportError:
        return {"error": "Cache stats not available (mlx_vlm not loaded)"}


@app.delete("/v1/cache")
async def clear_cache():
    """Clear all caches."""
    try:
        from mlx_vlm.utils import (
            clear_multimodal_kv_cache,
            clear_pixel_values_cache,
        )

        clear_multimodal_kv_cache()
        clear_pixel_values_cache()
        return {
            "status": "cleared",
            "caches": ["multimodal_kv", "pixel_values", "pil_image"],
        }
    except ImportError:
        return {"error": "Cache clear not available (mlx_vlm not loaded)"}


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models() -> ModelsResponse:
    """List available models."""
    models = []
    if _model_name:
        models.append(ModelInfo(id=_model_name))
    return ModelsResponse(data=models)


@app.get("/v1/capabilities", dependencies=[Depends(verify_api_key)])
async def get_capabilities() -> CapabilitiesResponse:
    """Return current runtime capabilities for client feature negotiation."""
    model_loaded = _engine is not None
    model_is_mllm = bool(_engine and _engine.is_mllm)
    model_type = "mllm" if model_is_mllm else ("llm" if model_loaded else None)

    audio_available = _module_available("mlx_audio")
    embeddings_available = _module_available("mlx_embeddings")

    return CapabilitiesResponse(
        model_loaded=model_loaded,
        model_name=_model_name,
        model_type=model_type,
        modalities=CapabilityModalities(
            text=model_loaded,
            image=model_is_mllm,
            video=model_is_mllm,
            audio_input=audio_available,
            audio_output=audio_available,
        ),
        features=CapabilityFeatures(
            streaming=True,
            tool_calling=model_loaded,
            auto_tool_choice=_enable_auto_tool_choice,
            structured_output=True,
            reasoning=_reasoning_parser is not None,
            embeddings=embeddings_available,
            anthropic_messages=True,
            mcp=_mcp_manager is not None,
        ),
        auth=CapabilityAuth(api_key_required=_api_key is not None),
        rate_limit=CapabilityRateLimit(
            enabled=_rate_limiter.enabled,
            requests_per_minute=(
                _rate_limiter.requests_per_minute if _rate_limiter.enabled else None
            ),
        ),
        limits=CapabilityLimits(
            default_max_tokens=_default_max_tokens,
            default_timeout_seconds=_default_timeout,
        ),
    )


# =============================================================================
# Embeddings Endpoint
# =============================================================================


@app.post(
    "/v1/embeddings",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Create embeddings for the given input text(s).

    OpenAI-compatible embeddings API supporting single or batch inputs.

    Single text:
    ```json
    {
      "model": "mlx-community/all-MiniLM-L6-v2-4bit",
      "input": "The quick brown fox jumps over the lazy dog"
    }
    ```

    Batch of texts:
    ```json
    {
      "model": "mlx-community/embeddinggemma-300m-6bit",
      "input": [
        "I love machine learning",
        "Deep learning is fascinating",
        "Neural networks are powerful"
      ]
    }
    ```

    Response:
    ```json
    {
      "object": "list",
      "data": [
        {"object": "embedding", "index": 0, "embedding": [0.023, -0.982, ...]},
        {"object": "embedding", "index": 1, "embedding": [0.112, -0.543, ...]},
        {"object": "embedding", "index": 2, "embedding": [0.876, 0.221, ...]}
      ],
      "model": "mlx-community/embeddinggemma-300m-6bit",
      "usage": {"prompt_tokens": 24, "total_tokens": 24}
    }
    ```

    Supported models:
    - mlx-community/all-MiniLM-L6-v2-4bit (fast, compact)
    - mlx-community/embeddinggemma-300m-6bit (high quality)
    - mlx-community/bge-large-en-v1.5-4bit (best for English)
    - Any BERT/XLM-RoBERTa/ModernBERT model from HuggingFace
    """
    global _embedding_engine

    try:
        # Resolve model name
        model_name = request.model

        # If an embedding model was pre-configured at startup, only allow that model
        if (
            _embedding_model_locked is not None
            and model_name != _embedding_model_locked
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Embedding model '{model_name}' is not available. "
                    f"This server was started with --embedding-model {_embedding_model_locked}. "
                    f"Only '{_embedding_model_locked}' can be used for embeddings. "
                    f"Restart the server with a different --embedding-model to use '{model_name}'."
                ),
            )

        # Lazy-load or swap embedding engine
        load_embedding_model(model_name, lock=False, reuse_existing=True)

        # Normalise input to list
        texts = request.input if isinstance(request.input, list) else [request.input]

        if not texts:
            raise HTTPException(status_code=400, detail="Input must not be empty")

        start_time = time.perf_counter()

        # Count tokens for usage reporting
        prompt_tokens = _embedding_engine.count_tokens(texts)

        # Generate embeddings (batch)
        embeddings = _embedding_engine.embed(texts)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Embeddings: {len(texts)} inputs, {prompt_tokens} tokens "
            f"in {elapsed:.2f}s"
        )

        # Build OpenAI-compatible response with ordered indices
        data = [
            EmbeddingData(index=i, embedding=vec) for i, vec in enumerate(embeddings)
        ]

        return EmbeddingResponse(
            data=data,
            model=model_name,
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail=(
                "mlx-embeddings not installed. "
                "Install with: pip install mlx-embeddings"
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MCP Endpoints
# =============================================================================


@app.get("/v1/mcp/tools", dependencies=[Depends(verify_api_key)])
async def list_mcp_tools() -> MCPToolsResponse:
    """List all available MCP tools."""
    if _mcp_manager is None:
        return MCPToolsResponse(tools=[], count=0)

    tools = []
    for tool in _mcp_manager.get_all_tools():
        tools.append(
            MCPToolInfo(
                name=tool.full_name,
                description=tool.description,
                server=tool.server_name,
                parameters=tool.input_schema,
            )
        )

    return MCPToolsResponse(tools=tools, count=len(tools))


@app.get("/v1/mcp/servers", dependencies=[Depends(verify_api_key)])
async def list_mcp_servers() -> MCPServersResponse:
    """Get status of all MCP servers."""
    if _mcp_manager is None:
        return MCPServersResponse(servers=[])

    servers = []
    for status in _mcp_manager.get_server_status():
        servers.append(
            MCPServerInfo(
                name=status.name,
                state=status.state.value,
                transport=status.transport.value,
                tools_count=status.tools_count,
                error=status.error,
            )
        )

    return MCPServersResponse(servers=servers)


@app.post("/v1/mcp/execute", dependencies=[Depends(verify_api_key)])
async def execute_mcp_tool(request: MCPExecuteRequest) -> MCPExecuteResponse:
    """Execute an MCP tool."""
    if _mcp_manager is None:
        raise HTTPException(
            status_code=503, detail="MCP not configured. Start server with --mcp-config"
        )

    result = await _mcp_manager.execute_tool(
        request.tool_name,
        request.arguments,
    )

    return MCPExecuteResponse(
        tool_name=result.tool_name,
        content=result.content,
        is_error=result.is_error,
        error_message=result.error_message,
    )


# =============================================================================
# Audio Endpoints
# =============================================================================

# Global audio engines (lazy loaded)
_stt_engine = None
_tts_engine = None


@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
async def create_transcription(
    file: UploadFile,
    model: str = "whisper-large-v3",
    language: str | None = None,
    response_format: str = "json",
):
    """
    Transcribe audio to text (OpenAI Whisper API compatible).

    Supported models:
    - whisper-large-v3 (multilingual, best quality)
    - whisper-large-v3-turbo (faster)
    - whisper-medium, whisper-small (lighter)
    - parakeet-tdt-0.6b-v2 (English, fastest)
    """
    global _stt_engine

    try:
        from .audio.stt import STTEngine  # Lazy import - optional feature

        # Map model aliases to full names
        model_map = {
            "whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
            "whisper-large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "whisper-medium": "mlx-community/whisper-medium-mlx",
            "whisper-small": "mlx-community/whisper-small-mlx",
            "parakeet": "mlx-community/parakeet-tdt-0.6b-v2",
            "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",
        }
        model_name = model_map.get(model, model)

        # Load engine if needed
        if _stt_engine is None or _stt_engine.model_name != model_name:
            _stt_engine = STTEngine(model_name)
            _stt_engine.load()

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = _stt_engine.transcribe(tmp_path, language=language)
        finally:
            os.unlink(tmp_path)

        if response_format == "text":
            return result.text

        return {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
        }

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="mlx-audio not installed. Install with: pip install mlx-audio",
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def create_speech(
    model: str = "kokoro",
    input: str = "",
    voice: str = "af_heart",
    speed: float = 1.0,
    response_format: str = "wav",
):
    """
    Generate speech from text (OpenAI TTS API compatible).

    Supported models:
    - kokoro (fast, lightweight)
    - chatterbox (multilingual, expressive)
    - vibevoice (realtime)
    - voxcpm (Chinese/English)
    """
    global _tts_engine

    try:
        from .audio.tts import TTSEngine  # Lazy import - optional feature

        # Map model aliases to full names
        model_map = {
            "kokoro": "mlx-community/Kokoro-82M-bf16",
            "kokoro-4bit": "mlx-community/Kokoro-82M-4bit",
            "chatterbox": "mlx-community/chatterbox-turbo-fp16",
            "chatterbox-4bit": "mlx-community/chatterbox-turbo-4bit",
            "vibevoice": "mlx-community/VibeVoice-Realtime-0.5B-4bit",
            "voxcpm": "mlx-community/VoxCPM1.5",
        }
        model_name = model_map.get(model, model)

        # Load engine if needed
        if _tts_engine is None or _tts_engine.model_name != model_name:
            _tts_engine = TTSEngine(model_name)
            _tts_engine.load()

        audio = _tts_engine.generate(input, voice=voice, speed=speed)
        audio_bytes = _tts_engine.to_bytes(audio, format=response_format)

        content_type = (
            "audio/wav" if response_format == "wav" else f"audio/{response_format}"
        )
        return Response(content=audio_bytes, media_type=content_type)

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="mlx-audio not installed. Install with: pip install mlx-audio",
        )
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/audio/voices", dependencies=[Depends(verify_api_key)])
async def list_voices(model: str = "kokoro"):
    """List available voices for a TTS model."""
    from .audio.tts import CHATTERBOX_VOICES, KOKORO_VOICES

    if "kokoro" in model.lower():
        return {"voices": KOKORO_VOICES}
    elif "chatterbox" in model.lower():
        return {"voices": CHATTERBOX_VOICES}
    else:
        return {"voices": ["default"]}


# =============================================================================
# Streaming disconnect detection
# =============================================================================


async def _disconnect_guard(
    generator: AsyncIterator[str],
    raw_request: Request,
    poll_interval: float = 0.5,
) -> AsyncIterator[str]:
    """Wrap streaming generator to abort on client disconnect.

    Uses asyncio racing: each __anext__() on the inner generator is
    raced against a disconnect poller.  This catches disconnects even
    during prefill when no chunks are being yielded for tens of seconds.

    On disconnect, aclose() propagates down the generator chain to
    engine_core.stream_outputs() finally-block → abort_request().
    """
    import time as _time

    _t0 = _time.monotonic()

    def _elapsed():
        return f"{_time.monotonic() - _t0:.1f}s"

    logger.info(f"[disconnect_guard] START poll_interval={poll_interval}s")

    async def _wait_disconnect():
        poll_count = 0
        while True:
            await asyncio.sleep(poll_interval)
            poll_count += 1
            is_disc = await raw_request.is_disconnected()
            if poll_count % 10 == 0 or is_disc:
                logger.info(
                    f"[disconnect_guard] poll #{poll_count} "
                    f"disconnected={is_disc} elapsed={_elapsed()}"
                )
            if is_disc:
                return

    chunk_count = 0
    disconnect_task: asyncio.Task | None = None
    anext_task: asyncio.Task | None = None
    try:
        aiter = generator.__aiter__()
        disconnect_task = asyncio.create_task(_wait_disconnect())
        while True:
            anext_task = asyncio.ensure_future(aiter.__anext__())
            done, _ = await asyncio.wait(
                [anext_task, disconnect_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                logger.info(
                    f"[disconnect_guard] CLIENT DISCONNECTED after "
                    f"{chunk_count} chunks, elapsed={_elapsed()}"
                )
                anext_task.cancel()
                try:
                    await anext_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
                break
            try:
                chunk = anext_task.result()
            except StopAsyncIteration:
                logger.info(
                    f"[disconnect_guard] generator exhausted normally, "
                    f"{chunk_count} chunks, elapsed={_elapsed()}"
                )
                break
            chunk_count += 1
            if chunk_count == 1:
                logger.info(
                    f"[disconnect_guard] first chunk arrived, elapsed={_elapsed()}"
                )
            yield chunk
    except GeneratorExit:
        logger.info(
            f"[disconnect_guard] GeneratorExit after {chunk_count} chunks, elapsed={_elapsed()}"
        )
    finally:
        if disconnect_task and not disconnect_task.done():
            disconnect_task.cancel()
        if anext_task and not anext_task.done():
            anext_task.cancel()
        # NOTE: Do NOT call generator.aclose() here.  With run_in_executor,
        # scheduler.step() runs in a background thread.  aclose() would throw
        # GeneratorExit into the async-generator chain, which can trigger
        # mlx::core::eval on the main thread while the executor thread is also
        # mid-eval → Metal assertion failure → SIGABRT.
        #
        # Instead, rely on the task cancellation propagation:
        #   anext_task.cancel() → CancelledError in stream_outputs()
        #   → finally block → abort_request() → request removed from scheduler
        logger.info(
            f"[disconnect_guard] CLEANUP done, {chunk_count} chunks total, elapsed={_elapsed()}"
        )


async def _wait_with_disconnect(
    coro,
    raw_request: Request,
    timeout: float,
    poll_interval: float = 0.5,
):
    """Run a coroutine with both timeout and client disconnect detection.

    For non-streaming requests where _disconnect_guard() can't be used.
    Races the coroutine against a disconnect poller, same pattern as
    _disconnect_guard but for awaitable (non-generator) coroutines.
    """
    import time as _time

    _t0 = _time.monotonic()

    task = asyncio.ensure_future(coro)

    async def _wait_disconnect():
        poll_count = 0
        while True:
            await asyncio.sleep(poll_interval)
            poll_count += 1
            is_disc = await raw_request.is_disconnected()
            if poll_count % 10 == 0 or is_disc:
                logger.info(
                    f"[disconnect_guard] poll #{poll_count} "
                    f"disconnected={is_disc} elapsed={_time.monotonic() - _t0:.1f}s"
                )
            if is_disc:
                return

    disconnect_task = asyncio.create_task(_wait_disconnect())

    try:
        done, _ = await asyncio.wait(
            [task, disconnect_task],
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            # Timeout
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            raise HTTPException(
                status_code=504,
                detail=f"Request timed out after {timeout:.1f} seconds",
            )

        if disconnect_task in done:
            # Client disconnected
            logger.info(
                f"[disconnect_guard] CLIENT DISCONNECTED (non-stream) "
                f"elapsed={_time.monotonic() - _t0:.1f}s"
            )
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            return None  # Signal to caller that client disconnected

        # Task completed
        return task.result()

    finally:
        if not disconnect_task.done():
            disconnect_task.cancel()
        if not task.done():
            task.cancel()


# =============================================================================
# Completion Endpoints
# =============================================================================


@app.post(
    "/v1/completions", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Create a text completion."""
    engine = get_engine()

    # Handle single prompt or list of prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

    # --- Detailed request logging ---
    prompt_preview = prompts[0][:200] if prompts else "(empty)"
    prompt_len = sum(len(p) for p in prompts)
    logger.info(
        f"[REQUEST] POST /v1/completions stream={request.stream} "
        f"max_tokens={request.max_tokens} temp={request.temperature} "
        f"prompt_chars={prompt_len} prompt_preview={prompt_preview!r}"
    )

    if request.stream:
        return StreamingResponse(
            _disconnect_guard(
                stream_completion(engine, prompts[0], request),
                raw_request,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout or _default_timeout
    choices = []
    total_completion_tokens = 0
    total_prompt_tokens = 0

    for i, prompt in enumerate(prompts):
        output = await _wait_with_disconnect(
            engine.generate(
                prompt=prompt,
                max_tokens=_resolve_effective_max_tokens(request.max_tokens),
                temperature=_resolve_temperature(request.temperature),
                top_p=_resolve_top_p(request.top_p),
                repetition_penalty=_resolve_repetition_penalty(
                    request.repetition_penalty,
                    request.frequency_penalty,
                ),
                stop=request.stop,
            ),
            raw_request,
            timeout=timeout,
        )
        if output is None:
            return Response(status_code=499)  # Client closed request

        choices.append(
            CompletionChoice(
                index=i,
                text=output.text,
                finish_reason=output.finish_reason,
            )
        )
        total_completion_tokens += output.completion_tokens
        total_prompt_tokens += (
            output.prompt_tokens if hasattr(output, "prompt_tokens") else 0
        )

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = total_completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Completion: {total_prompt_tokens} prompt + {total_completion_tokens} completion tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    return CompletionResponse(
        model=request.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
    )


@app.post(
    "/v1/chat/completions",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """
    Create a chat completion (supports multimodal content for VLM models).

    OpenAI-compatible multimodal format for images:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }]
    ```

    Video support:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]
    }]
    ```

    Structured output (JSON mode):
    ```json
    response_format={"type": "json_object"}
    ```

    Structured output (JSON Schema):
    ```json
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "my_schema",
            "schema": {"type": "object", "properties": {...}}
        }
    }
    ```
    """
    engine = get_engine()

    # --- Detailed request logging ---
    n_msgs = len(request.messages)
    msg_roles = [m.role for m in request.messages]
    total_chars = 0
    last_user_preview = ""
    for m in request.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total_chars += len(content)
        if m.role == "user":
            last_user_preview = content[:300]
    has_tools = bool(request.tools)
    n_tools = len(request.tools) if request.tools else 0
    logger.info(
        f"[REQUEST] POST /v1/chat/completions stream={request.stream} "
        f"model={request.model!r} max_tokens={request.max_tokens} "
        f"temp={request.temperature} msgs={n_msgs} roles={msg_roles} "
        f"total_chars={total_chars} tools={n_tools} "
        f"response_format={request.response_format}"
    )
    logger.info(f"[REQUEST] last user message preview: {last_user_preview!r}")

    # For MLLM models, keep original messages with embedded images
    # (MLLM.chat() extracts images from message content internally)
    if engine.is_mllm:
        # Convert Pydantic messages to dicts, excluding None fields
        # to prevent chat templates from misinterpreting key presence
        # (e.g. image_url: null on text parts triggers Qwen3-VL crash)
        messages = []
        for msg in request.messages:
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump(exclude_none=True)
            else:
                raw = dict(msg)
                msg_dict = {k: v for k, v in raw.items() if v is not None}
            messages.append(msg_dict)
        images, videos = [], []  # MLLM extracts these from messages
        logger.debug(f"MLLM: Processing {len(messages)} messages")
    else:
        # For LLM, extract text, images, and videos separately
        messages, images, videos = extract_multimodal_content(
            request.messages,
            preserve_native_format=engine.preserve_native_tool_format,
        )

    has_media = bool(images or videos)

    # Handle response_format - inject system prompt if needed
    response_format = request.response_format
    if response_format:
        json_instruction = build_json_system_prompt(response_format)
        if json_instruction:
            # Inject JSON instruction into messages
            messages = _inject_json_instruction(messages, json_instruction)

    # Prepare kwargs
    chat_kwargs = {
        "max_tokens": _resolve_effective_max_tokens(request.max_tokens),
        "temperature": _resolve_temperature(request.temperature),
        "top_p": _resolve_top_p(request.top_p),
        "stop": request.stop,
    }
    repetition_penalty = _resolve_repetition_penalty(
        request.repetition_penalty,
        request.frequency_penalty,
    )
    if repetition_penalty is not None:
        chat_kwargs["repetition_penalty"] = repetition_penalty
    chat_kwargs.update(_build_engine_thinking_kwargs(request.max_thinking_tokens))

    # Add multimodal content
    if has_media:
        chat_kwargs["images"] = images if images else None
        chat_kwargs["videos"] = videos if videos else None
        if request.video_fps:
            chat_kwargs["video_fps"] = request.video_fps
        if request.video_max_frames:
            chat_kwargs["video_max_frames"] = request.video_max_frames

    # Add tools if provided
    if request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(request.tools)
        if request.tool_choice is not None:
            chat_kwargs["tool_choice"] = request.tool_choice

    if request.stream:
        return StreamingResponse(
            _disconnect_guard(
                stream_chat_completion(engine, messages, request, **chat_kwargs),
                raw_request,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout or _default_timeout

    output = await _wait_with_disconnect(
        engine.chat(messages=messages, **chat_kwargs),
        raw_request,
        timeout=timeout,
    )
    if output is None:
        return Response(status_code=499)  # Client closed request

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Chat completion: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Extract reasoning before tool parsing so parser markup in final content
    # is not hidden behind reasoning wrappers.
    reasoning_text = None
    text_for_tool_parsing = output.text
    if _reasoning_parser:
        reasoning_text, text_for_tool_parsing = _reasoning_parser.extract_reasoning(
            output.text
        )

        max_thinking_tokens = _resolve_max_thinking_tokens(request.max_thinking_tokens)
        if max_thinking_tokens is not None and reasoning_text:
            tokenizer = _get_text_tokenizer()
            kept_reasoning, overflow_reasoning = _split_text_by_token_budget(
                reasoning_text,
                max_thinking_tokens,
                tokenizer,
            )
            if overflow_reasoning:
                logger.info(
                    "Applied max thinking budget: kept %s tokens of reasoning, "
                    "routed overflow to content/tool parsing",
                    max_thinking_tokens,
                )
                reasoning_text = kept_reasoning or None
                merged_content = (overflow_reasoning + (text_for_tool_parsing or "")).strip()
                text_for_tool_parsing = merged_content or None

    # Parse tool calls from (possibly reasoning-cleaned) output
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(
        text_for_tool_parsing or "",
        request,
    )

    # Process response_format if specified (after reasoning parser cleaned the text)
    if response_format and not tool_calls:
        json_input = cleaned_text or text_for_tool_parsing or output.text
        _, parsed_json, is_valid, error = parse_json_output(json_input, response_format)
        if parsed_json is not None:
            # Return JSON as string
            cleaned_text = json.dumps(parsed_json)
        if not is_valid:
            logger.warning(f"JSON validation failed: {error}")

    # Determine finish reason
    finish_reason = "tool_calls" if tool_calls else output.finish_reason

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=clean_output_text(cleaned_text) if cleaned_text else None,
                    reasoning=reasoning_text,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            total_tokens=output.prompt_tokens + output.completion_tokens,
        ),
    )


def _inject_json_instruction(messages: list, instruction: str) -> list:
    """
    Inject JSON instruction into messages.

    If a system message exists, append to it. Otherwise, prepend a new system message.
    """
    messages = list(messages)  # Make a copy

    # Find existing system message
    system_idx = None
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "system":
            system_idx = i
            break

    if system_idx is not None:
        # Append to existing system message
        msg = messages[system_idx]
        if isinstance(msg, dict):
            existing = msg.get("content", "")
            msg["content"] = f"{existing}\n\n{instruction}"
        else:
            existing = getattr(msg, "content", "") or ""
            msg.content = f"{existing}\n\n{instruction}"
    else:
        # Prepend new system message
        messages.insert(0, {"role": "system", "content": instruction})

    return messages


# =============================================================================
# Anthropic Messages API Endpoints
# =============================================================================


@app.post(
    "/v1/messages",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_anthropic_message(
    request: Request,
):
    """
    Anthropic Messages API endpoint.

    Translates Anthropic-format requests to OpenAI format, runs inference
    through the existing engine, and converts the response back.

    Supports both streaming and non-streaming modes.
    """
    engine = get_engine()

    # Parse the raw body to handle Anthropic request format
    body = await request.json()
    anthropic_request = AnthropicRequest(**body)

    # --- Detailed request logging ---
    n_msgs = len(anthropic_request.messages)
    total_chars = 0
    last_user_preview = ""
    for m in anthropic_request.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total_chars += len(content)
        if m.role == "user":
            last_user_preview = content[:300]
    sys_chars = len(anthropic_request.system) if anthropic_request.system else 0
    n_tools = len(anthropic_request.tools) if anthropic_request.tools else 0
    logger.info(
        f"[REQUEST] POST /v1/messages (anthropic) stream={anthropic_request.stream} "
        f"model={anthropic_request.model!r} max_tokens={anthropic_request.max_tokens} "
        f"msgs={n_msgs} total_chars={total_chars} system_chars={sys_chars} "
        f"tools={n_tools}"
    )
    logger.info(f"[REQUEST] last user message preview: {last_user_preview!r}")

    # Convert Anthropic request -> OpenAI request
    openai_request = anthropic_to_openai(anthropic_request)

    if anthropic_request.stream:
        return StreamingResponse(
            _disconnect_guard(
                _stream_anthropic_messages(engine, openai_request, anthropic_request),
                request,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming: run inference through existing engine
    messages, images, videos = extract_multimodal_content(
        openai_request.messages,
        preserve_native_format=engine.preserve_native_tool_format,
    )

    chat_kwargs = {
        "max_tokens": _resolve_effective_max_tokens(openai_request.max_tokens),
        "temperature": openai_request.temperature,
        "top_p": openai_request.top_p,
    }
    repetition_penalty = _resolve_repetition_penalty(
        openai_request.repetition_penalty,
        openai_request.frequency_penalty,
    )
    if repetition_penalty is not None:
        chat_kwargs["repetition_penalty"] = repetition_penalty
    chat_kwargs.update(
        _build_engine_thinking_kwargs(getattr(openai_request, "max_thinking_tokens", None))
    )

    if openai_request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)
        if openai_request.tool_choice is not None:
            chat_kwargs["tool_choice"] = openai_request.tool_choice

    start_time = time.perf_counter()
    timeout = _default_timeout

    output = await _wait_with_disconnect(
        engine.chat(messages=messages, **chat_kwargs),
        request,
        timeout=timeout,
    )
    if output is None:
        return Response(status_code=499)  # Client closed request

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Anthropic messages: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Extract reasoning before tool parsing to avoid parser-order conflicts.
    text_for_tool_parsing = output.text
    if _reasoning_parser:
        reasoning_text, text_for_tool_parsing = _reasoning_parser.extract_reasoning(
            output.text
        )
        max_thinking_tokens = _resolve_max_thinking_tokens(
            getattr(openai_request, "max_thinking_tokens", None)
        )
        if max_thinking_tokens is not None and reasoning_text:
            tokenizer = _get_text_tokenizer()
            _, overflow_reasoning = _split_text_by_token_budget(
                reasoning_text,
                max_thinking_tokens,
                tokenizer,
            )
            if overflow_reasoning:
                merged_content = (overflow_reasoning + (text_for_tool_parsing or "")).strip()
                text_for_tool_parsing = merged_content or None

    cleaned_text, tool_calls = _parse_tool_calls_with_parser(
        text_for_tool_parsing or "",
        openai_request,
    )

    # Clean output text
    final_content = None
    if cleaned_text:
        final_content = clean_output_text(cleaned_text)

    # Determine finish reason
    finish_reason = "tool_calls" if tool_calls else output.finish_reason

    # Build OpenAI response to convert
    openai_response = ChatCompletionResponse(
        model=openai_request.model,
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=final_content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            total_tokens=output.prompt_tokens + output.completion_tokens,
        ),
    )

    # Convert to Anthropic response
    anthropic_response = openai_to_anthropic(openai_response, anthropic_request.model)
    return Response(
        content=anthropic_response.model_dump_json(exclude_none=True),
        media_type="application/json",
    )


@app.post(
    "/v1/messages/count_tokens",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def count_anthropic_tokens(request: Request):
    """
    Count tokens for an Anthropic Messages API request.

    Uses the model's tokenizer for accurate counting.
    Claude Code calls this endpoint for token budgeting.
    Note: Don't parse via AnthropicRequest — count_tokens requests
    from Claude Code don't include max_tokens.
    """
    body = await request.json()

    engine = get_engine()
    tokenizer = engine.tokenizer

    total_tokens = 0

    # System message
    system = body.get("system", "")
    if isinstance(system, str) and system:
        total_tokens += len(tokenizer.encode(system))
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                text = block.get("text", "")
                if text:
                    total_tokens += len(tokenizer.encode(text))

    # Messages
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            if content:
                total_tokens += len(tokenizer.encode(content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        total_tokens += len(tokenizer.encode(text))
                    # tool_use input
                    if block.get("input"):
                        total_tokens += len(
                            tokenizer.encode(json.dumps(block["input"]))
                        )
                    # tool_result content
                    sub_content = block.get("content", "")
                    if isinstance(sub_content, str) and sub_content:
                        total_tokens += len(tokenizer.encode(sub_content))
                    elif isinstance(sub_content, list):
                        for item in sub_content:
                            if isinstance(item, dict):
                                item_text = item.get("text", "")
                                if item_text:
                                    total_tokens += len(tokenizer.encode(item_text))

    # Tools
    for tool in body.get("tools", []):
        name = tool.get("name", "")
        if name:
            total_tokens += len(tokenizer.encode(name))
        desc = tool.get("description", "")
        if desc:
            total_tokens += len(tokenizer.encode(desc))
        if tool.get("input_schema"):
            total_tokens += len(tokenizer.encode(json.dumps(tool["input_schema"])))

    return {"input_tokens": total_tokens}


async def _stream_anthropic_messages(
    engine: BaseEngine,
    openai_request: ChatCompletionRequest,
    anthropic_request: AnthropicRequest,
) -> AsyncIterator[str]:
    """
    Stream Anthropic Messages API SSE events.

    Converts OpenAI streaming chunks to Anthropic event format:
    message_start -> content_block_start -> content_block_delta* ->
    content_block_stop -> message_delta -> message_stop
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    start_time = time.perf_counter()

    # Extract messages for engine
    messages, images, videos = extract_multimodal_content(
        openai_request.messages,
        preserve_native_format=engine.preserve_native_tool_format,
    )

    chat_kwargs = {
        "max_tokens": _resolve_effective_max_tokens(openai_request.max_tokens),
        "temperature": openai_request.temperature,
        "top_p": openai_request.top_p,
    }
    repetition_penalty = _resolve_repetition_penalty(
        openai_request.repetition_penalty,
        openai_request.frequency_penalty,
    )
    if repetition_penalty is not None:
        chat_kwargs["repetition_penalty"] = repetition_penalty
    chat_kwargs.update(
        _build_engine_thinking_kwargs(getattr(openai_request, "max_thinking_tokens", None))
    )

    if openai_request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)
        if openai_request.tool_choice is not None:
            chat_kwargs["tool_choice"] = openai_request.tool_choice

    # Emit message_start
    message_start = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": anthropic_request.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    # Emit content_block_start for text
    content_block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"

    # Stream content deltas
    accumulated_text = ""
    completion_tokens = 0

    async for output in engine.stream_chat(messages=messages, **chat_kwargs):
        delta_text = output.new_text

        # Track token counts
        if hasattr(output, "completion_tokens") and output.completion_tokens:
            completion_tokens = output.completion_tokens

        if delta_text:
            # Filter special tokens
            content = SPECIAL_TOKENS_PATTERN.sub("", delta_text)

            if content:
                accumulated_text += content
                delta_event = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": content},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

    # Check for tool calls in accumulated text
    _, tool_calls = _parse_tool_calls_with_parser(accumulated_text, openai_request)

    # Emit content_block_stop for text block
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

    # If there are tool calls, emit tool_use blocks
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            tool_index = i + 1
            try:
                tool_input = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                tool_input = {}

            # content_block_start for tool_use
            tool_block_start = {
                "type": "content_block_start",
                "index": tool_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": {},
                },
            }
            yield f"event: content_block_start\ndata: {json.dumps(tool_block_start)}\n\n"

            # Send input as a single delta
            input_json = json.dumps(tool_input)
            input_delta = {
                "type": "content_block_delta",
                "index": tool_index,
                "delta": {"type": "input_json_delta", "partial_json": input_json},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(input_delta)}\n\n"

            # content_block_stop
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tool_index})}\n\n"

    # Determine stop reason
    stop_reason = "tool_use" if tool_calls else "end_turn"

    # Emit message_delta with stop_reason and usage
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": completion_tokens},
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

    # Log throughput
    elapsed = time.perf_counter() - start_time
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Anthropic messages (stream): {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Emit message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


# =============================================================================
# Streaming Helpers
# =============================================================================


async def stream_completion(
    engine: BaseEngine,
    prompt: str,
    request: CompletionRequest,
) -> AsyncIterator[str]:
    """Stream completion response."""
    repetition_penalty = _resolve_repetition_penalty(
        request.repetition_penalty,
        request.frequency_penalty,
    )
    async for output in engine.stream_generate(
        prompt=prompt,
        max_tokens=_resolve_effective_max_tokens(request.max_tokens),
        temperature=_resolve_temperature(request.temperature),
        top_p=_resolve_top_p(request.top_p),
        repetition_penalty=repetition_penalty,
        stop=request.stop,
    ):
        data = {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "text": output.new_text,
                    "finish_reason": output.finish_reason if output.finished else None,
                }
            ],
        }
        if output.finished:
            data["usage"] = get_usage(output).model_dump()
        yield f"data: {json.dumps(data)}\n\n"

    yield "data: [DONE]\n\n"


async def stream_chat_completion(
    engine: BaseEngine,
    messages: list,
    request: ChatCompletionRequest,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion response."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    # Check if we should include usage in the final chunk
    include_usage = request.stream_options and request.stream_options.include_usage

    # First chunk with role
    first_chunk = ChatCompletionChunk(
        id=response_id,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(role="assistant"),
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # Track if we need to add <think> prefix for thinking models (when no reasoning parser)
    # The template adds <think> to the prompt, so the model output starts inside the think block
    is_thinking_model = "nemotron" in request.model.lower() and not _reasoning_parser
    think_prefix_sent = False

    # Reset reasoning parser state for this stream
    if _reasoning_parser:
        _reasoning_parser.reset_state()

    # Track accumulated text for reasoning parser
    accumulated_text = ""

    # Track token counts for usage reporting
    prompt_tokens = 0
    completion_tokens = 0
    last_output = None

    # Tool call streaming state
    global _tool_parser_instance
    tool_parser = None
    tool_accumulated_text = ""
    tool_calls_detected = False
    tool_markup_possible = False  # Fast path: skip parsing until '<' seen
    tools_for_schema_coercion = (
        request.model_dump().get("tools") if request and request.tools else None
    )
    if _enable_auto_tool_choice and _tool_call_parser:
        # Initialize parser if needed (same as _parse_tool_calls_with_parser)
        if _tool_parser_instance is None:
            try:
                parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
                tokenizer = None
                if _engine is not None and hasattr(_engine, "_tokenizer"):
                    tokenizer = _engine._tokenizer
                _tool_parser_instance = parser_cls(tokenizer)
                logger.info(f"Initialized tool call parser: {_tool_call_parser}")
            except Exception as e:
                logger.warning(f"Failed to init tool parser for streaming: {e}")
        if _tool_parser_instance is not None:
            tool_parser = _tool_parser_instance
            tool_parser.reset()

    # Reasoning budget state
    max_thinking_tokens = _resolve_max_thinking_tokens(request.max_thinking_tokens)
    reasoning_tokenizer = _get_text_tokenizer() if max_thinking_tokens else None
    reasoning_tokens_used = 0

    def _route_tool_stream_delta(
        content_delta: str,
    ) -> tuple[str | None, list | None, bool]:
        """
        Route a content delta through the configured streaming tool parser.

        Returns:
            (content, tool_calls, suppress_output)
        """
        nonlocal tool_accumulated_text, tool_calls_detected, tool_markup_possible

        if not tool_parser or not content_delta:
            return content_delta, None, False

        # Fast path: avoid parser scans until tool markup starts.
        if not tool_markup_possible and "<" not in content_delta:
            tool_accumulated_text += content_delta
            return content_delta, None, False

        if not tool_markup_possible:
            tool_markup_possible = True

        tool_previous = tool_accumulated_text
        tool_accumulated_text += content_delta
        tool_result = tool_parser.extract_tool_calls_streaming(
            tool_previous, tool_accumulated_text, content_delta
        )

        if tool_result is None:
            # Parser requests suppression while inside tool markup.
            return None, None, True

        if "tool_calls" in tool_result:
            coerced_tool_calls = [
                _coerce_tool_call_delta_arguments(tc, tools_for_schema_coercion)
                for tc in tool_result["tool_calls"]
            ]
            filtered_tool_calls = _apply_tool_call_spray_policy_to_deltas(
                coerced_tool_calls,
                source="stream-parser",
            )
            if filtered_tool_calls:
                tool_calls_detected = True
                return None, filtered_tool_calls, False
            return None, None, False

        return tool_result.get("content", ""), None, False

    # Stream content
    async for output in engine.stream_chat(messages=messages, **kwargs):
        delta_text = output.new_text
        last_output = output

        # Track token counts from output (updated each chunk)
        if hasattr(output, "prompt_tokens") and output.prompt_tokens:
            prompt_tokens = output.prompt_tokens
        if hasattr(output, "completion_tokens") and output.completion_tokens:
            completion_tokens = output.completion_tokens

        reasoning_delta = None
        content_delta = None
        tool_calls_delta = None

        # Reasoning-aware path
        if _reasoning_parser and delta_text:
            previous_text = accumulated_text
            accumulated_text += delta_text
            delta_msg = _reasoning_parser.extract_reasoning_streaming(
                previous_text, accumulated_text, delta_text
            )

            if delta_msg is not None:
                reasoning_delta = delta_msg.reasoning
                content_delta = delta_msg.content

                # Enforce API-layer thinking budget by shifting overflow into content.
                if max_thinking_tokens is not None and reasoning_delta:
                    remaining = max_thinking_tokens - reasoning_tokens_used
                    kept_reasoning, overflow_reasoning = _split_text_by_token_budget(
                        reasoning_delta,
                        remaining,
                        reasoning_tokenizer,
                    )
                    reasoning_delta = kept_reasoning or None
                    if reasoning_delta:
                        reasoning_tokens_used += _count_text_tokens(
                            reasoning_delta,
                            reasoning_tokenizer,
                        )
                    if overflow_reasoning:
                        content_delta = overflow_reasoning + (content_delta or "")

            # Skip parser-only token deltas unless this is the final chunk.
            if delta_msg is None and not output.finished:
                continue

        # Standard path (no reasoning parser)
        if not _reasoning_parser:
            content_delta = delta_text
            if content_delta:
                content_delta = SPECIAL_TOKENS_PATTERN.sub("", content_delta)

            # Add <think> prefix on first content chunk for thinking models
            if is_thinking_model and not think_prefix_sent and content_delta:
                content_delta = "<think>" + content_delta
                think_prefix_sent = True

        # Route visible content through tool parser
        if content_delta:
            parsed_content, parsed_tool_calls, suppress = _route_tool_stream_delta(
                content_delta
            )
            if suppress and not reasoning_delta and not output.finished:
                continue
            content_delta = parsed_content
            tool_calls_delta = parsed_tool_calls

        if not reasoning_delta and not content_delta and not tool_calls_delta and not output.finished:
            continue

        chunk = ChatCompletionChunk(
            id=response_id,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(
                        content=content_delta if content_delta else None,
                        reasoning=reasoning_delta if reasoning_delta else None,
                        tool_calls=tool_calls_delta,
                    ),
                    finish_reason=(
                        "tool_calls"
                        if (
                            output.finished
                            and (tool_calls_delta is not None or tool_calls_detected)
                        )
                        else (output.finish_reason if output.finished else None)
                    ),
                )
            ],
            usage=get_usage(output) if output.finished else None,
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Fallback: if tool parser accumulated text but never emitted tool_calls
    # (e.g., </tool_call> never arrived - incomplete tool call)
    if tool_parser and tool_accumulated_text and not tool_calls_detected and tool_markup_possible:
        result = tool_parser.extract_tool_calls(tool_accumulated_text)
        if result.tools_called:
            tool_calls_with_coercion = [
                _coerce_tool_call_delta_arguments(tc, tools_for_schema_coercion)
                for tc in result.tool_calls
            ]
            tool_call_deltas = _apply_tool_call_spray_policy_to_deltas(
                tool_calls_with_coercion,
                source="stream-fallback",
            )
            tool_chunk = ChatCompletionChunk(
                id=response_id,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(tool_calls=tool_call_deltas),
                        finish_reason="tool_calls",
                    )
                ],
            )
            yield f"data: {tool_chunk.model_dump_json()}\n\n"

    # Log throughput
    elapsed = time.perf_counter() - start_time
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Chat completion (stream): {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Send final chunk with usage if requested
    if include_usage:
        usage_chunk = ChatCompletionChunk(
            id=response_id,
            model=request.model,
            choices=[],  # Empty choices for usage-only chunk
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        yield f"data: {usage_chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


# =============================================================================
# MCP Initialization
# =============================================================================


async def init_mcp(config_path: str):
    """Initialize MCP manager from config file."""
    global _mcp_manager, _mcp_executor

    try:
        from vllm_mlx.mcp import MCPClientManager, ToolExecutor, load_mcp_config

        config = load_mcp_config(config_path)
        _mcp_manager = MCPClientManager(config)
        await _mcp_manager.start()

        _mcp_executor = ToolExecutor(_mcp_manager)

        logger.info(f"MCP initialized with {len(_mcp_manager.get_all_tools())} tools")

    except ImportError:
        logger.error("MCP SDK not installed. Install with: pip install mcp")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize MCP: {e}")
        raise


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run the server."""
    parser = argparse.ArgumentParser(
        description="vllm-mlx OpenAI-compatible server for LLM and MLLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with simple mode (maximum throughput)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Start with continuous batching (for multiple users)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

    # With MCP tools
    python -m vllm_mlx.server --model mlx-community/Qwen3-4B-4bit --mcp-config mcp.json
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Model to load (HuggingFace model name or local path)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force loading as MLLM (multimodal language model)",
    )
    parser.add_argument(
        "--continuous-batching",
        action="store_true",
        help="Enable continuous batching for multiple concurrent users",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Enable reproducibility profile: force simple runtime, "
            "greedy sampling (temperature=0, top_p=1), and serialize tracked "
            "inference routes."
        ),
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Default max tokens for generation",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (if not set, no auth required)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Default request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Rate limit requests per minute per client (0 = disabled)",
    )
    parser.add_argument(
        "--memory-warn-threshold",
        type=float,
        default=70.0,
        help="Warn threshold for memory utilization percent (default: 70.0)",
    )
    parser.add_argument(
        "--memory-limit-threshold",
        type=float,
        default=85.0,
        help="Limit threshold for memory utilization percent (default: 85.0)",
    )
    parser.add_argument(
        "--memory-action",
        type=str,
        default="warn",
        choices=["warn", "reduce-context", "reject-new"],
        help=(
            "Action when memory limit threshold is crossed: "
            "warn, reduce-context, or reject-new."
        ),
    )
    parser.add_argument(
        "--memory-monitor-interval",
        type=float,
        default=5.0,
        help="Memory monitor polling interval in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--batch-divergence-monitor",
        action="store_true",
        help="Enable periodic batch divergence probes (serial vs concurrent).",
    )
    parser.add_argument(
        "--batch-divergence-interval",
        type=float,
        default=300.0,
        help="Batch divergence probe interval in seconds (default: 300.0)",
    )
    parser.add_argument(
        "--batch-divergence-threshold",
        type=float,
        default=0.95,
        help="Minimum token agreement before divergence warning (0-1, default: 0.95)",
    )
    parser.add_argument(
        "--batch-divergence-action",
        type=str,
        default="warn",
        choices=["warn", "serialize"],
        help=(
            "Action when divergence exceeds threshold: "
            "warn or serialize tracked inference routes."
        ),
    )
    # Reasoning parser options - choices loaded dynamically from registry
    from .reasoning import list_parsers

    reasoning_choices = list_parsers()
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        choices=reasoning_choices,
        help=(
            "Enable reasoning content extraction with specified parser. "
            f"Options: {', '.join(reasoning_choices)}."
        ),
    )
    parser.add_argument(
        "--max-thinking-tokens",
        type=int,
        default=None,
        help=(
            "Maximum reasoning tokens to emit before routing overflow into content. "
            "Requires --reasoning-parser."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Pre-load an embedding model at startup (e.g. mlx-community/all-MiniLM-L6-v2-4bit)",
    )
    parser.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="Default temperature for generation when not specified in request",
    )
    parser.add_argument(
        "--default-top-p",
        type=float,
        default=None,
        help="Default top_p for generation when not specified in request",
    )

    args = parser.parse_args()

    # Set global configuration
    global _api_key, _default_timeout, _rate_limiter
    global _memory_warn_threshold_pct, _memory_limit_threshold_pct
    global _memory_action, _memory_monitor_interval_seconds
    global _batch_divergence_monitor_enabled, _batch_divergence_interval_seconds
    global _batch_divergence_threshold, _batch_divergence_action
    global _default_temperature, _default_top_p, _max_thinking_tokens
    global _deterministic_mode, _deterministic_serialize
    _api_key = args.api_key
    _default_timeout = args.timeout
    _max_thinking_tokens = args.max_thinking_tokens
    if args.memory_warn_threshold <= 0:
        raise ValueError("--memory-warn-threshold must be > 0")
    if args.memory_limit_threshold <= args.memory_warn_threshold:
        raise ValueError(
            "--memory-limit-threshold must be greater than --memory-warn-threshold"
        )
    if args.memory_monitor_interval <= 0:
        raise ValueError("--memory-monitor-interval must be > 0")
    if args.batch_divergence_interval <= 0:
        raise ValueError("--batch-divergence-interval must be > 0")
    if args.batch_divergence_threshold <= 0 or args.batch_divergence_threshold > 1:
        raise ValueError("--batch-divergence-threshold must be in (0, 1].")
    _memory_warn_threshold_pct = args.memory_warn_threshold
    _memory_limit_threshold_pct = args.memory_limit_threshold
    _memory_action = args.memory_action
    _memory_monitor_interval_seconds = args.memory_monitor_interval
    _batch_divergence_monitor_enabled = args.batch_divergence_monitor
    _batch_divergence_interval_seconds = args.batch_divergence_interval
    _batch_divergence_threshold = args.batch_divergence_threshold
    _batch_divergence_action = args.batch_divergence_action
    _batch_divergence_state.configure(
        enabled=_batch_divergence_monitor_enabled,
        threshold=_batch_divergence_threshold,
        action=_batch_divergence_action,
    )
    _default_temperature = args.default_temperature
    _default_top_p = args.default_top_p
    _deterministic_mode = bool(args.deterministic)
    _deterministic_serialize = bool(args.deterministic)
    if _deterministic_mode:
        _default_temperature = 0.0
        _default_top_p = 1.0

    # Configure rate limiter
    if args.rate_limit > 0:
        _rate_limiter = RateLimiter(requests_per_minute=args.rate_limit, enabled=True)
        logger.info(
            f"Rate limiting enabled: {args.rate_limit} requests/minute per client"
        )

    # Security summary at startup
    logger.info("=" * 60)
    logger.info("SECURITY CONFIGURATION")
    logger.info("=" * 60)
    if _api_key:
        logger.info("  Authentication: ENABLED (API key required)")
    else:
        logger.warning("  Authentication: DISABLED - Use --api-key to enable")
    if args.rate_limit > 0:
        logger.info(f"  Rate limiting: ENABLED ({args.rate_limit} req/min)")
    else:
        logger.warning("  Rate limiting: DISABLED - Use --rate-limit to enable")
    logger.info(f"  Request timeout: {args.timeout}s")
    logger.info(
        "  Memory guardrails: warn=%.1f%% limit=%.1f%% action=%s interval=%.1fs",
        args.memory_warn_threshold,
        args.memory_limit_threshold,
        args.memory_action,
        args.memory_monitor_interval,
    )
    logger.info(
        "  Batch divergence: enabled=%s threshold=%.2f action=%s interval=%.1fs",
        args.batch_divergence_monitor,
        args.batch_divergence_threshold,
        args.batch_divergence_action,
        args.batch_divergence_interval,
    )
    if _deterministic_mode:
        logger.info(
            "  Deterministic profile: ENABLED (runtime=simple, temperature=0.0, top_p=1.0, serialize=true)"
        )
    else:
        logger.info("  Deterministic profile: DISABLED")
    logger.info("=" * 60)

    # Set MCP config for lifespan
    if args.mcp_config:
        os.environ["VLLM_MLX_MCP_CONFIG"] = args.mcp_config

    # Initialize reasoning parser if specified
    if args.max_thinking_tokens is not None and not args.reasoning_parser:
        raise ValueError("--max-thinking-tokens requires --reasoning-parser")

    if args.reasoning_parser:
        global _reasoning_parser
        from .reasoning import get_parser

        parser_cls = get_parser(args.reasoning_parser)
        _reasoning_parser = parser_cls()
        logger.info(f"Reasoning parser enabled: {args.reasoning_parser}")

    # Pre-load embedding model if specified
    load_embedding_model(args.embedding_model, lock=True)

    # Load model before starting server
    load_model(
        args.model,
        use_batching=(args.continuous_batching and not _deterministic_mode),
        max_tokens=args.max_tokens,
        force_mllm=args.mllm,
    )

    # Start server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
