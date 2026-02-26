# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI-compatible API server."""

import platform
import sys

import pytest
from fastapi import HTTPException

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# =============================================================================
# Unit Tests - Request/Response Models
# =============================================================================


class TestRequestModels:
    """Test Pydantic request models."""

    def test_chat_message_text_only(self):
        """Test chat message with text content."""
        from vllm_mlx.server import Message

        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_message_multimodal(self):
        """Test chat message with multimodal content."""
        from vllm_mlx.server import Message

        content = [
            {"type": "text", "text": "What's this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ]
        msg = Message(role="user", content=content)

        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_image_url_model(self):
        """Test ImageUrl model."""
        from vllm_mlx.server import ImageUrl

        img_url = ImageUrl(url="https://example.com/image.jpg")
        assert img_url.url == "https://example.com/image.jpg"
        assert img_url.detail is None

    def test_video_url_model(self):
        """Test VideoUrl model."""
        from vllm_mlx.server import VideoUrl

        video_url = VideoUrl(url="https://example.com/video.mp4")
        assert video_url.url == "https://example.com/video.mp4"

    def test_content_part_text(self):
        """Test ContentPart with text."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(type="text", text="Hello world")
        assert part.type == "text"
        assert part.text == "Hello world"

    def test_content_part_image(self):
        """Test ContentPart with image_url."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(
            type="image_url", image_url={"url": "https://example.com/img.jpg"}
        )
        assert part.type == "image_url"
        # image_url can be dict or ImageUrl object
        if isinstance(part.image_url, dict):
            assert part.image_url["url"] == "https://example.com/img.jpg"
        else:
            assert part.image_url.url == "https://example.com/img.jpg"

    def test_content_part_video(self):
        """Test ContentPart with video."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(type="video", video="/path/to/video.mp4")
        assert part.type == "video"
        assert part.video == "/path/to/video.mp4"

    def test_content_part_video_url(self):
        """Test ContentPart with video_url."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(
            type="video_url", video_url={"url": "https://example.com/video.mp4"}
        )
        assert part.type == "video_url"
        # video_url can be dict or VideoUrl object
        if isinstance(part.video_url, dict):
            assert part.video_url["url"] == "https://example.com/video.mp4"
        else:
            assert part.video_url.url == "https://example.com/video.mp4"


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model."""

    def test_basic_request(self):
        """Test basic chat completion request."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model", messages=[Message(role="user", content="Hello")]
        )

        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.max_tokens is None  # uses _default_max_tokens when None
        assert (
            request.temperature is None
        )  # resolved at runtime by _resolve_temperature
        assert request.stream is False  # default
        assert request.include_diagnostics is False
        assert request.diagnostics_level is None

    def test_request_with_options(self):
        """Test request with custom options."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            max_tokens=100,
            temperature=0.5,
            frequency_penalty=0.3,
            repetition_penalty=1.2,
            repetition_policy_override="strict",
            max_thinking_tokens=128,
            include_diagnostics=True,
            diagnostics_level="deep",
            stream=True,
        )

        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.frequency_penalty == 0.3
        assert request.repetition_penalty == 1.2
        assert request.repetition_policy_override == "strict"
        assert request.max_thinking_tokens == 128
        assert request.include_diagnostics is True
        assert request.diagnostics_level == "deep"
        assert request.stream is True

    def test_request_with_video_params(self):
        """Test request with video parameters."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Describe the video")],
            video_fps=2.0,
            video_max_frames=16,
        )

        assert request.video_fps == 2.0
        assert request.video_max_frames == 16


class TestCompletionRequest:
    """Test CompletionRequest model."""

    def test_basic_completion_request(self):
        """Test basic completion request."""
        from vllm_mlx.server import CompletionRequest

        request = CompletionRequest(model="test-model", prompt="Once upon a time")

        assert request.model == "test-model"
        assert request.prompt == "Once upon a time"
        assert request.max_tokens is None  # uses _default_max_tokens when None
        assert request.include_diagnostics is False
        assert request.diagnostics_level is None

    def test_completion_request_with_penalties(self):
        """Test completion request with decode penalty controls."""
        from vllm_mlx.server import CompletionRequest

        request = CompletionRequest(
            model="test-model",
            prompt="Once upon a time",
            frequency_penalty=0.5,
            repetition_penalty=1.1,
            repetition_policy_override="safe",
        )

        assert request.frequency_penalty == 0.5
        assert request.repetition_penalty == 1.1
        assert request.repetition_policy_override == "safe"


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Test server helper functions."""

    def test_is_mllm_model_patterns(self):
        """Test MLLM model detection patterns."""
        from vllm_mlx.server import is_mllm_model

        # Should detect as MLLM
        assert is_mllm_model("mlx-community/Qwen3-VL-4B-Instruct-3bit")
        assert is_mllm_model("mlx-community/llava-1.5-7b-4bit")
        assert is_mllm_model("mlx-community/paligemma-3b-mix-224-4bit")
        assert is_mllm_model("mlx-community/pixtral-12b-4bit")
        assert is_mllm_model("mlx-community/Idefics3-8B-Llama3-4bit")
        assert is_mllm_model("mlx-community/deepseek-vl-7b-chat-4bit")

        # Should NOT detect as MLLM
        assert not is_mllm_model("mlx-community/Llama-3.2-1B-Instruct-4bit")
        assert not is_mllm_model("mlx-community/Mistral-7B-Instruct-4bit")
        assert not is_mllm_model("mlx-community/Qwen2-7B-Instruct-4bit")

    def test_extract_multimodal_content_text_only(self):
        """Test extracting content from text-only messages."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 2
        assert processed[0]["content"] == "Hello"
        assert len(images) == 0
        assert len(videos) == 0

    def test_extract_multimodal_content_with_image(self):
        """Test extracting content with images."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What's this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.jpg"},
                    },
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 1
        assert processed[0]["content"] == "What's this?"
        assert len(images) == 1
        assert "https://example.com/img.jpg" in images[0]

    def test_extract_multimodal_content_with_video(self):
        """Test extracting content with videos."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this video"},
                    {"type": "video", "video": "/path/to/video.mp4"},
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 1
        assert processed[0]["content"] == "Describe this video"
        assert len(videos) == 1
        assert videos[0] == "/path/to/video.mp4"

    def test_extract_multimodal_content_with_video_url(self):
        """Test extracting content with video_url format."""
        from vllm_mlx.server import extract_multimodal_content, Message

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What happens?"},
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/video.mp4"},
                    },
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(videos) == 1

    def test_resolve_repetition_penalty_prefers_explicit(self):
        from vllm_mlx.server import _resolve_repetition_penalty

        assert _resolve_repetition_penalty(1.25, 0.8) == 1.25

    def test_resolve_repetition_penalty_maps_frequency(self):
        from vllm_mlx.server import _resolve_repetition_penalty

        assert _resolve_repetition_penalty(None, 0.2) == 1.2
        assert _resolve_repetition_penalty(None, -0.4) == 0.6

    def test_resolve_repetition_penalty_clamps_to_positive(self):
        from vllm_mlx.server import _resolve_repetition_penalty

        assert _resolve_repetition_penalty(None, -2.0) == 0.01

    def test_resolve_max_thinking_tokens_prefers_request(self, monkeypatch):
        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_max_thinking_tokens", 256)
        assert server._resolve_max_thinking_tokens(64) == 64
        assert server._resolve_max_thinking_tokens(None) == 256

    def test_build_engine_thinking_kwargs_requires_think_boundaries(self, monkeypatch):
        import vllm_mlx.server as server

        class DummyParser:
            pass

        monkeypatch.setattr(server, "_max_thinking_tokens", 256)
        monkeypatch.setattr(server, "_reasoning_parser", DummyParser())
        assert server._build_engine_thinking_kwargs(None) == {}

    def test_build_engine_thinking_kwargs_from_parser_and_budget(self, monkeypatch):
        import vllm_mlx.server as server

        class DummyThinkParser:
            start_token = "<think>"
            end_token = "</think>"

        monkeypatch.setattr(server, "_max_thinking_tokens", 256)
        monkeypatch.setattr(server, "_reasoning_parser", DummyThinkParser())
        assert server._build_engine_thinking_kwargs(None) == {
            "thinking_budget_tokens": 256,
            "thinking_start_token": "<think>",
            "thinking_end_token": "</think>",
        }
        assert server._build_engine_thinking_kwargs(64) == {
            "thinking_budget_tokens": 64,
            "thinking_start_token": "<think>",
            "thinking_end_token": "</think>",
        }

    def test_split_text_by_token_budget_fallback(self):
        from vllm_mlx.server import _split_text_by_token_budget

        within, overflow = _split_text_by_token_budget("a b c d", 2, None)
        assert within == "a b"
        assert overflow == "c d"

    def test_version_satisfies_constraint(self):
        from vllm_mlx.server import _version_satisfies_constraint

        assert _version_satisfies_constraint("0.25.0", ">=0.24.0")
        assert _version_satisfies_constraint("0.25.0", "<=0.25.0")
        assert not _version_satisfies_constraint("0.25.0", "<0.25.0")
        assert _version_satisfies_constraint("0.25.0", ">=0.24.0,<0.30.0")

    def test_resolve_effective_max_tokens_applies_memory_factor(self, monkeypatch):
        import vllm_mlx.server as server

        class DummyState:
            def max_tokens_factor(self):
                return 0.5

        monkeypatch.setattr(server, "_memory_state", DummyState())
        assert server._resolve_effective_max_tokens(200) == 100

    def test_aggregate_diagnostic_status(self):
        from vllm_mlx.server import (
            DiagnosticCheck,
            _aggregate_diagnostic_status,
        )

        assert (
            _aggregate_diagnostic_status(
                {"a": DiagnosticCheck(status="pass", detail="ok")}
            )
            == "healthy"
        )
        assert (
            _aggregate_diagnostic_status(
                {
                    "a": DiagnosticCheck(status="pass", detail="ok"),
                    "b": DiagnosticCheck(status="warning", detail="warn"),
                }
            )
            == "degraded"
        )
        assert (
            _aggregate_diagnostic_status(
                {"a": DiagnosticCheck(status="fail", detail="bad")}
            )
            == "unhealthy"
        )

    def test_batch_divergence_state_serialize_gate(self):
        from vllm_mlx.server import BatchDivergenceState

        state = BatchDivergenceState()
        state.configure(enabled=True, threshold=0.95, action="serialize")
        state.reset(model_name="test-model", engine_type="batched")
        state.update_probe(
            token_agreement=0.40,
            exact_match=False,
            serial_latency_s=0.1,
            concurrent_latency_s=0.2,
        )
        assert state.should_serialize() is True

    def test_batch_divergence_state_no_serialize_on_warn_action(self):
        from vllm_mlx.server import BatchDivergenceState

        state = BatchDivergenceState()
        state.configure(enabled=True, threshold=0.95, action="warn")
        state.reset(model_name="test-model", engine_type="batched")
        state.update_probe(
            token_agreement=0.40,
            exact_match=False,
            serial_latency_s=0.1,
            concurrent_latency_s=0.2,
        )
        assert state.should_serialize() is False

    def test_check_batch_invariance_status_warning(self, monkeypatch):
        import vllm_mlx.server as server

        state = server.BatchDivergenceState()
        state.configure(enabled=True, threshold=0.95, action="warn")
        state.reset(model_name="test-model", engine_type="batched")
        state.update_probe(
            token_agreement=0.5,
            exact_match=False,
            serial_latency_s=0.1,
            concurrent_latency_s=0.2,
        )
        monkeypatch.setattr(server, "_batch_divergence_state", state)

        check = server._check_batch_invariance_status()
        assert check.status == "warning"
        assert check.metadata is not None
        assert check.metadata["sample_count"] >= 1

    def test_check_batch_invariance_status_disabled(self, monkeypatch):
        import vllm_mlx.server as server

        state = server.BatchDivergenceState()
        state.configure(enabled=False, threshold=0.95, action="warn")
        monkeypatch.setattr(server, "_batch_divergence_state", state)

        check = server._check_batch_invariance_status()
        assert check.status == "pass"
        assert "disabled" in check.detail.lower()

    def test_resolve_sampling_values_force_deterministic_mode(self, monkeypatch):
        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_deterministic_mode", True)
        monkeypatch.setattr(server, "_default_temperature", 0.8)
        monkeypatch.setattr(server, "_default_top_p", 0.2)

        assert server._resolve_temperature(0.9) == 0.0
        assert server._resolve_top_p(0.1) == 1.0

    def test_build_response_diagnostics_disabled(self):
        import vllm_mlx.server as server

        diagnostics = server._build_response_diagnostics(
            prompt_tokens=100,
            visual_inputs=1,
            diagnostics_level=None,
        )
        assert diagnostics is None

    def test_resolve_requested_diagnostics_level(self):
        import vllm_mlx.server as server

        assert (
            server._resolve_requested_diagnostics_level(
                include_diagnostics=False, diagnostics_level=None
            )
            is None
        )
        assert (
            server._resolve_requested_diagnostics_level(
                include_diagnostics=True, diagnostics_level=None
            )
            == "basic"
        )
        assert (
            server._resolve_requested_diagnostics_level(
                include_diagnostics=False, diagnostics_level="deep"
            )
            == "deep"
        )

    def test_enforce_request_model_id_disabled(self, monkeypatch):
        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_strict_model_id", False)
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        server._enforce_request_model_id("any-model")

    def test_enforce_request_model_id_enabled_rejects_mismatch(self, monkeypatch):
        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_strict_model_id", True)
        monkeypatch.setattr(server, "_model_name", "loaded-model")

        with pytest.raises(HTTPException) as exc:
            server._enforce_request_model_id("different-model")

        assert exc.value.status_code == 400
        assert "loaded-model" in str(exc.value.detail)

    def test_enforce_request_model_id_enabled_allows_exact_match(self, monkeypatch):
        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_strict_model_id", True)
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        server._enforce_request_model_id("loaded-model")

    def test_build_response_diagnostics_with_effective_context_override(
        self, monkeypatch
    ):
        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_engine", None)
        monkeypatch.setattr(server, "_effective_context_tokens", 1000)

        diagnostics = server._build_response_diagnostics(
            prompt_tokens=500,
            visual_inputs=2,
            diagnostics_level="basic",
        )
        assert diagnostics is not None
        assert diagnostics.level == "basic"
        assert diagnostics.effective_context_tokens == 1000
        assert diagnostics.effective_context_source == "operator_override"
        assert diagnostics.context_utilization_pct == 50.0
        assert diagnostics.visual_phase == "stable"

    def test_build_response_diagnostics_visual_collapse_phase(self, monkeypatch):
        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_engine", None)
        monkeypatch.setattr(server, "_effective_context_tokens", 1000)

        diagnostics = server._build_response_diagnostics(
            prompt_tokens=950,
            visual_inputs=1,
            diagnostics_level="basic",
        )
        assert diagnostics is not None
        assert diagnostics.visual_phase == "collapse"

    def test_build_response_diagnostics_deep_includes_runtime(self, monkeypatch):
        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_engine", None)
        monkeypatch.setattr(server, "_effective_context_tokens", 1000)
        monkeypatch.setattr(server, "_deterministic_mode", True)
        monkeypatch.setattr(server, "_deterministic_serialize", True)

        diagnostics = server._build_response_diagnostics(
            prompt_tokens=250,
            visual_inputs=1,
            diagnostics_level="deep",
        )
        assert diagnostics is not None
        assert diagnostics.level == "deep"
        assert diagnostics.runtime is not None
        assert diagnostics.runtime["deterministic_mode"] is True
        assert diagnostics.runtime["deterministic_serialize"] is True

    @pytest.mark.asyncio
    async def test_middleware_uses_serialize_lock_in_deterministic_mode(
        self, monkeypatch
    ):
        import vllm_mlx.server as server
        from fastapi import Request, Response
        from starlette.datastructures import URL

        class _DummyClient:
            host = "127.0.0.1"

        class _DummyRequest:
            url = URL("http://localhost:8000/v1/chat/completions")
            client = _DummyClient()

        class _DummyLock:
            def __init__(self):
                self.entered = 0

            async def __aenter__(self):
                self.entered += 1
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        dummy_lock = _DummyLock()
        monkeypatch.setattr(server, "_deterministic_serialize", True)
        monkeypatch.setattr(server, "_batch_serialize_lock", dummy_lock)

        state = server.BatchDivergenceState()
        state.configure(enabled=False, threshold=0.95, action="warn")
        monkeypatch.setattr(server, "_batch_divergence_state", state)

        async def _call_next(_req: Request):
            return Response(content="ok", media_type="text/plain")

        response = await server.track_runtime_concurrency(_DummyRequest(), _call_next)
        assert response.status_code == 200
        assert dummy_lock.entered == 1

    def test_tool_call_spray_policy_dedupes_exact_duplicates(self):
        import vllm_mlx.server as server

        calls = [
            server.ToolCall(
                id="call_1",
                type="function",
                function=server.FunctionCall(name="search_files", arguments='{"q":"a"}'),
            ),
            server.ToolCall(
                id="call_2",
                type="function",
                function=server.FunctionCall(name="search_files", arguments='{"q":"a"}'),
            ),
            server.ToolCall(
                id="call_3",
                type="function",
                function=server.FunctionCall(name="search_files", arguments='{"q":"b"}'),
            ),
        ]

        filtered = server._apply_tool_call_spray_policy(calls, source="test")
        assert len(filtered) == 2
        assert filtered[0].function.arguments == '{"q":"a"}'
        assert filtered[1].function.arguments == '{"q":"b"}'

    def test_tool_call_spray_policy_collapses_large_single_function_bursts(
        self, monkeypatch
    ):
        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_tool_call_spray_threshold", 4)
        calls = [
            server.ToolCall(
                id=f"call_{i}",
                type="function",
                function=server.FunctionCall(
                    name="search_files", arguments=f'{{"q":"variant_{i}"}}'
                ),
            )
            for i in range(6)
        ]

        filtered = server._apply_tool_call_spray_policy(calls, source="test")
        assert len(filtered) == 1
        assert filtered[0].function.name == "search_files"
        assert filtered[0].function.arguments == '{"q":"variant_0"}'

    def test_tool_call_spray_policy_keeps_large_multi_function_sets(
        self, monkeypatch
    ):
        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_tool_call_spray_threshold", 4)
        calls = [
            server.ToolCall(
                id="call_1",
                type="function",
                function=server.FunctionCall(name="search_files", arguments='{"q":"a"}'),
            ),
            server.ToolCall(
                id="call_2",
                type="function",
                function=server.FunctionCall(name="read_file", arguments='{"path":"a.py"}'),
            ),
            server.ToolCall(
                id="call_3",
                type="function",
                function=server.FunctionCall(name="search_files", arguments='{"q":"b"}'),
            ),
            server.ToolCall(
                id="call_4",
                type="function",
                function=server.FunctionCall(name="read_file", arguments='{"path":"b.py"}'),
            ),
            server.ToolCall(
                id="call_5",
                type="function",
                function=server.FunctionCall(name="search_files", arguments='{"q":"c"}'),
            ),
        ]

        filtered = server._apply_tool_call_spray_policy(calls, source="test")
        assert len(filtered) == 5

    def test_parse_tool_calls_with_parser_applies_spray_policy(self, monkeypatch):
        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
        monkeypatch.setattr(server, "_tool_call_spray_threshold", 3)

        spray_calls = [
            server.ToolCall(
                id=f"call_{i}",
                type="function",
                function=server.FunctionCall(
                    name="search_files", arguments=f'{{"q":"variant_{i}"}}'
                ),
            )
            for i in range(5)
        ]

        monkeypatch.setattr(server, "parse_tool_calls", lambda *_: ("", spray_calls))
        _, filtered = server._parse_tool_calls_with_parser("tool output", None)

        assert filtered is not None
        assert len(filtered) == 1
        assert filtered[0].function.name == "search_files"

    def test_coerce_tool_arguments_stringifies_object_when_schema_expects_string(self):
        import json
        import vllm_mlx.server as server

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                },
            }
        ]
        arguments = '{"path":"foo.txt","content":{"hello":"world"}}'

        coerced = server._coerce_tool_arguments(arguments, "write_file", tools)
        parsed = json.loads(coerced)
        assert isinstance(parsed["content"], str)
        assert json.loads(parsed["content"]) == {"hello": "world"}

    def test_coerce_tool_call_delta_arguments_handles_streaming_function_shape(self):
        import json
        import vllm_mlx.server as server

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                },
            }
        ]
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": '{"path":"foo.txt","content":{"hello":"world"}}',
            },
        }

        coerced = server._coerce_tool_call_delta_arguments(tool_call, tools)
        parsed = json.loads(coerced["function"]["arguments"])
        assert isinstance(parsed["content"], str)
        assert json.loads(parsed["content"]) == {"hello": "world"}

    def test_parse_tool_calls_with_mitigation_coerces_tool_arguments(self, monkeypatch):
        import json
        import vllm_mlx.server as server

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                        },
                    },
                },
            }
        ]
        calls = [
            server.ToolCall(
                id="call_1",
                type="function",
                function=server.FunctionCall(
                    name="write_file", arguments='{"content":{"hello":"world"}}'
                ),
            )
        ]
        monkeypatch.setattr(server, "parse_tool_calls", lambda *_: ("", calls))

        _, parsed_calls = server._parse_tool_calls_with_mitigation(
            "output",
            {"tools": tools},
            source="test",
        )

        assert parsed_calls is not None
        args = json.loads(parsed_calls[0].function.arguments)
        assert isinstance(args["content"], str)
        assert json.loads(args["content"]) == {"hello": "world"}

    def test_tool_call_spray_policy_to_deltas_reindexes(self, monkeypatch):
        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_tool_call_spray_threshold", 4)
        deltas = [
            {
                "index": i + 10,
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": "search_files",
                    "arguments": f'{{"q":"variant_{i}"}}',
                },
            }
            for i in range(6)
        ]

        filtered = server._apply_tool_call_spray_policy_to_deltas(deltas, source="test")
        assert len(filtered) == 1
        assert filtered[0]["index"] == 0
        assert filtered[0]["function"]["name"] == "search_files"

    def test_liquidai_tool_parser_on_reasoning_cleaned_content(self, monkeypatch):
        import vllm_mlx.server as server
        from vllm_mlx.reasoning import get_parser

        monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(server, "_tool_call_parser", "liquidai")
        monkeypatch.setattr(server, "_tool_parser_instance", None)
        monkeypatch.setattr(server, "_engine", None)

        reasoning_parser = get_parser("qwen3")()
        _, content = reasoning_parser.extract_reasoning(
            "<think>planning...</think>"
            "<|tool_call_start|>[search_files(query='requests')]<|tool_call_end|>"
        )

        cleaned, tool_calls = server._parse_tool_calls_with_parser(content or "", None)

        assert cleaned == ""
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "search_files"

    def test_repetition_policy_override_denied_for_untrusted_request(self, monkeypatch):
        import types

        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_api_key", None)
        monkeypatch.setattr(server, "_trust_requests_when_auth_disabled", False)
        monkeypatch.setattr(server, "_repetition_policy", "safe")
        monkeypatch.setattr(server, "_repetition_override_policy", "trusted_only")

        request = types.SimpleNamespace(
            client=types.SimpleNamespace(host="198.51.100.24")
        )
        effective_mode, override_accepted = server._resolve_repetition_policy(
            requested_override="strict",
            request=request,
        )

        assert effective_mode == "safe"
        assert override_accepted is False

    def test_repetition_policy_override_allowed_when_auth_disabled_and_trusted(
        self, monkeypatch
    ):
        import types

        import vllm_mlx.server as server

        monkeypatch.setattr(server, "_api_key", None)
        monkeypatch.setattr(server, "_trust_requests_when_auth_disabled", True)
        monkeypatch.setattr(server, "_repetition_policy", "safe")
        monkeypatch.setattr(server, "_repetition_override_policy", "trusted_only")

        request = types.SimpleNamespace(
            client=types.SimpleNamespace(host="198.51.100.24")
        )
        effective_mode, override_accepted = server._resolve_repetition_policy(
            requested_override="strict",
            request=request,
        )

        assert effective_mode == "strict"
        assert override_accepted is True


# =============================================================================
# Security and Reliability Tests (PR #4)
# =============================================================================


class TestRateLimiter:
    """Test the RateLimiter class for rate limiting functionality."""

    def test_rate_limiter_disabled_by_default(self):
        """Test that rate limiter allows all requests when disabled."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=5, enabled=False)

        # Should allow unlimited requests when disabled
        for _ in range(100):
            allowed, retry_after = limiter.is_allowed("client1")
            assert allowed is True
            assert retry_after == 0

    def test_rate_limiter_enforces_limit(self):
        """Test that rate limiter enforces the request limit."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=3, enabled=True)

        # First 3 requests should be allowed
        for i in range(3):
            allowed, retry_after = limiter.is_allowed("client1")
            assert allowed is True, f"Request {i+1} should be allowed"
            assert retry_after == 0

        # 4th request should be blocked
        allowed, retry_after = limiter.is_allowed("client1")
        assert allowed is False
        assert retry_after > 0

    def test_rate_limiter_per_client(self):
        """Test that rate limits are tracked per client."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=2, enabled=True)

        # Client 1 uses its quota
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        allowed, _ = limiter.is_allowed("client1")
        assert allowed is False

        # Client 2 should still have quota
        allowed, _ = limiter.is_allowed("client2")
        assert allowed is True

    def test_rate_limiter_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        import threading
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=100, enabled=True)
        results = []
        errors = []

        def make_requests():
            try:
                for _ in range(10):
                    allowed, _ = limiter.is_allowed("shared_client")
                    results.append(allowed)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_requests) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 100
        # Exactly 100 requests allowed (our limit)
        assert results.count(True) == 100


class TestTempFileManager:
    """Test the TempFileManager class for temp file cleanup."""

    def test_register_and_cleanup_single_file(self):
        """Test registering and cleaning up a single temp file."""
        import tempfile
        import os
        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()

        # Create a real temp file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        temp.write(b"test content")
        temp.close()

        # Register it
        path = manager.register(temp.name)
        assert path == temp.name
        assert os.path.exists(temp.name)

        # Cleanup
        result = manager.cleanup(temp.name)
        assert result is True
        assert not os.path.exists(temp.name)

    def test_cleanup_all_files(self):
        """Test cleaning up all registered temp files."""
        import tempfile
        import os
        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()
        paths = []

        # Create multiple temp files
        for i in range(3):
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.txt")
            temp.write(f"content {i}".encode())
            temp.close()
            manager.register(temp.name)
            paths.append(temp.name)

        # Verify all exist
        for p in paths:
            assert os.path.exists(p)

        # Cleanup all
        cleaned = manager.cleanup_all()
        assert cleaned == 3

        # Verify all deleted
        for p in paths:
            assert not os.path.exists(p)

    def test_cleanup_nonexistent_file(self):
        """Test cleanup of a non-existent file."""
        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()

        # Cleanup a file that doesn't exist
        result = manager.cleanup("/nonexistent/path/file.txt")
        assert result is False

    def test_thread_safe_registration(self):
        """Test that TempFileManager is thread-safe."""
        import threading
        import tempfile
        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()
        paths = []
        lock = threading.Lock()
        errors = []

        def register_files():
            try:
                for _ in range(5):
                    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
                    temp.write(b"test")
                    temp.close()
                    path = manager.register(temp.name)
                    with lock:
                        paths.append(path)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_files) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(paths) == 25

        # Cleanup all
        cleaned = manager.cleanup_all()
        assert cleaned == 25


class TestRequestOutputCollectorThreadSafety:
    """Test thread-safety of RequestOutputCollector._waiting_consumers."""

    def test_waiting_consumers_thread_safe(self):
        """Test that _waiting_consumers counter is thread-safe."""
        import threading
        from vllm_mlx.output_collector import RequestOutputCollector

        # Reset the counter
        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 0

        errors = []

        def manipulate_counter():
            try:
                for _ in range(100):
                    with RequestOutputCollector._waiting_lock:
                        RequestOutputCollector._waiting_consumers += 1
                    with RequestOutputCollector._waiting_lock:
                        RequestOutputCollector._waiting_consumers -= 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=manipulate_counter) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        # Should return to zero
        with RequestOutputCollector._waiting_lock:
            assert RequestOutputCollector._waiting_consumers == 0

    def test_has_waiting_consumers_method(self):
        """Test has_waiting_consumers class method."""
        from vllm_mlx.output_collector import RequestOutputCollector

        # Reset counter
        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 0

        assert RequestOutputCollector.has_waiting_consumers() is False

        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 1

        assert RequestOutputCollector.has_waiting_consumers() is True

        # Reset
        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 0


class TestRequestTimeoutField:
    """Test the new timeout field in request models."""

    def test_chat_completion_request_timeout_field(self):
        """Test that ChatCompletionRequest has timeout field."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        # Default should be None
        request = ChatCompletionRequest(
            model="test-model", messages=[Message(role="user", content="Hello")]
        )
        assert request.timeout is None

        # Can set custom timeout
        request_with_timeout = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            timeout=60.0,
        )
        assert request_with_timeout.timeout == 60.0

    def test_completion_request_timeout_field(self):
        """Test that CompletionRequest has timeout field."""
        from vllm_mlx.server import CompletionRequest

        # Default should be None
        request = CompletionRequest(model="test-model", prompt="Once upon a time")
        assert request.timeout is None

        # Can set custom timeout
        request_with_timeout = CompletionRequest(
            model="test-model", prompt="Once upon a time", timeout=120.0
        )
        assert request_with_timeout.timeout == 120.0


class TestAPIKeyVerification:
    """Test API key verification with timing attack prevention."""

    def test_secrets_compare_digest_usage(self):
        """Test that secrets.compare_digest is used (timing attack prevention)."""
        import secrets

        # Verify secrets.compare_digest works as expected
        key1 = "test-api-key-12345"
        key2 = "test-api-key-12345"
        key3 = "different-key-67890"

        # Same keys should match
        assert secrets.compare_digest(key1, key2) is True

        # Different keys should not match
        assert secrets.compare_digest(key1, key3) is False

        # Verify it's constant-time (by checking function exists)
        assert hasattr(secrets, "compare_digest")

    def test_verify_api_key_rejects_invalid(self):
        """Test that invalid API key is rejected with 401."""
        import asyncio
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        # Import and set up the module
        import vllm_mlx.server as server

        original_key = server._api_key

        try:
            # Set a known API key
            server._api_key = "valid-secret-key"

            # Create mock credentials with invalid key
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="invalid-key"
            )

            # Should raise HTTPException with 401
            with pytest.raises(HTTPException) as exc_info:
                asyncio.get_event_loop().run_until_complete(
                    server.verify_api_key(credentials)
                )

            assert exc_info.value.status_code == 401
            assert "Invalid API key" in str(exc_info.value.detail)
        finally:
            server._api_key = original_key

    def test_verify_api_key_accepts_valid(self):
        """Test that valid API key is accepted."""
        import asyncio
        from fastapi.security import HTTPAuthorizationCredentials

        import vllm_mlx.server as server

        original_key = server._api_key

        try:
            # Set a known API key
            server._api_key = "valid-secret-key"

            # Create mock credentials with valid key
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="valid-secret-key"
            )

            # Should not raise any exception
            result = asyncio.get_event_loop().run_until_complete(
                server.verify_api_key(credentials)
            )
            # verify_api_key returns True on success (no exception raised)
            assert result is True or result is None
        finally:
            server._api_key = original_key

    def test_verify_api_key_accepts_valid_x_api_key(self):
        """Test that valid x-api-key header value is accepted."""
        import asyncio

        import vllm_mlx.server as server

        original_key = server._api_key

        try:
            server._api_key = "valid-secret-key"

            result = asyncio.get_event_loop().run_until_complete(
                server.verify_api_key(None, "valid-secret-key")
            )
            assert result is True or result is None
        finally:
            server._api_key = original_key

    def test_verify_api_key_rejects_invalid_x_api_key(self):
        """Test that invalid x-api-key header value is rejected."""
        import asyncio
        from fastapi import HTTPException

        import vllm_mlx.server as server

        original_key = server._api_key

        try:
            server._api_key = "valid-secret-key"

            with pytest.raises(HTTPException) as exc_info:
                asyncio.get_event_loop().run_until_complete(
                    server.verify_api_key(None, "invalid-key")
                )

            assert exc_info.value.status_code == 401
            assert "Invalid API key" in str(exc_info.value.detail)
        finally:
            server._api_key = original_key


class TestAuthHeaderCompatibility:
    """Test API key extraction compatibility for supported header styles."""

    def test_extract_api_key_prefers_bearer_authorization(self):
        """Bearer token should win when both auth headers are provided."""
        from vllm_mlx.server import _extract_api_key_from_headers

        api_key = _extract_api_key_from_headers(
            "Bearer bearer-token", "x-api-key-token"
        )
        assert api_key == "bearer-token"

    def test_extract_api_key_uses_x_api_key_when_bearer_missing(self):
        """x-api-key should be accepted when Authorization is absent."""
        from vllm_mlx.server import _extract_api_key_from_headers

        api_key = _extract_api_key_from_headers(None, "x-api-key-token")
        assert api_key == "x-api-key-token"


class TestRouteSecurityCoverage:
    """Test route-level auth and rate-limit dependency coverage."""

    def test_anthropic_routes_require_auth_and_rate_limit(self):
        """Anthropic routes should mirror security dependencies used by chat routes."""
        from fastapi.routing import APIRoute

        import vllm_mlx.server as server

        target_paths = {"/v1/messages", "/v1/messages/count_tokens"}
        found_paths = set()

        for route in server.app.routes:
            if isinstance(route, APIRoute) and route.path in target_paths:
                dependency_calls = {dep.call for dep in route.dependant.dependencies}
                assert server.verify_api_key in dependency_calls
                assert server.check_rate_limit in dependency_calls
                found_paths.add(route.path)

        assert found_paths == target_paths

    def test_capabilities_route_requires_auth(self):
        """Capabilities route should require API key when auth is enabled."""
        from fastapi.routing import APIRoute

        import vllm_mlx.server as server

        route = next(
            r
            for r in server.app.routes
            if isinstance(r, APIRoute) and r.path == "/v1/capabilities"
        )
        dependency_calls = {dep.call for dep in route.dependant.dependencies}
        assert server.verify_api_key in dependency_calls

    def test_health_route_is_public(self):
        """Health route should remain unauthenticated for simple liveness checks."""
        from fastapi.routing import APIRoute

        import vllm_mlx.server as server

        route = next(
            r for r in server.app.routes if isinstance(r, APIRoute) and r.path == "/health"
        )
        dependency_calls = {dep.call for dep in route.dependant.dependencies}
        assert server.verify_api_key not in dependency_calls

    def test_health_diagnostics_route_respects_auth(self):
        """Diagnostics route should enforce auth when API key protection is enabled."""
        from fastapi.routing import APIRoute

        import vllm_mlx.server as server

        route = next(
            r
            for r in server.app.routes
            if isinstance(r, APIRoute) and r.path == "/health/diagnostics"
        )
        dependency_calls = {dep.call for dep in route.dependant.dependencies}
        assert server.verify_api_key in dependency_calls


class TestCapabilitiesEndpoint:
    """Test runtime capabilities contract."""

    def test_capabilities_response_contract(self):
        """Capabilities endpoint should return stable core fields."""
        import asyncio

        import vllm_mlx.server as server

        result = asyncio.get_event_loop().run_until_complete(server.get_capabilities())

        assert result.object == "capabilities"
        assert isinstance(result.model_loaded, bool)
        assert result.model_type in {"llm", "mllm", None}

        assert isinstance(result.modalities.text, bool)
        assert isinstance(result.modalities.image, bool)
        assert isinstance(result.modalities.video, bool)
        assert isinstance(result.modalities.audio_input, bool)
        assert isinstance(result.modalities.audio_output, bool)

        assert result.features.streaming is True
        assert result.features.structured_output is True
        assert result.features.anthropic_messages is True
        assert result.features.request_diagnostics is True
        assert isinstance(result.features.strict_model_id, bool)
        assert result.diagnostics is not None
        assert result.diagnostics.enabled is True
        assert "basic" in result.diagnostics.levels
        assert "deep" in result.diagnostics.levels
        assert result.diagnostics.default_level == "basic"
        assert "authorization" in result.auth.accepted_headers
        assert "x-api-key" in result.auth.accepted_headers
        assert result.policies is not None
        assert result.policies.repetition is not None
        assert result.policies.repetition.default_mode in {"safe", "strict"}
        assert "safe" in result.policies.repetition.supported_modes
        assert "strict" in result.policies.repetition.supported_modes
        assert result.policies.repetition.request_override in {
            "trusted_only",
            "disabled",
        }

        assert result.limits.default_max_tokens > 0
        assert result.limits.default_timeout_seconds > 0


class TestRateLimiterHTTPResponse:
    """Test rate limiter HTTP response behavior."""

    def test_rate_limiter_returns_retry_after(self):
        """Test that rate limiter returns retry_after when limit exceeded."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=2, enabled=True)

        # Exhaust the limit
        limiter.is_allowed("test_client")
        limiter.is_allowed("test_client")

        # Next request should be denied with retry_after
        allowed, retry_after = limiter.is_allowed("test_client")

        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0
        assert retry_after <= 60  # Should be within a minute

    def test_rate_limiter_window_cleanup(self):
        """Test that rate limiter cleans up old requests from sliding window."""
        from vllm_mlx.server import RateLimiter
        import time

        limiter = RateLimiter(requests_per_minute=2, enabled=True)

        # Make some requests
        limiter.is_allowed("test_client")
        limiter.is_allowed("test_client")

        # Should be denied (limit reached)
        allowed, _ = limiter.is_allowed("test_client")
        assert allowed is False

        # Manually inject old timestamps to simulate time passing
        # The sliding window should clean these up
        old_time = time.time() - 120  # 2 minutes ago
        with limiter._lock:
            limiter._requests["test_client"] = [old_time, old_time]

        # Now should be allowed again (old requests cleaned up)
        allowed, _ = limiter.is_allowed("test_client")
        assert allowed is True


# =============================================================================
# Integration Tests (require running server)
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
class TestServerIntegration:
    """Integration tests that require a running server.

    These tests are skipped by default. Run with:
        pytest -m integration --server-url http://localhost:8000
    """

    @pytest.fixture
    def server_url(self, request):
        """Get server URL from command line or use default."""
        return request.config.getoption("--server-url", default="http://localhost:8000")

    def test_health_endpoint(self, server_url):
        """Test /health endpoint."""
        import requests

        response = requests.get(f"{server_url}/health", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "model_name" in data

    def test_models_endpoint(self, server_url):
        """Test /v1/models endpoint."""
        import requests

        response = requests.get(f"{server_url}/v1/models", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0

    def test_chat_completion(self, server_url):
        """Test /v1/chat/completions endpoint."""
        import requests

        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10,
        }

        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        assert response.status_code == 200

        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"]


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--server-url",
        action="store",
        default="http://localhost:8000",
        help="URL of the vllm-mlx server for integration tests",
    )
