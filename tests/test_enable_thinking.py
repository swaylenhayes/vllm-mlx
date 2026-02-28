# SPDX-License-Identifier: Apache-2.0
"""Targeted unit tests for request-level enable_thinking controls."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_chat_completion_request_accepts_enable_thinking():
    from vllm_mlx.api.models import ChatCompletionRequest, Message

    request = ChatCompletionRequest(
        model="test-model",
        messages=[Message(role="user", content="Hello")],
        enable_thinking=False,
    )

    assert request.enable_thinking is False


def test_mlx_language_model_chat_forwards_enable_thinking_to_template():
    from vllm_mlx.models.llm import GenerationOutput, MLXLanguageModel

    model = MLXLanguageModel("test-model")
    model._loaded = True
    model.tokenizer = MagicMock()
    model.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
    model.generate = MagicMock(
        return_value=GenerationOutput(text="ok", tokens=[1], finish_reason="stop")
    )

    output = model.chat(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=8,
        enable_thinking=False,
    )

    assert output.text == "ok"
    _, template_kwargs = model.tokenizer.apply_chat_template.call_args
    assert template_kwargs["enable_thinking"] is False


@pytest.mark.asyncio
async def test_simple_engine_prompt_builder_honors_enable_thinking_override():
    from vllm_mlx.engine.simple import SimpleEngine

    model = MagicMock()
    model.tokenizer = MagicMock()
    model.tokenizer.apply_chat_template = MagicMock(return_value="prompt")

    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
        engine = SimpleEngine("test-llm")
        engine._model = model
        engine._loaded = True

        prompt = engine._build_llm_prompt(
            [{"role": "user", "content": "ping"}],
            enable_thinking=False,
        )

        assert prompt == "prompt"
        _, template_kwargs = model.tokenizer.apply_chat_template.call_args
        assert template_kwargs["enable_thinking"] is False


def test_batched_engine_text_template_honors_enable_thinking_override():
    from vllm_mlx.engine.batched import BatchedEngine

    with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
        engine = BatchedEngine("test-llm")
        engine._tokenizer = MagicMock()
        engine._tokenizer.apply_chat_template = MagicMock(return_value="prompt")

        prompt = engine._apply_chat_template(
            [{"role": "user", "content": "ping"}],
            enable_thinking=False,
        )

        assert prompt == "prompt"
        _, template_kwargs = engine._tokenizer.apply_chat_template.call_args
        assert template_kwargs["enable_thinking"] is False


def test_batched_engine_mllm_template_honors_enable_thinking_override():
    from vllm_mlx.engine.batched import BatchedEngine

    with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=True):
        engine = BatchedEngine("test-mllm")
        engine._processor = MagicMock()
        engine._processor.apply_chat_template = MagicMock(return_value="prompt")

        prompt = engine._apply_chat_template(
            [{"role": "user", "content": "ping"}],
            enable_thinking=False,
        )

        assert prompt == "prompt"
        _, template_kwargs = engine._processor.apply_chat_template.call_args
        assert template_kwargs["enable_thinking"] is False


@pytest.mark.asyncio
async def test_simple_engine_mllm_chat_forwards_enable_thinking_to_model():
    from vllm_mlx.engine.simple import SimpleEngine

    model = MagicMock()
    model.chat = MagicMock(
        return_value=SimpleNamespace(
            text="ok",
            prompt_tokens=4,
            completion_tokens=1,
            finish_reason="stop",
        )
    )

    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=True):
        engine = SimpleEngine("test-mllm")
        engine._model = model
        engine._loaded = True

        output = await engine.chat(
            [{"role": "user", "content": "ping"}],
            max_tokens=8,
            enable_thinking=False,
        )

        assert output.text == "ok"
        _, chat_kwargs = model.chat.call_args
        assert chat_kwargs["enable_thinking"] is False


@pytest.mark.asyncio
async def test_simple_engine_mllm_stream_chat_forwards_enable_thinking_to_model():
    from vllm_mlx.engine.simple import SimpleEngine

    model = MagicMock()
    model.stream_chat = MagicMock(
        return_value=[
            SimpleNamespace(text="ok", prompt_tokens=4, finish_reason="stop"),
        ]
    )

    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=True):
        engine = SimpleEngine("test-mllm")
        engine._model = model
        engine._loaded = True

        chunks = []
        async for chunk in engine.stream_chat(
            [{"role": "user", "content": "ping"}],
            max_tokens=8,
            enable_thinking=False,
        ):
            chunks.append(chunk)

        assert chunks[-1].text == "ok"
        _, chat_kwargs = model.stream_chat.call_args
        assert chat_kwargs["enable_thinking"] is False


def test_mlx_multimodal_chat_forwards_enable_thinking_to_template(monkeypatch):
    from vllm_mlx.models.mllm import MLXMultimodalLM

    template_calls = []

    def fake_get_chat_template(processor, messages, add_generation_prompt, tokenize=False, **kwargs):
        del processor, tokenize
        template_calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            }
        )
        return "prompt"

    def fake_generate(
        model,
        processor,
        formatted_prompt,
        images,
        max_tokens,
        temp,
        top_p,
        verbose=False,
        prompt_cache=None,
        **kwargs,
    ):
        del (
            model,
            processor,
            formatted_prompt,
            images,
            max_tokens,
            temp,
            top_p,
            verbose,
            prompt_cache,
            kwargs,
        )
        return SimpleNamespace(text="ok", prompt_tokens=4, generation_tokens=1)

    mlx_vlm_module = ModuleType("mlx_vlm")
    mlx_vlm_module.generate = fake_generate
    prompt_utils_module = ModuleType("mlx_vlm.prompt_utils")
    prompt_utils_module.get_chat_template = fake_get_chat_template
    models_module = ModuleType("mlx_vlm.models")
    cache_module = ModuleType("mlx_vlm.models.cache")
    cache_module.make_prompt_cache = lambda language_model: None
    models_module.cache = cache_module

    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models", models_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.cache", cache_module)

    model = MLXMultimodalLM("test-mllm")
    model._loaded = True
    model.model = SimpleNamespace(language_model=object())
    model.processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text: [1, 2, 3]))
    model.config = {}

    output = model.chat(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=8,
        enable_thinking=False,
    )

    assert output.text == "ok"
    assert template_calls[0]["kwargs"]["enable_thinking"] is False


def test_mlx_multimodal_stream_chat_forwards_enable_thinking_to_template(monkeypatch):
    from vllm_mlx.models.mllm import MLXMultimodalLM

    template_calls = []

    def fake_get_chat_template(processor, messages, add_generation_prompt, tokenize=False, **kwargs):
        del processor, tokenize
        template_calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            }
        )
        return "prompt"

    def fake_stream_generate(
        model,
        processor,
        formatted_prompt,
        images,
        max_tokens,
        temp,
        top_p,
        **kwargs,
    ):
        del model, processor, formatted_prompt, images, max_tokens, temp, top_p, kwargs
        yield SimpleNamespace(text="ok", prompt_tokens=4, finish_reason="stop")

    mlx_vlm_module = ModuleType("mlx_vlm")
    mlx_vlm_module.stream_generate = fake_stream_generate
    prompt_utils_module = ModuleType("mlx_vlm.prompt_utils")
    prompt_utils_module.get_chat_template = fake_get_chat_template
    models_module = ModuleType("mlx_vlm.models")
    cache_module = ModuleType("mlx_vlm.models.cache")
    cache_module.make_prompt_cache = lambda language_model: None
    models_module.cache = cache_module

    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models", models_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.cache", cache_module)

    model = MLXMultimodalLM("test-mllm")
    model._loaded = True
    model.model = SimpleNamespace()
    model.processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text: [1, 2, 3]))

    chunks = list(
        model.stream_chat(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=8,
            enable_thinking=False,
        )
    )

    assert any(chunk.text == "ok" for chunk in chunks)
    assert chunks[-1].finish_reason == "stop"
    assert template_calls[0]["kwargs"]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_create_chat_completion_forwards_enable_thinking_to_engine(
    monkeypatch,
):
    import vllm_mlx.server as server

    engine = SimpleNamespace(
        is_mllm=False,
        preserve_native_tool_format=False,
        chat=AsyncMock(
            return_value=SimpleNamespace(
                text="ok",
                prompt_tokens=4,
                completion_tokens=1,
                finish_reason="stop",
            )
        ),
    )

    async def _await_result(result, raw_request, timeout):
        del raw_request, timeout
        return await result

    monkeypatch.setattr(server, "get_engine", lambda: engine)
    monkeypatch.setattr(server, "_wait_with_disconnect", _await_result)
    monkeypatch.setattr(server, "_record_repetition_intervention", lambda **kwargs: None)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_strict_model_id", False)

    request = server.ChatCompletionRequest(
        model="test-model",
        messages=[server.Message(role="user", content="Hello")],
        enable_thinking=False,
        max_tokens=8,
    )
    raw_request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))

    response = await server.create_chat_completion(request, raw_request)

    assert response.choices[0].message.content == "ok"
    _, chat_kwargs = engine.chat.call_args
    assert chat_kwargs["enable_thinking"] is False
