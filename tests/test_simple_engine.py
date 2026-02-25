# SPDX-License-Identifier: Apache-2.0
"""Tests for SimpleEngine concurrency handling."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


class TestSimpleEngineConcurrency:
    """Test SimpleEngine lock behavior with concurrent requests."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that tracks concurrent calls."""
        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        # Track concurrent executions
        model._concurrent_count = 0
        model._max_concurrent = 0

        def generate_side_effect(**kwargs):
            model._concurrent_count += 1
            model._max_concurrent = max(model._max_concurrent, model._concurrent_count)
            # Simulate some work
            import time

            time.sleep(0.05)
            model._concurrent_count -= 1
            result = MagicMock()
            result.text = "test response"
            result.tokens = [1, 2, 3]
            result.finish_reason = "stop"
            return result

        model.generate = MagicMock(side_effect=generate_side_effect)
        return model

    @pytest.fixture
    def mock_llm_model(self):
        """Create a mock LLM model."""
        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        # Track concurrent executions
        model._concurrent_count = 0
        model._max_concurrent = 0

        def chat_side_effect(**kwargs):
            model._concurrent_count += 1
            model._max_concurrent = max(model._max_concurrent, model._concurrent_count)
            import time

            time.sleep(0.05)
            model._concurrent_count -= 1
            result = MagicMock()
            result.text = "test response"
            result.tokens = [1, 2, 3]
            result.finish_reason = "stop"
            return result

        model.chat = MagicMock(side_effect=chat_side_effect)
        return model

    @pytest.mark.asyncio
    async def test_lock_prevents_concurrent_generate(self, mock_model):
        """Test that the lock prevents concurrent generate calls."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True

            # Launch multiple concurrent generate calls
            tasks = [
                engine.generate(prompt=f"test prompt {i}", max_tokens=10)
                for i in range(5)
            ]

            await asyncio.gather(*tasks)

            # With the lock, max concurrent should be 1
            assert mock_model._max_concurrent == 1, (
                f"Expected max concurrent to be 1, but got {mock_model._max_concurrent}. "
                "The lock is not working correctly."
            )

    @pytest.mark.asyncio
    async def test_lock_prevents_concurrent_chat(self, mock_llm_model):
        """Test that the lock prevents concurrent chat calls."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_llm_model
            engine._loaded = True

            # Launch multiple concurrent chat calls
            tasks = [
                engine.chat(
                    messages=[{"role": "user", "content": f"test {i}"}], max_tokens=10
                )
                for i in range(5)
            ]

            await asyncio.gather(*tasks)

            # With the lock, max concurrent should be 1
            assert mock_llm_model._max_concurrent == 1, (
                f"Expected max concurrent to be 1, but got {mock_llm_model._max_concurrent}. "
                "The lock is not working correctly."
            )

    @pytest.mark.asyncio
    async def test_lock_serializes_stream_generate(self, mock_model):
        """Test that stream_generate uses the same lock as other methods."""
        from vllm_mlx.engine.simple import SimpleEngine

        def stream_generate_side_effect(**kwargs):
            # Yield a few chunks
            for i in range(3):
                chunk = MagicMock()
                chunk.text = f"chunk{i}"
                chunk.prompt_tokens = 5
                chunk.finished = i == 2
                chunk.finish_reason = "stop" if i == 2 else None
                yield chunk

        mock_model.stream_generate = MagicMock(side_effect=stream_generate_side_effect)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True

            # Test that stream_generate acquires the lock
            # by checking if it blocks when lock is already held
            lock_acquired = asyncio.Event()
            stream_started = asyncio.Event()

            async def hold_lock():
                async with engine._generation_lock:
                    lock_acquired.set()
                    # Wait until stream tries to start
                    await asyncio.sleep(0.1)

            async def try_stream():
                # Wait for lock to be held
                await lock_acquired.wait()
                stream_started.set()
                # This should block until hold_lock releases
                result = []
                async for chunk in engine.stream_generate(prompt="test", max_tokens=10):
                    result.append(chunk)
                return result

            # Start both tasks
            hold_task = asyncio.create_task(hold_lock())
            stream_task = asyncio.create_task(try_stream())

            # Wait a bit for stream to try to acquire lock
            await asyncio.sleep(0.05)

            # Stream should have started but be blocked on the lock
            assert stream_started.is_set(), "Stream should have attempted to start"

            # Stream task should not be done yet (blocked on lock)
            assert not stream_task.done(), "Stream should be blocked waiting for lock"

            # Let hold_lock finish
            await hold_task

            # Now stream should complete
            result = await stream_task
            assert len(result) == 3, f"Expected 3 chunks, got {len(result)}"

    @pytest.mark.asyncio
    async def test_engine_initialization_creates_lock(self):
        """Test that SimpleEngine creates a lock on initialization."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")

            assert hasattr(engine, "_generation_lock")
            assert isinstance(engine._generation_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_requests_complete_in_order(self, mock_model):
        """Test that concurrent requests complete (may be in any order due to lock)."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True

            # Launch multiple concurrent generate calls
            results = await asyncio.gather(
                *[
                    engine.generate(prompt=f"test prompt {i}", max_tokens=10)
                    for i in range(3)
                ]
            )

            # All requests should complete
            assert len(results) == 3
            for result in results:
                assert result.text == "test response"


class TestSimpleEngineToolChoicePassthrough:
    """Test tool/tool_choice propagation for LLM and MLLM paths."""

    @pytest.mark.asyncio
    async def test_mllm_chat_passes_tools_and_tool_choice(self):
        from vllm_mlx.engine.simple import SimpleEngine

        model = MagicMock()
        model.chat = MagicMock(
            return_value=MagicMock(
                text='<tool_call>{"name":"search_files","arguments":{"q":"x"}}</tool_call>',
                prompt_tokens=12,
                completion_tokens=4,
                finish_reason="stop",
            )
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search files",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                },
            }
        ]
        tool_choice = {"type": "function", "function": {"name": "search_files"}}

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=True):
            engine = SimpleEngine("test-mllm")
            engine._model = model
            engine._loaded = True

            await engine.chat(
                messages=[{"role": "user", "content": "Find X"}],
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=32,
            )

            _, kwargs = model.chat.call_args
            assert kwargs["tools"] == tools
            assert kwargs["tool_choice"] == tool_choice

    @pytest.mark.asyncio
    async def test_mllm_stream_chat_passes_tools_and_tool_choice(self):
        from vllm_mlx.engine.simple import SimpleEngine

        chunk1 = MagicMock()
        chunk1.text = "<tool_call>"
        chunk1.finish_reason = None
        chunk1.prompt_tokens = 8
        chunk2 = MagicMock()
        chunk2.text = "</tool_call>"
        chunk2.finish_reason = "stop"
        chunk2.prompt_tokens = 8

        model = MagicMock()
        model.stream_chat = MagicMock(return_value=iter([chunk1, chunk2]))

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search files",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        tool_choice = "required"

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=True):
            engine = SimpleEngine("test-mllm")
            engine._model = model
            engine._loaded = True

            outputs = []
            async for output in engine.stream_chat(
                messages=[{"role": "user", "content": "Find X"}],
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=16,
            ):
                outputs.append(output)

            assert outputs
            _, kwargs = model.stream_chat.call_args
            assert kwargs["tools"] == tools
            assert kwargs["tool_choice"] == tool_choice

    @pytest.mark.asyncio
    async def test_llm_chat_does_not_leak_tool_choice_to_model_call(self):
        from vllm_mlx.engine.simple import SimpleEngine

        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        model.chat = MagicMock(
            return_value=MagicMock(
                text="ok",
                tokens=[1, 2],
                finish_reason="stop",
            )
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search files",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-llm")
            engine._model = model
            engine._loaded = True

            await engine.chat(
                messages=[{"role": "user", "content": "Find X"}],
                tools=tools,
                tool_choice="required",
                max_tokens=16,
            )

            _, chat_kwargs = model.chat.call_args
            assert "tool_choice" not in chat_kwargs
