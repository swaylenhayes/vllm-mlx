# SPDX-License-Identifier: Apache-2.0
"""
LiquidAI/LFM tool call parser for vllm-mlx.

Handles formats like:
    <|tool_call_start|>[send_slack_message(channel='#general', message='done')]<|tool_call_end|>
"""

import ast
import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["liquidai", "liquid", "lfm"])
class LiquidAIToolParser(ToolParser):
    """
    Tool parser for LiquidAI/LFM models.

    Expected call format:
        <|tool_call_start|>[function_name(key='value', n=123)]<|tool_call_end|>
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    TOOL_CALL_START = "<|tool_call_start|>"
    TOOL_CALL_END = "<|tool_call_end|>"

    TOOL_CALL_PATTERN = re.compile(
        r"<\|tool_call_start\|>\s*\[\s*(?P<call>.*?)\s*\]\s*<\|tool_call_end\|>",
        re.DOTALL,
    )

    def _parse_call_expression(self, call_expr: str) -> tuple[str, str]:
        """
        Parse `func(a=1, b='x')` into (name, json_arguments).

        Falls back to {"_raw": "..."} when args cannot be safely parsed.
        """
        try:
            parsed = ast.parse(call_expr, mode="eval")
            node = parsed.body
            if not isinstance(node, ast.Call):
                raise ValueError("expression is not a function call")
            if not isinstance(node.func, ast.Name):
                raise ValueError("unsupported callable type")

            name = node.func.id
            args: dict[str, Any] = {}
            for kw in node.keywords:
                if kw.arg is None:
                    raise ValueError("positional unpacking is not supported")
                args[kw.arg] = ast.literal_eval(kw.value)
            return name, json.dumps(args, ensure_ascii=False)
        except Exception:
            fallback = {"_raw": call_expr.strip()}
            fallback_name = call_expr.split("(", 1)[0].strip() or "unknown_tool"
            return fallback_name, json.dumps(fallback, ensure_ascii=False)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract LiquidAI tool calls from full model output.
        """
        if self.TOOL_CALL_START not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[dict[str, Any]] = []
        cleaned_text = model_output

        for match in self.TOOL_CALL_PATTERN.finditer(model_output):
            call_expr = match.group("call").strip()
            name, args_json = self._parse_call_expression(call_expr)
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": args_json,
                }
            )

        if tool_calls:
            cleaned_text = self.TOOL_CALL_PATTERN.sub("", cleaned_text).strip()
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text or None,
            )

        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Extract LiquidAI tool calls from streaming output.

        Emits tool_calls only once a complete <|tool_call_start|>...<|tool_call_end|>
        block is available.
        """
        if self.TOOL_CALL_START not in current_text:
            return {"content": delta_text}

        # Inside tool block (suppress partial markup)
        start_count = current_text.count(self.TOOL_CALL_START)
        end_count = current_text.count(self.TOOL_CALL_END)
        if end_count < start_count and self.TOOL_CALL_END not in delta_text:
            prefix = ""
            if self.TOOL_CALL_START in delta_text:
                prefix = delta_text.split(self.TOOL_CALL_START, 1)[0]
            return {"content": prefix} if prefix else None

        if self.TOOL_CALL_END in delta_text:
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                return {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for i, tc in enumerate(result.tool_calls)
                    ]
                }

        return None
