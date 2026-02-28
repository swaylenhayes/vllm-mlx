# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for text processing and model detection.
"""

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from huggingface_hub import snapshot_download

from .models import Message

logger = logging.getLogger(__name__)

# =============================================================================
# Special Token Patterns
# =============================================================================

# Pattern to match special tokens that should be removed from output
# Keeps <think>...</think> blocks intact for reasoning models
SPECIAL_TOKENS_PATTERN = re.compile(
    r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|"
    r"<\|end\|>|<\|eot_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|"
    r"<\|channel\|>|<\|message\|>|<\|start\|>|<\|return\|>|<\|call\|>|<\|constrain\|>|"
    r"</s>|<s>|<pad>|\[PAD\]|\[SEP\]|\[CLS\]"
)


# Regex for matching final channel marker with optional constrain token:
#   <|channel|>final<|message|>
#   <|channel|>final <|constrain|>JSON<|message|>
_FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final[^<]*(?:<\|constrain\|>[^<]*)?<\|message\|>"
)


def _clean_gpt_oss_output(text: str) -> str:
    """
    Extract final channel content from GPT-OSS channel-based output.

    When reasoning parser is not enabled, this provides a fallback that
    extracts the 'final' channel content so the API response is usable.

    Handles both standard and extended format with constrain token:
        <|channel|>final<|message|>...
        <|channel|>final <|constrain|>JSON<|message|>...

    Args:
        text: Raw model output containing channel tokens.

    Returns:
        Extracted final content, or text with channel tokens stripped.
    """
    match = _FINAL_CHANNEL_RE.search(text)
    if match:
        content = text[match.end() :]
        # Strip trailing structural tokens (including <|constrain|>)
        content = re.sub(
            r"<\|start\|>|<\|end\|>|<\|channel\|>|<\|return\|>|<\|call\|>|<\|message\|>|<\|constrain\|>",
            "",
            content,
        )
        return content.strip()

    # No final channel — strip all channel/structural tokens (including constrain)
    cleaned = re.sub(
        r"<\|channel\|>[^<]*(?:<\|constrain\|>[^<]*)?<\|message\|>|<\|start\|>[^<]*|<\|return\|>|<\|call\|>|<\|constrain\|>[^<]*",
        "",
        text,
    )
    return cleaned.strip()


def clean_output_text(text: str) -> str:
    """
    Clean model output by removing special tokens.

    Keeps <think>...</think> blocks intact for reasoning models.
    Adds opening <think> tag if missing (happens when thinking is enabled
    in the prompt template but the tag is part of the prompt, not output).
    Handles GPT-OSS channel-based format as fallback when reasoning parser
    is not enabled.

    Args:
        text: Raw model output

    Returns:
        Cleaned text with special tokens removed
    """
    if not text:
        return text

    # GPT-OSS channel format — extract final content before general stripping
    if "<|channel|>" in text and "<|message|>" in text:
        text = _clean_gpt_oss_output(text)
        return text

    text = SPECIAL_TOKENS_PATTERN.sub("", text)
    text = text.strip()

    # Add opening <think> tag if response has closing but not opening
    # This happens when enable_thinking=True in the chat template
    if "</think>" in text and not text.lstrip().startswith("<think>"):
        text = "<think>" + text

    return text


# =============================================================================
# Model Detection
# =============================================================================

# Patterns that indicate a multimodal language model (MLLM/VLM)
MLLM_PATTERNS = [
    "-VL-",
    "-VL/",
    "VL-",  # Qwen-VL, Qwen2-VL, Qwen3-VL, etc.
    "llava",
    "LLaVA",  # LLaVA models
    "idefics",
    "Idefics",  # Idefics models
    "paligemma",
    "PaliGemma",  # PaliGemma
    "gemma-3",
    "gemma3",  # Gemma 3 (multimodal)
    "medgemma",
    "MedGemma",  # MedGemma (medical multimodal with SigLIP vision encoder)
    "pixtral",
    "Pixtral",  # Pixtral
    "molmo",
    "Molmo",  # Molmo
    "phi3-vision",
    "phi-3-vision",  # Phi-3 Vision
    "cogvlm",
    "CogVLM",  # CogVLM
    "internvl",
    "InternVL",  # InternVL
    "deepseek-vl",
    "DeepSeek-VL",  # DeepSeek-VL
]

MLLM_METADATA_FILES = [
    "config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "model.safetensors.index.json",
]

MLLM_CONFIG_KEYS = {
    "vision_config",
    "image_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "video_token_id",
    "image_processor_type",
    "video_processor_type",
}

MLLM_PROCESSOR_HINTS = (
    "VL",
    "Llava",
    "Idefics",
    "PaliGemma",
    "Pixtral",
    "Molmo",
    "CogVLM",
    "InternVL",
    "MiniCPM",
    "Florence",
    "Vision",
)

MLLM_WEIGHT_PREFIXES = (
    "vision_tower.",
    "language_model.vision_tower.",
    "vision_model.",
    "visual.",
    "multi_modal_projector.",
    "image_newline.",
)


def _load_json_file(path: Path) -> dict | None:
    """Best-effort JSON loader for model metadata files."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _metadata_indicates_mllm(model_dir: Path) -> bool:
    """Check local model metadata for multimodal markers."""
    config = _load_json_file(model_dir / "config.json")
    if isinstance(config, dict):
        if any(key in config for key in MLLM_CONFIG_KEYS):
            return True
        architectures = config.get("architectures")
        if isinstance(architectures, list) and any(
            "ConditionalGeneration" in str(arch) for arch in architectures
        ):
            text_config = config.get("text_config")
            if isinstance(text_config, dict) and "vision_config" in config:
                return True

    for name in (
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        processor_config = _load_json_file(model_dir / name)
        if not isinstance(processor_config, dict):
            continue
        if any(key in processor_config for key in MLLM_CONFIG_KEYS):
            return True
        processor_class = processor_config.get("processor_class")
        if isinstance(processor_class, str) and any(
            hint in processor_class for hint in MLLM_PROCESSOR_HINTS
        ):
            return True

    weights_index = _load_json_file(model_dir / "model.safetensors.index.json")
    if isinstance(weights_index, dict):
        weight_map = weights_index.get("weight_map")
        if isinstance(weight_map, dict) and any(
            str(key).startswith(prefix)
            for key in weight_map
            for prefix in MLLM_WEIGHT_PREFIXES
        ):
            return True

    return False


@lru_cache(maxsize=128)
def _resolve_model_metadata_dir(model_name: str, offline: bool = False) -> Path | None:
    """Resolve a model directory containing enough metadata for MLLM detection."""
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return model_path if model_path.is_dir() else model_path.parent

    try:
        return Path(
            snapshot_download(
                model_name,
                allow_patterns=MLLM_METADATA_FILES,
                local_files_only=True,
            )
        )
    except Exception:
        if offline:
            return None

    try:
        return Path(
            snapshot_download(
                model_name,
                allow_patterns=MLLM_METADATA_FILES,
            )
        )
    except Exception as exc:
        logger.debug("MLLM metadata probe failed for %s: %s", model_name, exc)
        return None


def is_mllm_model(model_name: str, offline: bool = False) -> bool:
    """
    Check if model name indicates a multimodal language model.

    Args:
        model_name: HuggingFace model name or local path
        offline: When True, only use local paths or cached metadata

    Returns:
        True if model is detected as MLLM/VLM
    """
    model_lower = model_name.lower()
    for pattern in MLLM_PATTERNS:
        if pattern.lower() in model_lower:
            return True

    metadata_dir = _resolve_model_metadata_dir(model_name, offline=offline)
    if metadata_dir is not None:
        return _metadata_indicates_mllm(metadata_dir)

    return False


# Backwards compatibility alias
is_vlm_model = is_mllm_model


# =============================================================================
# Multimodal Content Extraction
# =============================================================================


def _content_to_text(content) -> str:
    """Extract text from content that can be str, list[ContentPart], or None."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif hasattr(item, "dict"):
                item = item.dict()
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return str(content)


def extract_multimodal_content(
    messages: list[Message],
    preserve_native_format: bool = False,
) -> tuple[list[dict], list[str], list[str]]:
    """
    Extract text content, images, and videos from OpenAI-format messages.

    Handles:
    - Simple text messages
    - Multimodal messages with images/videos
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool")

    Args:
        messages: List of Message objects
        preserve_native_format: If True, preserve native tool message format
            (role="tool", tool_calls field) instead of converting to text.
            Required for models with native tool support in chat templates
            (e.g., Mistral, Llama 3+, DeepSeek V3).

    Returns:
        Tuple of (processed_messages, images, videos)
        - processed_messages: List of {"role": str, "content": str}
        - images: List of image URLs/paths/base64
        - videos: List of video URLs/paths/base64
    """
    processed_messages = []
    images = []
    videos = []

    for msg in messages:
        # Handle both dict and Pydantic model messages
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content")
        else:
            role = msg.role
            content = msg.content

        # Handle tool response messages (role="tool")
        if role == "tool":
            if isinstance(msg, dict):
                tool_call_id = msg.get("tool_call_id", "") or ""
            else:
                tool_call_id = getattr(msg, "tool_call_id", None) or ""
            tool_content = content if content else ""

            if preserve_native_format:
                # Preserve native tool format for models that support it
                processed_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_content,
                    }
                )
            else:
                # Convert to user role for models without native support
                processed_messages.append(
                    {
                        "role": "user",
                        "content": f"[Tool Result ({tool_call_id})]: {tool_content}",
                    }
                )
            continue

        # Handle assistant messages with tool_calls
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
        else:
            tool_calls = getattr(msg, "tool_calls", None)

        if role == "assistant" and tool_calls:
            if preserve_native_format:
                # Preserve native tool_calls format
                tool_calls_list = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc_copy = tc
                    elif hasattr(tc, "model_dump"):
                        tc_copy = tc.model_dump()
                    elif hasattr(tc, "dict"):
                        tc_copy = tc.dict()
                    else:
                        continue

                    # Chat templates (e.g. Qwen3) iterate arguments|items,
                    # but OpenAI API sends arguments as a JSON string.
                    # Parse it into a dict so the template can iterate it.
                    func = tc_copy.get("function") or {}
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            import json

                            func["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            pass

                    tool_calls_list.append(tc_copy)

                msg_dict = {"role": role, "content": _content_to_text(content)}
                if tool_calls_list:
                    msg_dict["tool_calls"] = tool_calls_list
                processed_messages.append(msg_dict)
            else:
                # Convert tool calls to text for models without native support
                tool_calls_text = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        name = func.get("name", "unknown")
                        args = func.get("arguments", "{}")
                        tool_calls_text.append(f"[Calling tool: {name}({args})]")

                text = _content_to_text(content)
                if tool_calls_text:
                    text = (text + "\n" if text else "") + "\n".join(tool_calls_text)

                processed_messages.append({"role": role, "content": text})
            continue

        # Handle None content
        if content is None:
            processed_messages.append({"role": role, "content": ""})
            continue

        if isinstance(content, str):
            # Simple text message
            processed_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Multimodal message - extract text and media
            text_parts = []
            for item in content:
                # Handle both Pydantic models and dicts
                if hasattr(item, "model_dump"):
                    item = item.model_dump()
                elif hasattr(item, "dict"):
                    item = item.dict()

                item_type = item.get("type", "")

                if item_type == "text":
                    text_parts.append(item.get("text", ""))

                elif item_type == "image_url":
                    img_url = item.get("image_url", {})
                    if isinstance(img_url, str):
                        images.append(img_url)
                    elif isinstance(img_url, dict):
                        images.append(img_url.get("url", ""))

                elif item_type == "image":
                    images.append(item.get("image", item.get("url", "")))

                elif item_type == "video":
                    videos.append(item.get("video", item.get("url", "")))

                elif item_type == "video_url":
                    vid_url = item.get("video_url", {})
                    if isinstance(vid_url, str):
                        videos.append(vid_url)
                    elif isinstance(vid_url, dict):
                        videos.append(vid_url.get("url", ""))

            # Combine text parts
            combined_text = "\n".join(text_parts) if text_parts else ""
            processed_messages.append({"role": role, "content": combined_text})
        else:
            # Unknown format, try to convert
            processed_messages.append({"role": role, "content": str(content)})

    return processed_messages, images, videos
