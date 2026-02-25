# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for OpenAI-compatible API.

These models define the request and response schemas for:
- Chat completions
- Text completions
- Tool calling
- MCP (Model Context Protocol) integration
"""

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field, computed_field

# =============================================================================
# Content Types (for multimodal messages)
# =============================================================================


class ImageUrl(BaseModel):
    """Image URL with optional detail level."""

    url: str
    detail: str | None = None


class VideoUrl(BaseModel):
    """Video URL."""

    url: str


class AudioUrl(BaseModel):
    """Audio URL for audio content."""

    url: str


class ContentPart(BaseModel):
    """
    A part of a multimodal message content.

    Supports:
    - text: Plain text content
    - image_url: Image from URL or base64
    - video: Video from local path
    - video_url: Video from URL or base64
    - audio_url: Audio from URL or base64
    """

    type: str  # "text", "image_url", "video", "video_url", "audio_url"
    text: str | None = None
    image_url: ImageUrl | dict | str | None = None
    video: str | None = None
    video_url: VideoUrl | dict | str | None = None
    audio_url: AudioUrl | dict | str | None = None


# =============================================================================
# Messages
# =============================================================================


class Message(BaseModel):
    """
    A message in a chat conversation.

    Supports:
    - Simple text messages (role + content string)
    - Multimodal messages (role + content list with text/images/videos)
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool" with tool_call_id)
    """

    role: str
    content: str | list[ContentPart] | list[dict] | None = None
    # For assistant messages with tool calls
    tool_calls: list[dict] | None = None
    # For tool response messages (role="tool")
    tool_call_id: str | None = None


# =============================================================================
# Tool Calling
# =============================================================================


class FunctionCall(BaseModel):
    """A function call with name and arguments."""

    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A tool call from the model."""

    id: str
    type: str = "function"
    function: FunctionCall


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by the model."""

    type: str = "function"
    function: dict


# =============================================================================
# Structured Output (JSON Schema)
# =============================================================================


class ResponseFormatJsonSchema(BaseModel):
    """JSON Schema definition for structured output."""

    name: str
    description: str | None = None
    schema_: dict = Field(alias="schema")  # JSON Schema specification
    strict: bool | None = False

    class Config:
        populate_by_name = True


class ResponseFormat(BaseModel):
    """
    Response format specification for structured output.

    Supports:
    - "text": Default text output (no structure enforcement)
    - "json_object": Forces valid JSON output
    - "json_schema": Forces JSON matching a specific schema
    """

    type: str = "text"  # "text", "json_object", "json_schema"
    json_schema: ResponseFormatJsonSchema | None = None


# =============================================================================
# Chat Completion
# =============================================================================


class StreamOptions(BaseModel):
    """Options for streaming responses."""

    include_usage: bool = False  # Include usage stats in final chunk


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    model: str
    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    # OpenAI-style frequency penalty. Server maps this to repetition_penalty
    # for backends that support repetition logits processors.
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    # Native repetition penalty passthrough for backends that support it.
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    # Optional per-request cap for reasoning tokens when reasoning parser is active.
    # If omitted, server-level --max-thinking-tokens policy applies.
    max_thinking_tokens: int | None = Field(default=None, gt=0)
    max_tokens: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = (
        None  # Streaming options (include_usage, etc.)
    )
    stop: list[str] | None = None
    # Tool calling
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None  # "auto", "none", or specific tool
    # Structured output
    response_format: ResponseFormat | dict | None = None
    # MLLM-specific parameters
    video_fps: float | None = None
    video_max_frames: int | None = None
    # Request timeout in seconds (None = use server default)
    timeout: float | None = None


class AssistantMessage(BaseModel):
    """Response message from the assistant."""

    role: str = "assistant"
    content: str | None = None
    reasoning: str | None = (
        None  # Reasoning/thinking content (when --reasoning-parser is used)
    )
    tool_calls: list[ToolCall] | None = None

    @computed_field
    @property
    def reasoning_content(self) -> str | None:
        """Alias for reasoning field. Serialized for backwards compatibility with clients expecting reasoning_content."""
        return self.reasoning


class ChatCompletionChoice(BaseModel):
    """A single choice in chat completion response."""

    index: int = 0
    message: AssistantMessage
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Response for chat completion."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Text Completion
# =============================================================================


class CompletionRequest(BaseModel):
    """Request for text completion."""

    model: str
    prompt: str | list[str]
    temperature: float | None = None
    top_p: float | None = None
    # OpenAI-style frequency penalty mapped to repetition_penalty by server.
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    # Native repetition penalty passthrough.
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | None = None
    # Request timeout in seconds (None = use server default)
    timeout: float | None = None


class CompletionChoice(BaseModel):
    """A single choice in text completion response."""

    index: int = 0
    text: str
    finish_reason: str | None = "stop"


class CompletionResponse(BaseModel):
    """Response for text completion."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Models List
# =============================================================================


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm-mlx"


class ModelsResponse(BaseModel):
    """Response for listing models."""

    object: str = "list"
    data: list[ModelInfo]


# =============================================================================
# Capabilities
# =============================================================================


class CapabilityModalities(BaseModel):
    """Modality capabilities for the currently loaded runtime."""

    text: bool
    image: bool
    video: bool
    audio_input: bool
    audio_output: bool


class CapabilityFeatures(BaseModel):
    """Feature capability flags for API/runtime behavior."""

    streaming: bool
    tool_calling: bool
    auto_tool_choice: bool
    structured_output: bool
    reasoning: bool
    embeddings: bool
    anthropic_messages: bool
    mcp: bool


class CapabilityAuth(BaseModel):
    """Authentication behavior exposed to API clients."""

    api_key_required: bool
    accepted_headers: list[str] = Field(
        default_factory=lambda: ["authorization", "x-api-key"]
    )


class CapabilityRateLimit(BaseModel):
    """Rate limit behavior for the current server runtime."""

    enabled: bool
    requests_per_minute: int | None = None


class CapabilityLimits(BaseModel):
    """Default request limits used when clients do not override values."""

    default_max_tokens: int
    default_timeout_seconds: float


class CapabilitiesResponse(BaseModel):
    """Response for runtime API capability discovery."""

    object: str = "capabilities"
    model_loaded: bool
    model_name: str | None = None
    model_type: str | None = None
    modalities: CapabilityModalities
    features: CapabilityFeatures
    auth: CapabilityAuth
    rate_limit: CapabilityRateLimit
    limits: CapabilityLimits


# =============================================================================
# Diagnostics
# =============================================================================


class DiagnosticCheck(BaseModel):
    """Single diagnostic check result."""

    status: str  # pass | warning | fail | unknown
    detail: str
    metadata: dict[str, Any] | None = None


class DiagnosticMemory(BaseModel):
    """Memory diagnostic snapshot."""

    active_gb: float | None = None
    peak_gb: float | None = None
    system_gb: float | None = None
    utilization_pct: float | None = None
    trend: str = "unknown"  # growing | stable | declining | unknown
    pressure: str = "unknown"  # normal | elevated | critical | unknown


class DiagnosticsHealthResponse(BaseModel):
    """Response for diagnostic health endpoint."""

    status: str  # healthy | degraded | unhealthy
    model: str | None = None
    checks: dict[str, DiagnosticCheck]
    memory: DiagnosticMemory | None = None
    timestamp: str


# =============================================================================
# MCP (Model Context Protocol)
# =============================================================================


class MCPToolInfo(BaseModel):
    """Information about an MCP tool."""

    name: str
    description: str
    server: str
    parameters: dict = Field(default_factory=dict)


class MCPToolsResponse(BaseModel):
    """Response for listing MCP tools."""

    tools: list[MCPToolInfo]
    count: int


class MCPServerInfo(BaseModel):
    """Information about an MCP server."""

    name: str
    state: str
    transport: str
    tools_count: int
    error: str | None = None


class MCPServersResponse(BaseModel):
    """Response for listing MCP servers."""

    servers: list[MCPServerInfo]


class MCPExecuteRequest(BaseModel):
    """Request to execute an MCP tool."""

    tool_name: str
    arguments: dict = Field(default_factory=dict)


class MCPExecuteResponse(BaseModel):
    """Response from executing an MCP tool."""

    tool_name: str
    content: str | list | dict | None = None
    is_error: bool = False
    error_message: str | None = None


# =============================================================================
# Audio (STT/TTS)
# =============================================================================


class AudioTranscriptionRequest(BaseModel):
    """Request for audio transcription (STT)."""

    model: str = "whisper-large-v3"
    language: str | None = None
    response_format: str = "json"
    temperature: float = 0.0
    timestamp_granularities: list[str] | None = None


class AudioTranscriptionResponse(BaseModel):
    """Response from audio transcription."""

    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[dict] | None = None


class AudioSpeechRequest(BaseModel):
    """Request for text-to-speech."""

    model: str = "kokoro"
    input: str
    voice: str = "af_heart"
    speed: float = 1.0
    response_format: str = "wav"


class AudioSeparationRequest(BaseModel):
    """Request for audio source separation."""

    model: str = "htdemucs"
    stems: list[str] = Field(default_factory=lambda: ["vocals", "accompaniment"])


# =============================================================================
# Embeddings
# =============================================================================


class EmbeddingRequest(BaseModel):
    """Request for text embeddings (OpenAI compatible)."""

    input: str | list[str]
    model: str
    encoding_format: str | None = "float"  # "float" or "base64"


class EmbeddingData(BaseModel):
    """A single embedding result."""

    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingUsage(BaseModel):
    """Token usage for embedding requests."""

    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    """Response for embeddings endpoint (OpenAI compatible)."""

    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)


# =============================================================================
# Streaming (for SSE responses)
# =============================================================================


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None
    reasoning: str | None = (
        None  # Reasoning/thinking content (when --reasoning-parser is used)
    )
    tool_calls: list[dict] | None = None

    @computed_field
    @property
    def reasoning_content(self) -> str | None:
        """Alias for reasoning field. Serialized for backwards compatibility with clients expecting reasoning_content."""
        return self.reasoning


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """A streaming chunk for chat completion."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Usage | None = None  # Included when stream_options.include_usage=true
