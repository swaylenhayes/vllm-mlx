# SPDX-License-Identifier: Apache-2.0
"""Typed client helpers for consuming /v1/capabilities safely."""

from __future__ import annotations

from typing import Any

import requests
from pydantic import ValidationError

from .api.models import CapabilitiesResponse


class CapabilityContractError(RuntimeError):
    """Raised when /v1/capabilities response cannot satisfy contract checks."""


def _build_capabilities_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/capabilities"
    return f"{base}/v1/capabilities"


def parse_capabilities_payload(payload: dict[str, Any]) -> CapabilitiesResponse:
    """Validate capabilities payload against the typed response model."""
    try:
        return CapabilitiesResponse.model_validate(payload)
    except ValidationError as exc:
        raise CapabilityContractError(
            f"Capabilities payload failed contract validation: {exc}"
        ) from exc


def fetch_capabilities(
    base_url: str,
    api_key: str | None = None,
    timeout: float = 5.0,
) -> CapabilitiesResponse:
    """Fetch and validate runtime capabilities from a running server."""
    url = _build_capabilities_url(base_url)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise CapabilityContractError(
            f"Failed to fetch capabilities from {url}: {exc}"
        ) from exc

    if response.status_code != 200:
        raise CapabilityContractError(
            f"Capabilities endpoint returned HTTP {response.status_code}: {response.text}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise CapabilityContractError(
            "Capabilities response is not valid JSON"
        ) from exc

    if not isinstance(payload, dict):
        raise CapabilityContractError("Capabilities response must be a JSON object")

    return parse_capabilities_payload(payload)


def summarize_capabilities(capabilities: CapabilitiesResponse) -> dict[str, Any]:
    """Return a stable summary shape for runtime tooling decisions."""
    return {
        "model_loaded": capabilities.model_loaded,
        "model_type": capabilities.model_type,
        "supports_multimodal": capabilities.modalities.image
        or capabilities.modalities.video,
        "supports_tool_calling": capabilities.features.tool_calling,
        "supports_reasoning": capabilities.features.reasoning,
        "auth_required": capabilities.auth.api_key_required,
        "rate_limit_enabled": capabilities.rate_limit.enabled,
        "default_max_tokens": capabilities.limits.default_max_tokens,
        "default_timeout_seconds": capabilities.limits.default_timeout_seconds,
    }
