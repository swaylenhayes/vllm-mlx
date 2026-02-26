"""Tests for capabilities client contract checks."""

from __future__ import annotations

import pytest

from vllm_mlx.capabilities_client import (
    CapabilityContractError,
    _build_capabilities_url,
    fetch_capabilities,
    parse_capabilities_payload,
    summarize_capabilities,
)


def _sample_payload():
    return {
        "object": "capabilities",
        "model_loaded": True,
        "model_name": "test-model",
        "model_type": "mllm",
        "modalities": {
            "text": True,
            "image": True,
            "video": True,
            "audio_input": False,
            "audio_output": False,
        },
        "features": {
            "streaming": True,
            "tool_calling": True,
            "auto_tool_choice": False,
            "structured_output": True,
            "reasoning": False,
            "embeddings": True,
            "anthropic_messages": True,
            "mcp": False,
            "request_diagnostics": True,
            "strict_model_id": True,
        },
        "diagnostics": {
            "enabled": True,
            "levels": ["basic", "deep"],
            "default_level": "basic",
            "deep_supported": True,
        },
        "policies": {
            "repetition": {
                "default_mode": "safe",
                "supported_modes": ["safe", "strict"],
                "request_override": "trusted_only",
            }
        },
        "auth": {"api_key_required": False, "accepted_headers": ["authorization"]},
        "rate_limit": {"enabled": False, "requests_per_minute": None},
        "limits": {
            "default_max_tokens": 32768,
            "default_timeout_seconds": 300.0,
            "effective_context_tokens": 8192,
        },
    }


def test_build_capabilities_url_normalizes_v1_suffix():
    assert _build_capabilities_url("http://localhost:8000") == (
        "http://localhost:8000/v1/capabilities"
    )
    assert _build_capabilities_url("http://localhost:8000/v1") == (
        "http://localhost:8000/v1/capabilities"
    )


def test_parse_capabilities_payload_validates_contract():
    parsed = parse_capabilities_payload(_sample_payload())
    assert parsed.object == "capabilities"
    assert parsed.modalities.image is True


def test_parse_capabilities_payload_rejects_invalid_contract():
    payload = _sample_payload()
    del payload["limits"]
    with pytest.raises(CapabilityContractError):
        parse_capabilities_payload(payload)


def test_summarize_capabilities_returns_stable_shape():
    parsed = parse_capabilities_payload(_sample_payload())
    summary = summarize_capabilities(parsed)
    assert summary["model_loaded"] is True
    assert summary["supports_multimodal"] is True
    assert summary["supports_tool_calling"] is True
    assert summary["supports_request_diagnostics"] is True
    assert summary["strict_model_id_enforced"] is True
    assert summary["default_diagnostics_level"] == "basic"
    assert "deep" in summary["diagnostics_levels"]
    assert summary["repetition_policy_default_mode"] == "safe"
    assert "strict" in summary["repetition_policy_supported_modes"]
    assert summary["repetition_policy_request_override"] == "trusted_only"


def test_fetch_capabilities_validates_http_and_payload(monkeypatch):
    class _Response:
        status_code = 200
        text = "ok"

        @staticmethod
        def json():
            return _sample_payload()

    calls = {}

    def _fake_get(url, headers, timeout):
        calls["url"] = url
        calls["headers"] = headers
        calls["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("vllm_mlx.capabilities_client.requests.get", _fake_get)

    capabilities = fetch_capabilities(
        "http://localhost:8000", api_key="secret", timeout=9
    )
    assert capabilities.model_name == "test-model"
    assert calls["url"].endswith("/v1/capabilities")
    assert calls["headers"]["Authorization"] == "Bearer secret"
    assert calls["timeout"] == 9
