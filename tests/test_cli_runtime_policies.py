"""Tests for CLI runtime/cache policy helpers."""

from __future__ import annotations

from types import SimpleNamespace

from vllm_mlx.cli import _build_startup_diagnostics, _resolve_cache_profile


def _make_args(**overrides):
    base = {
        "enable_prefix_cache": True,
        "disable_prefix_cache": False,
        "no_memory_aware_cache": False,
        "use_paged_cache": False,
        "cache_strategy": "auto",
        "max_num_seqs": 256,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_startup_diagnostics_warn_on_exposed_unauthenticated_server():
    diagnostics = _build_startup_diagnostics(
        bind_host="0.0.0.0",
        api_key=None,
        rate_limit=0,
        runtime_mode="simple",
    )
    assert any("non-localhost bind without API key auth" in msg for msg in diagnostics)
    assert any("rate limiting disabled" in msg for msg in diagnostics)


def test_startup_diagnostics_include_local_development_note():
    diagnostics = _build_startup_diagnostics(
        bind_host="127.0.0.1",
        api_key=None,
        rate_limit=0,
        runtime_mode="simple",
    )
    assert any("Localhost-only mode" in msg for msg in diagnostics)


def test_resolve_cache_profile_auto_uses_paged_for_high_concurrency_batching():
    profile = _resolve_cache_profile(_make_args(), use_batching=True)
    assert profile.enable_prefix_cache is True
    assert profile.use_paged_cache is True
    assert profile.use_memory_aware_cache is False
    assert profile.strategy_label.startswith("auto->paged")


def test_resolve_cache_profile_legacy_strategy():
    profile = _resolve_cache_profile(
        _make_args(cache_strategy="legacy"),
        use_batching=True,
    )
    assert profile.enable_prefix_cache is True
    assert profile.use_paged_cache is False
    assert profile.use_memory_aware_cache is False
    assert profile.strategy_label == "legacy"
