"""Tests for runtime mode policy and observed-concurrency profile persistence."""

from __future__ import annotations

from pathlib import Path

from vllm_mlx.runtime_mode import (
    load_observed_peak_concurrency,
    save_observed_peak_concurrency,
    select_runtime_mode,
)


def test_select_runtime_mode_auto_defaults_to_simple_without_observation():
    use_batching, reason = select_runtime_mode(
        requested_mode="auto",
        continuous_batching_flag=False,
        observed_peak=None,
        threshold=2,
    )
    assert use_batching is False
    assert "defaulting to simple" in reason


def test_select_runtime_mode_auto_uses_observed_peak_threshold():
    use_batching, reason = select_runtime_mode(
        requested_mode="auto",
        continuous_batching_flag=False,
        observed_peak=4,
        threshold=2,
    )
    assert use_batching is True
    assert "selecting batched" in reason


def test_select_runtime_mode_explicit_simple_overrides_legacy_flag():
    use_batching, reason = select_runtime_mode(
        requested_mode="simple",
        continuous_batching_flag=True,
        observed_peak=10,
        threshold=2,
    )
    assert use_batching is False
    assert "--runtime-mode simple" in reason


def test_runtime_profile_persists_peak_concurrency(tmp_path: Path):
    profile_path = tmp_path / "runtime_profile.json"

    save_observed_peak_concurrency(3, profile_path)
    assert load_observed_peak_concurrency(profile_path) == 3

    # Historical observed peak should keep the max seen across runs.
    profile = save_observed_peak_concurrency(2, profile_path)
    assert profile["last_run_peak_concurrency"] == 2
    assert profile["observed_peak_concurrency"] == 3
