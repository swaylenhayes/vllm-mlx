# SPDX-License-Identifier: Apache-2.0
"""Runtime mode policy and observed-concurrency profile persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_RUNTIME_PROFILE_PATH = Path.home() / ".cache" / "vllm-mlx" / "runtime_profile.json"


def get_runtime_profile_path() -> Path:
    """Return the default profile path used for runtime mode observations."""
    return _RUNTIME_PROFILE_PATH


def _coerce_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def load_runtime_profile(path: Path | None = None) -> dict[str, Any]:
    """Load runtime profile JSON if it exists, otherwise return an empty dict."""
    profile_path = path or get_runtime_profile_path()
    if not profile_path.exists():
        return {}

    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def load_observed_peak_concurrency(path: Path | None = None) -> int | None:
    """Load historical observed peak concurrency from profile."""
    profile = load_runtime_profile(path)
    return _coerce_int(profile.get("observed_peak_concurrency"))


def save_observed_peak_concurrency(
    peak: int, path: Path | None = None
) -> dict[str, Any]:
    """Persist latest and historical peak concurrency values to profile."""
    profile_path = path or get_runtime_profile_path()
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    profile = load_runtime_profile(profile_path)
    existing_peak = _coerce_int(profile.get("observed_peak_concurrency")) or 0
    peak = max(0, int(peak))

    profile["last_run_peak_concurrency"] = peak
    profile["observed_peak_concurrency"] = max(existing_peak, peak)
    profile["updated_at_epoch"] = int(time.time())

    profile_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8"
    )
    return profile


def select_runtime_mode(
    requested_mode: str,
    continuous_batching_flag: bool,
    observed_peak: int | None,
    threshold: int,
) -> tuple[bool, str]:
    """
    Resolve runtime mode selection.

    Returns:
        Tuple of (use_batching, reason)
    """
    normalized_mode = requested_mode.strip().lower()
    if normalized_mode not in {"auto", "simple", "batched"}:
        raise ValueError(f"Unsupported runtime mode: {requested_mode!r}")

    threshold = max(1, int(threshold))

    if normalized_mode == "simple":
        return False, "forced simple mode via --runtime-mode simple"
    if normalized_mode == "batched":
        return True, "forced batched mode via --runtime-mode batched"

    if continuous_batching_flag:
        return True, "batched mode via --continuous-batching override"

    if observed_peak is None:
        return False, "auto mode: no historical concurrency data, defaulting to simple"

    if observed_peak >= threshold:
        return (
            True,
            "auto mode: observed peak concurrency "
            f"{observed_peak} >= threshold {threshold}, selecting batched",
        )

    return (
        False,
        "auto mode: observed peak concurrency "
        f"{observed_peak} < threshold {threshold}, selecting simple",
    )
