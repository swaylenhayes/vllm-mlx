#!/usr/bin/env python3
"""Stage 3 backend runtime qualification runner for distilled Qwen3.5 models."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_MODELS = [
    "Jackrong/MLX-Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-8bit",
    "Jackrong/MLX-Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-4bit",
    "Jackrong/MLX-Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-8bit",
    "Jackrong/MLX-Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-4bit",
    "Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-8bit",
    "Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-4bit",
    "Jackrong/MLX-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-4bit",
    "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit",
    "wbkou/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-8bit-MLX",
]


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-").lower()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_rss_mb(pid: int) -> float | None:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
        ).strip()
        if not out:
            return None
        # `ps rss` is in KB on macOS.
        return round(int(out) / 1024.0, 2)
    except Exception:
        return None


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 120.0,
) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    response = requests.request(method, url, json=payload, timeout=timeout_s)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError(f"Expected dict JSON response, got: {type(body)!r}")
    return body, round(latency_ms, 2)


def _wait_ready(
    base_url: str,
    timeout_s: float,
    proc: subprocess.Popen[Any],
) -> tuple[bool, float, str]:
    started = time.perf_counter()
    deadline = started + timeout_s
    last_error = "not ready yet"
    while time.perf_counter() < deadline:
        return_code = proc.poll()
        if return_code is not None:
            return (
                False,
                round(time.perf_counter() - started, 2),
                f"server process exited early with code {return_code}",
            )
        try:
            body, _ = _request_json("GET", f"{base_url}/health", timeout_s=2.5)
            status = str(body.get("status", "")).lower()
            if status in {"healthy", "ok"}:
                return True, round(time.perf_counter() - started, 2), "healthy"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        try:
            body, _ = _request_json("GET", f"{base_url}/v1/models", timeout_s=2.5)
            if body.get("object") == "list" and isinstance(body.get("data"), list):
                return True, round(time.perf_counter() - started, 2), "models endpoint ready"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(1.0)
    return False, round(time.perf_counter() - started, 2), last_error


def _extract_content(resp: dict[str, Any]) -> str:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return str(content) if content is not None else ""


def _run_stream_check(
    base_url: str,
    model: str,
    timeout_s: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly 8 to 14 words about local inference."}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 64,
        "stream": True,
    }
    started = time.perf_counter()
    first_chunk_time: float | None = None
    chunk_count = 0
    content_parts: list[str] = []
    with requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout_s,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk: dict[str, Any]
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices or not isinstance(choices[0], dict):
                continue
            delta = choices[0].get("delta", {})
            if not isinstance(delta, dict):
                continue
            token = delta.get("content")
            if token:
                chunk_count += 1
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                content_parts.append(str(token))
    ended = time.perf_counter()
    ttft_ms = None
    if first_chunk_time is not None:
        ttft_ms = round((first_chunk_time - started) * 1000.0, 2)
    return {
        "stream_ok": chunk_count > 0,
        "chunk_count": chunk_count,
        "ttft_ms": ttft_ms,
        "stream_total_ms": round((ended - started) * 1000.0, 2),
        "stream_text_preview": "".join(content_parts)[:200],
    }


def _decision(result: dict[str, Any]) -> str:
    checks = result["checks"]
    critical = [
        "health",
        "models_endpoint",
        "capabilities_endpoint",
        "chat_non_stream",
        "stream_chat",
        "json_mode",
    ]
    if not all(checks.get(key, False) for key in critical):
        return "hold"
    if not checks.get("determinism_exact", False):
        return "promote_with_warning"
    return "promote"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Stage 3 Runtime Qualification - Distilled Qwen3.5 Batch")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{report['meta']['generated_at']}`")
    lines.append(f"- Runtime env: `{report['meta']['runtime_env']}`")
    lines.append(f"- Models in scope: `{len(report['results'])}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- `pass_count`: **{report['summary']['pass_count']}**")
    lines.append(f"- `warn_count`: **{report['summary']['warn_count']}**")
    lines.append(f"- `hold_count`: **{report['summary']['hold_count']}**")
    lines.append("")
    lines.append(
        "| # | Model | Decision | Health | Chat | Determinism | JSON | Stream | Startup(s) | Non-stream(ms) | TTFT(ms) | Peak RSS(MB) |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---:|---:|---:|---:|"
    )
    for idx, item in enumerate(report["results"], start=1):
        checks = item["checks"]
        metrics = item["metrics"]
        lines.append(
            "| "
            f"{idx} | `{item['model_id']}` | `{item['decision']}` | "
            f"{'pass' if checks['health'] else 'fail'} | "
            f"{'pass' if checks['chat_non_stream'] else 'fail'} | "
            f"{'pass' if checks['determinism_exact'] else 'warn'} | "
            f"{'pass' if checks['json_mode'] else 'fail'} | "
            f"{'pass' if checks['stream_chat'] else 'fail'} | "
            f"{item.get('startup_s', 'n/a')} | "
            f"{metrics.get('non_stream_ms', 'n/a')} | "
            f"{metrics.get('stream_ttft_ms', 'n/a')} | "
            f"{metrics.get('peak_rss_mb', 'n/a')} |"
        )
    lines.append("")
    lines.append("## Claim Boundary")
    lines.append("")
    lines.append("- This Stage 3 pass provides backend runtime qualification evidence.")
    lines.append("- It does not establish global quality/performance superiority claims outside this tested setup.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- JSON: `{report['meta']['json_path']}`")
    lines.append("- Per-model server logs are in this run directory.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_model(
    *,
    model: str,
    python_bin: str,
    port: int,
    health_timeout_s: float,
    request_timeout_s: float,
    log_path: Path,
) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{port}"
    checks = {
        "health": False,
        "models_endpoint": False,
        "capabilities_endpoint": False,
        "chat_non_stream": False,
        "determinism_exact": False,
        "json_mode": False,
        "stream_chat": False,
    }
    metrics: dict[str, Any] = {}
    errors: list[str] = []
    startup_s: float | None = None
    started_at = _utc_now()
    peak_rss_mb: float | None = None
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_fp:
        cmd = [
            python_bin,
            "-m",
            "vllm_mlx.cli",
            "serve",
            model,
            "--localhost",
            "--port",
            str(port),
            "--runtime-mode",
            "simple",
            "--cache-strategy",
            "auto",
            "--default-temperature",
            "0.0",
            "--default-top-p",
            "1.0",
            "--timeout",
            "900",
        ]
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            ready, startup_s, startup_note = _wait_ready(
                base_url,
                timeout_s=health_timeout_s,
                proc=proc,
            )
            rss = _read_rss_mb(proc.pid)
            if rss is not None:
                peak_rss_mb = rss if peak_rss_mb is None else max(peak_rss_mb, rss)
            if not ready:
                errors.append(f"startup_timeout: {startup_note}")
                raise RuntimeError(startup_note)
            checks["health"] = True

            models_body, _ = _request_json("GET", f"{base_url}/v1/models", timeout_s=10)
            data = models_body.get("data")
            checks["models_endpoint"] = isinstance(data, list)

            caps_body, _ = _request_json("GET", f"{base_url}/v1/capabilities", timeout_s=10)
            checks["capabilities_endpoint"] = isinstance(caps_body, dict) and bool(caps_body)

            try:
                payload_chat = {
                    "model": "default",
                    "messages": [{"role": "user", "content": "Reply with one short sentence proving inference works."}],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 64,
                }
                chat_body, chat_ms = _request_json(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    payload=payload_chat,
                    timeout_s=request_timeout_s,
                )
                chat_text = _extract_content(chat_body)
                checks["chat_non_stream"] = len(chat_text.strip()) > 0
                metrics["non_stream_ms"] = chat_ms
                metrics["chat_preview"] = chat_text[:200]
            except Exception as exc:  # noqa: BLE001
                errors.append(f"chat_non_stream: {exc}")

            try:
                det_prompt = "Return exactly this token sequence and nothing else: ALPHA-BRAVO-CHARLIE."
                det_payload = {
                    "model": "default",
                    "messages": [{"role": "user", "content": det_prompt}],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 32,
                }
                det_a, _ = _request_json(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    payload=det_payload,
                    timeout_s=request_timeout_s,
                )
                det_b, _ = _request_json(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    payload=det_payload,
                    timeout_s=request_timeout_s,
                )
                det_text_a = _extract_content(det_a).strip()
                det_text_b = _extract_content(det_b).strip()
                checks["determinism_exact"] = bool(det_text_a) and det_text_a == det_text_b
                metrics["determinism_a"] = det_text_a[:120]
                metrics["determinism_b"] = det_text_b[:120]
            except Exception as exc:  # noqa: BLE001
                errors.append(f"determinism_exact: {exc}")

            try:
                json_payload = {
                    "model": "default",
                    "messages": [
                        {
                            "role": "user",
                            "content": 'Return ONLY JSON object: {"ok": true, "source": "stage3"}',
                        }
                    ],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 80,
                    "response_format": {"type": "json_object"},
                }
                json_body, _ = _request_json(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    payload=json_payload,
                    timeout_s=request_timeout_s,
                )
                json_text = _extract_content(json_body).strip()
                parsed_json = json.loads(json_text)
                checks["json_mode"] = isinstance(parsed_json, dict)
                metrics["json_preview"] = json_text[:160]
            except Exception as exc:  # noqa: BLE001
                errors.append(f"json_mode: {exc}")

            try:
                stream_info = _run_stream_check(base_url, "default", timeout_s=request_timeout_s)
                checks["stream_chat"] = bool(stream_info.get("stream_ok"))
                metrics["stream_ttft_ms"] = stream_info.get("ttft_ms")
                metrics["stream_total_ms"] = stream_info.get("stream_total_ms")
                metrics["stream_chunks"] = stream_info.get("chunk_count")
                metrics["stream_preview"] = stream_info.get("stream_text_preview", "")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stream_chat: {exc}")

            rss = _read_rss_mb(proc.pid)
            if rss is not None:
                peak_rss_mb = rss if peak_rss_mb is None else max(peak_rss_mb, rss)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    metrics["peak_rss_mb"] = peak_rss_mb
    finished_at = _utc_now()
    result: dict[str, Any] = {
        "model_id": model,
        "started_at": started_at,
        "finished_at": finished_at,
        "startup_s": startup_s,
        "checks": checks,
        "metrics": metrics,
        "errors": errors,
        "log_path": str(log_path),
    }
    result["decision"] = _decision(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=str(Path(".venv/bin/python").resolve()))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--health-timeout", type=float, default=420.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument(
        "--output-dir",
        default=str(
            Path("_docs/exports")
            / f"stage3-distilled-qwen35-runtime-qualification-{dt.datetime.now().strftime('%Y-%m-%d')}"
        ),
    )
    parser.add_argument("--models-file", default="")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    models = list(DEFAULT_MODELS)
    if args.models_file:
        lines = Path(args.models_file).read_text(encoding="utf-8").splitlines()
        models = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
        if not models:
            raise ValueError(f"No models found in models file: {args.models_file}")

    results: list[dict[str, Any]] = []
    for idx, model in enumerate(models, start=1):
        print(f"[{idx}/{len(models)}] Stage 3: {model}")
        log_path = out_dir / f"server-{_slug(model)}.log"
        result = _run_model(
            model=model,
            python_bin=args.python_bin,
            port=args.port,
            health_timeout_s=args.health_timeout,
            request_timeout_s=args.request_timeout,
            log_path=log_path,
        )
        results.append(result)
        print(
            f"  decision={result['decision']} "
            f"health={result['checks']['health']} "
            f"chat={result['checks']['chat_non_stream']} "
            f"det={result['checks']['determinism_exact']} "
            f"json={result['checks']['json_mode']} "
            f"stream={result['checks']['stream_chat']}"
        )

    summary = {
        "pass_count": sum(1 for r in results if r["decision"] == "promote"),
        "warn_count": sum(1 for r in results if r["decision"] == "promote_with_warning"),
        "hold_count": sum(1 for r in results if r["decision"] == "hold"),
    }
    report = {
        "meta": {
            "generated_at": _utc_now(),
            "runtime_env": f"{sys.platform} python={sys.version.split()[0]}",
            "scope_models": models,
            "method": (
                "Sequential backend runtime qualification: startup, health, models endpoint, "
                "capabilities endpoint, non-stream chat, deterministic repeat, JSON mode, stream chat."
            ),
        },
        "summary": summary,
        "results": results,
    }

    json_path = out_dir / "stage3_runtime_qualification_results.json"
    report["meta"]["json_path"] = str(json_path)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_path = out_dir / "stage3_runtime_qualification_summary.md"
    _write_markdown(md_path, report)

    print(f"\nWrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(
        "Summary:",
        f"pass={summary['pass_count']}",
        f"warn={summary['warn_count']}",
        f"hold={summary['hold_count']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
