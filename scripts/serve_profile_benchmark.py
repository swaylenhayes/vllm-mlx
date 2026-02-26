#!/usr/bin/env python3
"""Benchmark /v1/chat/completions throughput against a running server."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

DEFAULT_PROMPTS = [
    "Write a short poem about nature.",
    "Write a short poem about love.",
    "Write a short poem about technology.",
    "Write a short poem about space.",
    "Write a short poem about music.",
    "Write a short poem about art.",
    "Write a short poem about science.",
    "Write a short poem about history.",
    "Write a short poem about food.",
    "Write a short poem about travel.",
]


def _call_chat(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    usage = payload.get("usage") or {}
    return {
        "prompt": prompt,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _wait_for_health(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    health_url = f"{base_url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            response = requests.get(health_url, timeout=2.0)
            if response.ok:
                payload = response.json()
                if payload.get("status") == "healthy":
                    return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Server health check did not pass within {timeout_s:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--health-timeout", type=float, default=120.0)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    prompts = list(DEFAULT_PROMPTS)
    _wait_for_health(args.base_url, args.health_timeout)

    started = time.perf_counter()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(
                _call_chat,
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                timeout=args.request_timeout,
            )
            for prompt in prompts
        ]
        for future in as_completed(futures):
            results.append(future.result())

    total_time = time.perf_counter() - started
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    total_tokens = sum(r["total_tokens"] for r in results)

    summary = {
        "total_time_s": round(total_time, 4),
        "prompts": len(prompts),
        "prompts_per_second": round(len(prompts) / total_time, 4),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": round(total_completion_tokens / total_time, 4),
        "throughput_tok_per_s": round(total_tokens / total_time, 4),
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "model": args.model,
        "base_url": args.base_url,
    }

    print("Results:")
    print(f"  Total time: {summary['total_time_s']:.2f}s")
    print(f"  Prompts: {summary['prompts']}")
    print(f"  Prompts/second: {summary['prompts_per_second']:.2f}")
    print(f"  Total prompt tokens: {summary['total_prompt_tokens']}")
    print(f"  Total completion tokens: {summary['total_completion_tokens']}")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"  Tokens/second: {summary['tokens_per_second']:.2f}")
    print(f"  Throughput: {summary['throughput_tok_per_s']:.2f} tok/s")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
