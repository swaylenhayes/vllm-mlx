#!/usr/bin/env python3
"""Measure streaming latency metrics against a running server."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
]


def _load_prompts(path: str | None) -> list[str]:
    if not path:
        return list(DEFAULT_PROMPTS)
    prompt_path = Path(path)
    lines = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in lines if line]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "stdev": 0.0,
        }
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "stdev": round(stdev, 4),
    }


def _is_ready_payload(payload: dict) -> bool:
    if payload.get("status") == "healthy":
        return True
    if payload.get("object") == "list" and isinstance(payload.get("data"), list):
        return True
    return False


async def _wait_for_health(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    probe_urls = [
        f"{base_url.rstrip('/')}/health",
        f"{base_url.rstrip('/')}/v1/models",
    ]
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.time() < deadline:
            for probe_url in probe_urls:
                try:
                    response = await client.get(probe_url)
                    if response.is_success:
                        payload = response.json()
                        if _is_ready_payload(payload):
                            return
                except (httpx.HTTPError, ValueError):
                    pass
            await asyncio.sleep(0.5)
    raise TimeoutError(f"Server health check did not pass within {timeout_s:.1f}s")


async def _measure_streaming_latency(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout: float,
) -> dict:
    start_time = time.perf_counter()
    first_token_time = None
    token_times: list[float] = []
    token_count = 0

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=request_timeout) as client:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                current_time = time.perf_counter()
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    token_count += 1
                    if first_token_time is None:
                        first_token_time = current_time
                    token_times.append(current_time)

    end_time = time.perf_counter()
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0.0
    total_time_ms = (end_time - start_time) * 1000
    inter_token_latencies_ms = [
        (token_times[i] - token_times[i - 1]) * 1000 for i in range(1, len(token_times))
    ]

    return {
        "prompt": prompt,
        "ttft_ms": round(ttft_ms, 4),
        "total_time_ms": round(total_time_ms, 4),
        "token_count": token_count,
        "inter_token_latencies_ms": [round(value, 4) for value in inter_token_latencies_ms],
    }


async def _run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-file")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--health-timeout", type=float, default=120.0)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    prompts = _load_prompts(args.prompts_file)
    await _wait_for_health(args.base_url, args.health_timeout)

    if prompts and args.warmup_requests > 0:
        warmup_prompt = prompts[0]
        for _ in range(args.warmup_requests):
            await _measure_streaming_latency(
                base_url=args.base_url,
                model=args.model,
                prompt=warmup_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                request_timeout=args.request_timeout,
            )

    runs: list[dict] = []
    for prompt in prompts:
        for _ in range(args.iterations):
            runs.append(
                await _measure_streaming_latency(
                    base_url=args.base_url,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    request_timeout=args.request_timeout,
                )
            )

    ttft_values = [run["ttft_ms"] for run in runs]
    total_values = [run["total_time_ms"] for run in runs]
    token_values = [run["token_count"] for run in runs]
    itl_values = [
        latency
        for run in runs
        for latency in run["inter_token_latencies_ms"]
    ]
    total_time_s = sum(total_values) / 1000.0
    total_tokens = sum(token_values)
    throughput = (total_tokens / total_time_s) if total_time_s > 0 else 0.0

    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "prompts_file": args.prompts_file,
        "iterations": args.iterations,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "prompt_count": len(prompts),
        "run_count": len(runs),
        "warmup_requests": args.warmup_requests,
        "ttft_ms": _stats(ttft_values),
        "itl_ms": _stats(itl_values),
        "total_time_ms": _stats(total_values),
        "tokens_per_second": round(throughput, 4),
        "total_tokens": total_tokens,
    }

    print("Streaming Latency Results:")
    print(f"  Runs: {summary['run_count']}")
    print(f"  TTFT mean: {summary['ttft_ms']['mean']:.2f} ms")
    print(f"  ITL mean: {summary['itl_ms']['mean']:.2f} ms")
    print(f"  Total time mean: {summary['total_time_ms']['mean']:.2f} ms")
    print(f"  Tokens/sec: {summary['tokens_per_second']:.2f}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "runs": runs}
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report: {out_path}")

    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
