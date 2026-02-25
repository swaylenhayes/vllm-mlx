#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Batch invariance harness for vllm-mlx.

Compares deterministic outputs for the same prompt set under:
1) isolated requests (one-by-one)
2) concurrent requests (batch-composition pressure proxy)

Usage:
  python scripts/batch_invariance_harness.py \
    --base-url http://localhost:8000 \
    --model mlx-community/Qwen3-4B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

DEFAULT_PROMPTS = [
    "Write one sentence explaining why caching helps inference throughput.",
    "List three steps to verify an API endpoint is healthy.",
    "What is the capital of France? Answer in one word.",
    "Return exactly: ping",
    "Name two risks of running without authentication.",
    "Summarize why deterministic decoding matters in testing.",
    "What does a 503 status code usually indicate?",
    "Give a one-line definition of batch invariance.",
    "Provide two bullet points on memory pressure guardrails.",
    "Output the word READY and nothing else.",
]


@dataclass
class ProbeResult:
    prompt: str
    output: str
    latency_s: float
    finish_reason: str | None


def _load_prompts(path: str | None) -> list[str]:
    if not path:
        return list(DEFAULT_PROMPTS)
    prompt_path = Path(path)
    lines = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in lines if line]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _run_probe(
    base_url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    timeout_s: float,
    api_key: str | None,
) -> ProbeResult:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    t0 = time.perf_counter()
    response = requests.post(
        url,
        headers=_headers(api_key),
        json=payload,
        timeout=timeout_s,
    )
    latency_s = time.perf_counter() - t0
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    message = choice.get("message", {})
    return ProbeResult(
        prompt=prompt,
        output=message.get("content") or "",
        latency_s=latency_s,
        finish_reason=choice.get("finish_reason"),
    )


def _tokenize(text: str) -> list[str]:
    return text.split()


def _token_agreement(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    max_len = max(len(ta), len(tb), 1)
    matches = 0
    for i in range(max_len):
        tok_a = ta[i] if i < len(ta) else None
        tok_b = tb[i] if i < len(tb) else None
        if tok_a == tok_b:
            matches += 1
    return matches / max_len


def _run_serial(
    prompts: list[str],
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout_s: float,
    api_key: str | None,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for prompt in prompts:
        results.append(
            _run_probe(
                base_url,
                model,
                prompt,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                api_key=api_key,
            )
        )
    return results


def _run_concurrent(
    prompts: list[str],
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout_s: float,
    api_key: str | None,
    concurrency: int,
) -> list[ProbeResult]:
    results: list[ProbeResult] = [None] * len(prompts)  # type: ignore[assignment]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures: dict[concurrent.futures.Future[ProbeResult], int] = {}
        for idx, prompt in enumerate(prompts):
            fut = pool.submit(
                _run_probe,
                base_url,
                model,
                prompt,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                api_key=api_key,
            )
            futures[fut] = idx
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch invariance harness")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Server base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier to include in request payload",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Optional text file (one prompt per line)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max tokens per probe request",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout seconds",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrent request count for batch-composition run",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key (Authorization: Bearer)",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write JSON report",
    )
    args = parser.parse_args()

    prompts = _load_prompts(args.prompts_file)
    concurrency = max(1, min(args.concurrency, len(prompts)))

    print(f"Running serial pass on {len(prompts)} prompts...")
    serial = _run_serial(
        prompts,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
        api_key=args.api_key,
    )

    print(f"Running concurrent pass (concurrency={concurrency})...")
    concurrent = _run_concurrent(
        prompts,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
        api_key=args.api_key,
        concurrency=concurrency,
    )

    exact_matches = 0
    per_prompt_agreement: list[float] = []
    rows: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        a = serial[idx]
        b = concurrent[idx]
        agreement = _token_agreement(a.output, b.output)
        exact = a.output == b.output
        if exact:
            exact_matches += 1
        per_prompt_agreement.append(agreement)
        rows.append(
            {
                "index": idx,
                "prompt": prompt,
                "serial_output": a.output,
                "concurrent_output": b.output,
                "token_agreement": round(agreement, 4),
                "exact_match": exact,
                "serial_latency_s": round(a.latency_s, 4),
                "concurrent_latency_s": round(b.latency_s, 4),
            }
        )

    exact_match_rate = exact_matches / len(prompts)
    token_agreement_rate = statistics.mean(per_prompt_agreement)

    print()
    print("Batch Invariance Report")
    print("-" * 60)
    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Exact output match rate: {exact_match_rate * 100:.2f}%")
    print(f"Token agreement rate:    {token_agreement_rate * 100:.2f}%")
    print("-" * 60)
    for row in rows:
        print(
            f"[{row['index']:02d}] agreement={row['token_agreement']:.2f} "
            f"exact={row['exact_match']}"
        )

    if token_agreement_rate < 0.95:
        print("Result: potential batch invariance violation (<95% token agreement).")
    else:
        print("Result: no significant batch invariance violation detected.")

    report = {
        "model": args.model,
        "base_url": args.base_url,
        "prompt_count": len(prompts),
        "max_tokens": args.max_tokens,
        "concurrency": concurrency,
        "exact_match_rate": exact_match_rate,
        "token_agreement_rate": token_agreement_rate,
        "rows": rows,
        "timestamp_epoch": int(time.time()),
    }
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote JSON report: {out_path}")


if __name__ == "__main__":
    main()
