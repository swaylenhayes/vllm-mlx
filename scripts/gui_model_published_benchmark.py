#!/usr/bin/env python3
"""Run published GUI-model benchmark prompts against a Hugging Face repo id."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOME_DIR = Path.home()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_mlx.api.utils import is_mllm_model
from vllm_mlx.benchmark import get_mlx_memory_info, reset_mlx_peak_memory


@dataclass
class PromptCase:
    name: str
    image: str
    prompt: str


def _safe_package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _safe_git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        try:
            return f"~/{resolved.relative_to(HOME_DIR)}"
        except ValueError:
            return f"[external] {resolved.name}"


def _derive_packet_id(packet_name: str | None, cohort: str) -> str:
    if not packet_name:
        return cohort
    cleaned = packet_name.strip()
    if not cleaned or "/" in cleaned or "\\" in cleaned:
        return cohort
    return cleaned


def _runtime_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx": _safe_package_version("mlx"),
        "mlx_vlm": _safe_package_version("mlx-vlm"),
        "vllm_mlx_commit": _safe_git_commit(REPO_ROOT),
    }


def _load_seed_cases(path: Path) -> tuple[str | None, int | None, list[PromptCase]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    packet = payload.get("packet")
    prefill_step_size = payload.get("prefill_step_size")
    results = payload.get("results") or []
    cases = [
        PromptCase(
            name=str(result["name"]),
            image=str(result["image"]),
            prompt=str(result["prompt"]),
        )
        for result in results
    ]
    if not cases:
        raise ValueError(f"No prompt cases found in {path}")
    return packet, prefill_step_size, cases


def _result_from_generate(case: PromptCase, output, elapsed_s: float, peak_memory_gb: float) -> dict:
    prompt_tokens = int(getattr(output, "prompt_tokens", 0) or 0)
    generation_tokens = int(getattr(output, "generation_tokens", 0) or 0)
    text = getattr(output, "text", str(output))
    return {
        "name": case.name,
        "image": case.image,
        "prompt": case.prompt,
        "text": text,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prompt_tps": round(prompt_tokens / elapsed_s, 4) if elapsed_s > 0 and prompt_tokens > 0 else 0.0,
        "generation_tps": round(generation_tokens / elapsed_s, 4) if elapsed_s > 0 and generation_tokens > 0 else 0.0,
        "peak_memory": round(peak_memory_gb, 4),
        "elapsed_s": round(elapsed_s, 4),
    }


def _run_stream_check(model, processor, case: PromptCase, *, max_tokens: int, temperature: float, top_p: float) -> dict:
    from mlx_vlm import stream_generate

    reset_mlx_peak_memory()
    start = time.perf_counter()
    chunks: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    first_chunk = ""

    for chunk in stream_generate(
        model,
        processor,
        case.prompt,
        [case.image],
        max_tokens=max_tokens,
        temp=temperature,
        top_p=top_p,
    ):
        text = getattr(chunk, "text", str(chunk))
        if text:
            if not first_chunk:
                first_chunk = text
            chunks.append(text)
        prompt_tokens = int(getattr(chunk, "prompt_tokens", prompt_tokens) or prompt_tokens)
        completion_tokens = int(getattr(chunk, "completion_tokens", completion_tokens) or completion_tokens)

    elapsed_s = time.perf_counter() - start
    peak_memory = get_mlx_memory_info(reset_peak=True).get("peak_memory_gb", 0.0)
    full_text = "".join(chunks)
    if completion_tokens <= 0 and full_text:
        completion_tokens = len(full_text.split())
    return {
        "name": f"{case.name}_stream",
        "image": case.image,
        "prompt": case.prompt,
        "text": full_text,
        "first_chunk": first_chunk,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": completion_tokens,
        "generation_tps": round(completion_tokens / elapsed_s, 4) if elapsed_s > 0 and completion_tokens > 0 else 0.0,
        "peak_memory": round(peak_memory, 4),
        "elapsed_s": round(elapsed_s, 4),
    }


def _write_markdown_note(
    *,
    out_path: Path,
    model_id: str,
    cohort: str,
    packet_doc: Path,
    seed_json: Path,
    packet_id: str,
    load_time_s: float,
    remote_sanity: dict,
    results: list[dict],
    stream_check: dict,
    reference_model: str | None,
    environment: dict[str, str],
    seed_provenance_note: str | None,
) -> None:
    lines = [
        "# Published GUI Benchmark Run Note",
        "",
        f"Date: `{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}`",
        "",
        "## Run",
        "",
        f"- model: `{model_id}`",
        f"- cohort: `{cohort}`",
        f"- reference: `{reference_model or 'n/a'}`",
        f"- packet id: `{packet_id}`",
        f"- packet doc: `{_display_path(packet_doc)}`",
        f"- seed json: `{_display_path(seed_json)}`",
        f"- load time: `{load_time_s:.2f}s`",
        "",
        "## Environment",
        "",
        f"- python: `{environment['python']}`",
        f"- platform: `{environment['platform']}`",
        f"- machine: `{environment['machine']}`",
        f"- mlx: `{environment['mlx']}`",
        f"- mlx-vlm: `{environment['mlx_vlm']}`",
        f"- vllm-mlx commit: `{environment['vllm_mlx_commit']}`",
        f"- runner: `scripts/gui_model_published_benchmark.py`",
        "",
        "## Remote Sanity",
        "",
        f"- `is_mllm_model`: `{remote_sanity['is_mllm_model']}`",
        f"- packet doc exists: `{remote_sanity['packet_doc_exists']}`",
        f"- seed json exists: `{remote_sanity['seed_json_exists']}`",
        "",
    ]

    if seed_provenance_note:
        lines.extend(
            [
                "## Seed Provenance",
                "",
                f"- {seed_provenance_note}",
                "",
            ]
        )

    lines.extend(
        [
        "## Prompt Results",
        "",
        ]
    )

    for result in results:
        preview = result["text"].replace("\n", " ").strip()
        if len(preview) > 180:
            preview = preview[:177] + "..."
        lines.extend(
            [
                f"- `{result['name']}`: `{result['elapsed_s']:.2f}s`, `{result['peak_memory']:.2f} GB`, "
                f"`prompt={result['prompt_tokens']}`, `gen={result['generation_tokens']}`",
                f"  preview: `{preview}`",
            ]
        )

    stream_preview = stream_check["text"].replace("\n", " ").strip()
    if len(stream_preview) > 180:
        stream_preview = stream_preview[:177] + "..."
    lines.extend(
        [
            "",
            "## Stream Check",
            "",
            f"- `{stream_check['name']}`: `{stream_check['elapsed_s']:.2f}s`, `{stream_check['peak_memory']:.2f} GB`, "
            f"`prompt={stream_check['prompt_tokens']}`, `gen={stream_check['generation_tokens']}`",
            f"  first chunk: `{stream_check['first_chunk']}`",
            f"  preview: `{stream_preview}`",
            "",
            "## Status",
            "",
            "- `completed`",
            "",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cohort", choices=["track-a", "track-b"], required=True)
    parser.add_argument("--packet-doc", required=True)
    parser.add_argument("--seed-json", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--note-out", required=True)
    parser.add_argument("--status-out", required=True)
    parser.add_argument("--packet-id")
    parser.add_argument("--seed-provenance-note")
    parser.add_argument("--reference-model")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    packet_doc = Path(args.packet_doc)
    seed_json = Path(args.seed_json)
    json_out = Path(args.json_out)
    note_out = Path(args.note_out)
    status_out = Path(args.status_out)

    packet_name, prefill_step_size, cases = _load_seed_cases(seed_json)
    packet_id = args.packet_id or _derive_packet_id(packet_name, args.cohort)
    environment = _runtime_environment()
    remote_sanity = {
        "is_mllm_model": is_mllm_model(args.model_id),
        "packet_doc_exists": packet_doc.exists(),
        "seed_json_exists": seed_json.exists(),
    }
    if not all(remote_sanity.values()):
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(
                {
                    "model_path": args.model_id,
                    "packet_id": packet_id,
                    "packet_doc": _display_path(packet_doc),
                    "seed_json": _display_path(seed_json),
                    "seed_provenance_note": args.seed_provenance_note,
                    "environment": environment,
                    "remote_sanity": remote_sanity,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        note_out.parent.mkdir(parents=True, exist_ok=True)
        note_out.write_text(
            "# Published GUI Benchmark Run Note\n\n## Status\n\n- `failed_remote_sanity`\n",
            encoding="utf-8",
        )
        status_out.parent.mkdir(parents=True, exist_ok=True)
        status_out.write_text("failed_remote_sanity\n", encoding="utf-8")
        raise SystemExit(2)

    from mlx_vlm import generate, load

    load_start = time.perf_counter()
    model, processor = load(args.model_id)
    load_time_s = time.perf_counter() - load_start

    results: list[dict] = []
    for case in cases:
        reset_mlx_peak_memory()
        start = time.perf_counter()
        output = generate(
            model,
            processor,
            case.prompt,
            [case.image],
            max_tokens=args.max_tokens,
            temp=args.temperature,
            top_p=args.top_p,
            verbose=False,
        )
        elapsed_s = time.perf_counter() - start
        peak_memory = get_mlx_memory_info(reset_peak=True).get("peak_memory_gb", 0.0)
        results.append(_result_from_generate(case, output, elapsed_s, peak_memory))

    stream_case = cases[-1]
    stream_check = _run_stream_check(
        model,
        processor,
        stream_case,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    payload = {
        "model_path": args.model_id,
        "cohort": args.cohort,
        "reference_model": args.reference_model,
        "packet_id": packet_id,
        "packet_doc": _display_path(packet_doc),
        "seed_json": _display_path(seed_json),
        "seed_provenance_note": args.seed_provenance_note,
        "packet": packet_name,
        "prefill_step_size": prefill_step_size,
        "runner": "scripts/gui_model_published_benchmark.py",
        "runner_commit": environment["vllm_mlx_commit"],
        "environment": environment,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "load_time_s": round(load_time_s, 4),
        "remote_sanity": remote_sanity,
        "results": results,
        "stream_check": stream_check,
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown_note(
        out_path=note_out,
        model_id=args.model_id,
        cohort=args.cohort,
        packet_doc=packet_doc,
        seed_json=seed_json,
        packet_id=packet_id,
        load_time_s=load_time_s,
        remote_sanity=remote_sanity,
        results=results,
        stream_check=stream_check,
        reference_model=args.reference_model,
        environment=environment,
        seed_provenance_note=args.seed_provenance_note,
    )
    status_out.parent.mkdir(parents=True, exist_ok=True)
    status_out.write_text("completed\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
