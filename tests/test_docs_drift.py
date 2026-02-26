"""Docs-vs-code drift checks for public server contract."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_DOC_PATH = ROOT / "docs" / "guides" / "server.md"
CLI_PATH = ROOT / "vllm_mlx" / "cli.py"
SERVER_PATH = ROOT / "vllm_mlx" / "server.py"

HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
TRACKED_SERVER_OPTIONS = {
    "--host",
    "--port",
    "--api-key",
    "--rate-limit",
    "--repetition-policy",
    "--trust-requests-when-auth-disabled",
    "--memory-warn-threshold",
    "--memory-limit-threshold",
    "--memory-action",
    "--memory-monitor-interval",
    "--batch-divergence-monitor",
    "--batch-divergence-interval",
    "--batch-divergence-threshold",
    "--batch-divergence-action",
    "--timeout",
    "--runtime-mode",
    "--runtime-mode-threshold",
    "--effective-context-tokens",
    "--deterministic",
    "--strict-model-id",
    "--continuous-batching",
    "--cache-strategy",
    "--use-paged-cache",
    "--cache-memory-percent",
    "--disable-mllm-vision-cache",
    "--mllm-vision-cache-size",
    "--max-tokens",
    "--default-temperature",
    "--default-top-p",
    "--stream-interval",
    "--mcp-config",
    "--reasoning-parser",
    "--max-thinking-tokens",
    "--embedding-model",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _literal_value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    return ast.literal_eval(node)


def _extract_doc_endpoints(doc_text: str) -> set[str]:
    endpoints = set()
    for method, path in re.findall(
        r"\b(GET|POST|PUT|PATCH|DELETE)\s+(/[^\s`]+)", doc_text
    ):
        if path.startswith("/v1/") or path.startswith("/health"):
            endpoints.add(f"{method} {path}")
    return endpoints


def _extract_server_routes(server_source: str) -> set[str]:
    routes = set()
    tree = ast.parse(server_source)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if not isinstance(decorator.func, ast.Attribute):
                continue
            if not isinstance(decorator.func.value, ast.Name):
                continue
            if decorator.func.value.id != "app":
                continue

            method = decorator.func.attr.upper()
            if method not in HTTP_METHODS:
                continue
            if not decorator.args:
                continue

            path_node = decorator.args[0]
            if not isinstance(path_node, ast.Constant) or not isinstance(
                path_node.value, str
            ):
                continue
            routes.add(f"{method} {path_node.value}")

    return routes


def _extract_server_options_table_defaults(doc_text: str) -> dict[str, str]:
    lines = doc_text.splitlines()
    in_server_options = False
    defaults: dict[str, str] = {}

    for line in lines:
        if line.startswith("## Server Options"):
            in_server_options = True
            continue
        if in_server_options and line.startswith("## "):
            break
        if not in_server_options:
            continue

        stripped = line.strip()
        if not stripped.startswith("|"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        option = cells[0].strip("`")
        if option in {"Option", "--------"} or not option.startswith("--"):
            continue
        defaults[option] = cells[2]

    return defaults


def _extract_cli_serve_defaults(cli_source: str) -> dict[str, object]:
    defaults: dict[str, object] = {}
    tree = ast.parse(cli_source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != "serve_parser":
            continue
        if not node.args:
            continue

        arg0 = node.args[0]
        if not isinstance(arg0, ast.Constant) or not isinstance(arg0.value, str):
            continue
        option = arg0.value
        if not option.startswith("--"):
            continue

        action = None
        has_default = False
        default_value = None
        for kw in node.keywords:
            if kw.arg == "action":
                action = _literal_value(kw.value)
            elif kw.arg == "default":
                has_default = True
                default_value = _literal_value(kw.value)

        if has_default:
            defaults[option] = default_value
        elif action == "store_true":
            defaults[option] = False

    return defaults


def _extract_server_globals(server_source: str) -> dict[str, object]:
    values: dict[str, object] = {}
    tree = ast.parse(server_source)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name) or node.value is None:
                continue
            name = node.target.id
            value_node = node.value
        else:
            continue

        if name in {"_default_max_tokens", "_default_timeout"}:
            values[name] = _literal_value(value_node)

    return values


def _assert_doc_default_matches(option: str, doc_default: str, cli_default: object):
    doc_value = doc_default.strip().strip("`")

    if cli_default is None:
        assert (
            doc_value.lower() == "none"
        ), f"Default drift for {option}: docs={doc_default!r}, code={cli_default!r}"
        return

    if isinstance(cli_default, bool):
        expected = "true" if cli_default else "false"
        assert (
            doc_value.lower() == expected
        ), f"Default drift for {option}: docs={doc_default!r}, code={cli_default!r}"
        return

    if isinstance(cli_default, int):
        assert (
            int(doc_value) == cli_default
        ), f"Default drift for {option}: docs={doc_default!r}, code={cli_default!r}"
        return

    if isinstance(cli_default, float):
        assert (
            abs(float(doc_value) - cli_default) < 1e-9
        ), f"Default drift for {option}: docs={doc_default!r}, code={cli_default!r}"
        return

    assert doc_value == str(
        cli_default
    ), f"Default drift for {option}: docs={doc_default!r}, code={cli_default!r}"


def test_documented_endpoints_exist_in_server_routes():
    docs_endpoints = _extract_doc_endpoints(_read(SERVER_DOC_PATH))
    server_routes = _extract_server_routes(_read(SERVER_PATH))

    missing = sorted(docs_endpoints - server_routes)
    assert not missing, f"Documented endpoints missing from server routes: {missing}"


def test_server_option_defaults_match_cli_definition():
    docs_defaults = _extract_server_options_table_defaults(_read(SERVER_DOC_PATH))
    cli_defaults = _extract_cli_serve_defaults(_read(CLI_PATH))

    missing_in_docs = sorted(TRACKED_SERVER_OPTIONS - set(docs_defaults))
    missing_in_cli = sorted(TRACKED_SERVER_OPTIONS - set(cli_defaults))

    assert (
        not missing_in_docs
    ), f"Tracked options missing in docs table: {missing_in_docs}"
    assert (
        not missing_in_cli
    ), f"Tracked options missing in CLI parser: {missing_in_cli}"

    for option in sorted(TRACKED_SERVER_OPTIONS):
        _assert_doc_default_matches(option, docs_defaults[option], cli_defaults[option])


def test_key_defaults_match_server_globals_cli_and_docs():
    docs_defaults = _extract_server_options_table_defaults(_read(SERVER_DOC_PATH))
    cli_defaults = _extract_cli_serve_defaults(_read(CLI_PATH))
    server_globals = _extract_server_globals(_read(SERVER_PATH))

    assert cli_defaults["--max-tokens"] == server_globals["_default_max_tokens"]
    assert abs(cli_defaults["--timeout"] - server_globals["_default_timeout"]) < 1e-9

    assert (
        int(docs_defaults["--max-tokens"].strip())
        == server_globals["_default_max_tokens"]
    )
    assert (
        abs(
            float(docs_defaults["--timeout"].strip())
            - server_globals["_default_timeout"]
        )
        < 1e-9
    )
