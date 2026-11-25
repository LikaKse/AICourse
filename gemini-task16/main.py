"""CLI Дня 16: MCP Filesystem + локальные инструменты Дня 6.

Бесплатные режимы:
    python main.py --list-tools      # запустить MCP, показать инструменты и схемы
    python main.py --dry-run         # MCP + локальные схемы, без Gemini
    python main.py --demo-scripted   # весь смешанный путь без Gemini

Платный режим (только после отдельного разрешения):
    python main.py --demo            # реальная модель Gemini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import config
import local_tools
from agent import Agent, AgentResult, redact_secrets
from mcp_client import MCPConnectionError, MCPFilesystemClient
from mcp_schema import to_gemini_declaration
from scripted_client import ScriptedGeminiClient
from tool_registry import ToolRegistry

ROOT = config.PROJECT_ROOT


# ---------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------

class RedactSecretsFilter(logging.Filter):
    """Вырезает значение API-ключа из любой строки лога."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = redact_secrets(message)
        if cleaned != message:
            record.msg, record.args = cleaned, None
        return True


def setup_logging(verbose: bool, log_file: bool) -> Path | None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%H:%M:%S")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    log_path = None
    if log_file:
        log_dir = ROOT / config.LOG_DIR
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"agent_{datetime.now():%Y%m%d_%H%M%S}.log"
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(fmt)
        handler.addFilter(RedactSecretsFilter())
    logging.basicConfig(level=logging.WARNING, handlers=handlers, force=True)
    for name in ("agent", "mcp", "registry"):
        logging.getLogger(name).setLevel(logging.DEBUG if verbose else logging.INFO)
    for name in ("httpx", "httpcore", "google_genai", "google.genai", "mcp.client"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_path


# ---------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------

def make_client():
    """Создаёт Gemini-клиент. Ключ читается только из окружения."""
    api_key = os.environ.get(config.API_KEY_ENV)
    if not api_key:
        sys.exit(
            f"Не задана переменная окружения {config.API_KEY_ENV}. "
            f"Пример: export {config.API_KEY_ENV}=...  (ключ не сохраняйте в файлы проекта)"
        )
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=config.REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(
                attempts=config.RETRY_ATTEMPTS,
                http_status_codes=list(config.RETRY_STATUS_CODES),
            ),
        ),
    )


def confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        return input(f"{question} [y/N]: ").strip().lower() in {"y", "yes", "д", "да"}
    except EOFError:
        return False


def _digits(text: str) -> str:
    """'330 000' / '330,000' / '330.000' → '330000' внутри текста."""
    return re.sub(r"(?<=\d)[\s  .,](?=\d{3}\b)", "", text)


def print_mcp_summary(mcp: MCPFilesystemClient) -> None:
    print("=== MCP Filesystem ===")
    print(f"Сервер: {mcp.server_info or '?'}")
    print(f"Протокол: {mcp.protocol_version or '?'}")
    print(f"Node: {mcp.node_bin}")
    print(f"Разрешённая папка: {mcp.workspace}")
    print(f"Всего инструментов от сервера: {len(mcp.tools)}")
    for tool in mcp.tools:
        print(f"  - {tool.name}")
    exposed = mcp.exposed_tools()
    print(f"Отдано модели (read-only allowlist): {', '.join(t.name for t in exposed)}")


def print_request_preview(agent: Agent, user_message: str | None = None) -> None:
    print("\n=== Тело запроса к модели (без отправки) ===")
    payload = agent.request_preview(user_message)
    print(redact_secrets(json.dumps(payload, ensure_ascii=False, indent=2)))


def print_tool_calls(result: AgentResult) -> None:
    print("\n=== Вызовы инструментов ===")
    if not result.tool_calls:
        print("(вызовов не было)")
        return
    for index, call in enumerate(result.tool_calls, 1):
        print(f"[{index}] {call.source.upper():5} {call.name}  (итерация {call.iteration})")
        print(f"    args:   {json.dumps(call.args, ensure_ascii=False, default=str)}")
        print(f"    result: {json.dumps(call.response, ensure_ascii=False, default=str)}")


def check_mixed(result: AgentResult, expected_total: int = config.DEMO_EXPECTED_TOTAL) -> list[str]:
    """Проверяет смешанный сценарий read_text_file → calculate."""
    problems: list[str] = []
    if not result.ok:
        problems.append(f"статус {result.status}: {result.text}")
    successful = [call for call in result.tool_calls if call.ok]
    if not successful:
        return problems + ["не было успешных вызовов инструментов"]

    first = successful[0]
    if first.source != "mcp" or first.name != "read_text_file":
        problems.append(f"первый вызов {first.source}/{first.name}, ожидался mcp/read_text_file")

    calc = next((call for call in successful if call.name == "calculate" and call.source == "local"), None)
    if calc is None:
        problems.append("локальный calculate не вызывался")
    else:
        if calc.iteration <= first.iteration:
            problems.append("calculate вызван не после read_text_file")
        value = calc.response.get("result", {}).get("result") if isinstance(calc.response.get("result"), dict) else None
        if value != expected_total:
            problems.append(f"calculate вернул {value!r}, ожидалось {expected_total}")
        expression = str(calc.args.get("expression", ""))
        if not any(part in expression for part in ("12", "25000", "300000")):
            problems.append(f"в выражении calculate нет чисел из файла: {expression!r}")

    if str(expected_total) not in _digits(result.text):
        problems.append(f"в ответе нет числа {expected_total}")
    if not any(call.source == "mcp" for call in successful):
        problems.append("не было MCP-вызова")
    if not any(call.source == "local" for call in successful):
        problems.append("не было локального вызова")
    return problems


def _build_registry(mcp: MCPFilesystemClient, *, local_dispatcher=None) -> ToolRegistry:
    return ToolRegistry(
        local_declarations=local_tools.FUNCTION_DECLARATIONS,
        local_dispatcher=local_dispatcher or local_tools.dispatch,
        mcp_client=mcp,
        mcp_tools=mcp.exposed_tools(),
    )


# ---------------------------------------------------------------------
# Режимы
# ---------------------------------------------------------------------

async def cmd_list_tools() -> int:
    async with MCPFilesystemClient() as mcp:
        print_mcp_summary(mcp)
        print("\n=== Схемы всех инструментов сервера ===")
        for tool in mcp.tools:
            print(f"\n--- {tool.name} ---")
            print(json.dumps(tool.input_schema, ensure_ascii=False, indent=2))
        print("\n=== FunctionDeclaration, которые увидит Gemini ===")
        for tool in mcp.exposed_tools():
            declaration = to_gemini_declaration(tool)
            print(json.dumps(declaration.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
    return 0


async def cmd_dry_run() -> int:
    async with MCPFilesystemClient() as mcp:
        registry = _build_registry(mcp)
        agent = Agent(client=None, declarations=registry.declarations, dispatcher=registry.adispatch)
        print_mcp_summary(mcp)
        print(f"\nМодель: {config.MODEL}")
        print(f"Макс. итераций: {config.MAX_ITERATIONS}; thinking_level: {config.THINKING_LEVEL}; "
              f"max_output_tokens: {config.MAX_OUTPUT_TOKENS}")
        print(f"Node: {config.NODE_VERSION}; MCP server: {config.MCP_SERVER_VERSION}; "
              f"read-only: {config.MCP_READ_ONLY}; confine-roots: {config.MCP_CONFINE_ROOTS}")
        print("\n=== Реестр инструментов ===")
        for item in registry.describe():
            print(f"  {item['source']:5} {item['name']}  ← {item['original_name']}")
        print("\n=== Системная инструкция ===")
        print(config.SYSTEM_INSTRUCTION)
        print("\n=== FunctionDeclaration ===")
        for declaration in registry.declarations:
            print(json.dumps(declaration.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
        print_request_preview(agent, config.DEMO_PROMPT)
        print(f"\nДемо-запрос: {config.DEMO_PROMPT}")
        print(f"Ожидаемая цепочка: mcp/read_text_file → local/calculate → {config.DEMO_EXPECTED_TOTAL}")
    return 0


async def _run_demo(client, *, scripted: bool) -> int:
    async with MCPFilesystemClient() as mcp:
        registry = _build_registry(mcp)
        agent = Agent(client=client, declarations=registry.declarations, dispatcher=registry.adispatch)
        print_mcp_summary(mcp)
        print(f"\nРежим: {'scripted (без Gemini)' if scripted else 'реальная модель Gemini'}")
        print(f"Пользователь: {config.DEMO_PROMPT}")
        print_request_preview(agent, config.DEMO_PROMPT)
        result = await agent.arun(config.DEMO_PROMPT)
        print(f"\nАгент: {result.text}")
        print_tool_calls(result)
        problems = check_mixed(result)
        verdict = "PASS" if not problems else "FAIL: " + "; ".join(problems)
        print(f"\n[{verdict}] итераций: {result.iterations}, "
              f"токены in/out: {result.input_tokens}/{result.output_tokens}")
        return 0 if not problems else 1


async def cmd_demo_scripted() -> int:
    return await _run_demo(ScriptedGeminiClient(), scripted=True)


async def cmd_demo(assume_yes: bool) -> int:
    if not os.environ.get(config.API_KEY_ENV):
        print(f"Не задана переменная окружения {config.API_KEY_ENV}. "
              "Реальная модель не запускается. Для бесплатной проверки используйте --demo-scripted.",
              file=sys.stderr)
        return 2
    print(f"Будет выполнен один запрос к {config.MODEL} (платно, доли цента).")
    if not confirm("Запустить реальный вызов Gemini API?", assume_yes):
        print("Отменено.")
        return 1
    client = make_client()
    return await _run_demo(client, scripted=False)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="День 16: MCP Filesystem + локальные инструменты Дня 6"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-tools", action="store_true", help="MCP: инструменты и схемы (бесплатно)")
    mode.add_argument("--dry-run", action="store_true", help="MCP + локальные схемы, без Gemini")
    mode.add_argument("--demo-scripted", action="store_true", help="смешанный сценарий без Gemini")
    mode.add_argument("--demo", action="store_true", help="смешанный сценарий с реальной моделью (платно)")
    parser.add_argument("--yes", action="store_true", help="не спрашивать подтверждение платного вызова")
    parser.add_argument("--verbose", action="store_true", help="подробный лог (DEBUG)")
    parser.add_argument("--log-file", action="store_true", help="дублировать лог в logs/")
    args = parser.parse_args(argv)

    log_path = setup_logging(args.verbose, args.log_file)
    if log_path:
        print(f"Лог пишется в {log_path.relative_to(ROOT)}", file=sys.stderr)

    try:
        if args.list_tools:
            return asyncio.run(cmd_list_tools())
        if args.dry_run:
            return asyncio.run(cmd_dry_run())
        if args.demo_scripted:
            return asyncio.run(cmd_demo_scripted())
        return asyncio.run(cmd_demo(args.yes))
    except MCPConnectionError as exc:
        print(f"Ошибка MCP: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nПрервано.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
