"""CLI Дня 17: собственный MCP get_task + локальные инструменты.

Бесплатные режимы:
    python main.py --check-api
    python main.py --list-tools
    python main.py --dry-run
    python main.py --demo-scripted

Платный режим (только после отдельного разрешения):
    python main.py --demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import config
import local_tools
from agent import Agent, AgentResult, redact_secrets
from mcp_client import MCPConnectionError, TaskMCPClient
from mcp_schema import to_gemini_declaration
from scripted_client import ScriptedGeminiClient
from task_api import TaskApi
from tool_registry import ToolRegistry

ROOT = config.PROJECT_ROOT


class RedactSecretsFilter(logging.Filter):
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
    for name in ("httpx", "httpcore", "google_genai", "google.genai", "mcp.client", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_path


def make_client():
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


def check_task(result: AgentResult, expected_id: int = config.TASK_ID) -> list[str]:
    problems: list[str] = []
    if not result.ok:
        problems.append(f"статус {result.status}: {result.text}")
        return problems

    successful = [call for call in result.tool_calls if call.ok]
    if not successful:
        return problems + ["не было успешных вызовов инструментов"]

    first = successful[0]
    if first.source != "mcp" or first.name != "get_task":
        problems.append(f"первый вызов {first.source}/{first.name}, ожидался mcp/get_task")
        return problems

    task = first.response.get("result")
    if not isinstance(task, dict):
        problems.append(f"get_task вернул неожиданный payload: {first.response!r}")
        return problems
    if task.get("id") != expected_id:
        problems.append(f"get_task вернул id={task.get('id')!r}, ожидался {expected_id}")
    title = task.get("title")
    if not isinstance(title, str) or not title:
        problems.append("в результате get_task нет title")
    if not isinstance(task.get("completed"), bool):
        problems.append("в результате get_task нет completed")
    if str(expected_id) not in result.text:
        problems.append(f"в ответе нет id {expected_id}")
    text_lower = result.text.lower()
    first_word = (title.split()[0].lower() if isinstance(title, str) and title else "")
    if first_word and first_word not in text_lower and not any(
        word in text_lower for word in ("не выполн", "выполнена", "false")
    ):
        problems.append("в ответе не использованы данные задачи (title/статус)")
    return problems


async def cmd_check_api() -> int:
    task = await TaskApi().fetch_task(config.TASK_ID)
    print(json.dumps(task.model_dump(mode="json", by_alias=True), ensure_ascii=False, indent=2))
    return 0


async def cmd_list_tools() -> int:
    async with TaskMCPClient() as mcp:
        print("=== MCP ===")
        print(f"Сервер: {mcp.server_info or '?'}")
        print(f"Протокол: {mcp.protocol_version or '?'}")
        print(f"Инструменты: {', '.join(tool.name for tool in mcp.tools)}")
        for tool in mcp.exposed_tools():
            print(f"\n--- {tool.name} ---")
            print(tool.description)
            print(json.dumps(tool.input_schema, ensure_ascii=False, indent=2))
            declaration = to_gemini_declaration(tool)
            print("FunctionDeclaration:")
            print(json.dumps(declaration.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
    return 0


def _build_registry(mcp: TaskMCPClient) -> ToolRegistry:
    return ToolRegistry(
        local_declarations=local_tools.FUNCTION_DECLARATIONS,
        local_dispatcher=local_tools.dispatch,
        mcp_client=mcp,
        mcp_tools=mcp.exposed_tools(),
    )


async def cmd_dry_run() -> int:
    async with TaskMCPClient() as mcp:
        registry = _build_registry(mcp)
        agent = Agent(client=None, declarations=registry.declarations, dispatcher=registry.adispatch)
        print(f"Сервер: {mcp.server_info or '?'} / протокол {mcp.protocol_version or '?'}")
        print(f"Модель: {config.MODEL}; max_iterations: {config.MAX_ITERATIONS}")
        print("\n=== Реестр инструментов ===")
        for item in registry.describe():
            print(f"  {item['source']:5} {item['name']}  ← {item['original_name']}")
        print("\n=== System instruction ===")
        print(config.SYSTEM_INSTRUCTION)
        print("\n=== FunctionDeclaration ===")
        for declaration in registry.declarations:
            print(json.dumps(declaration.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
        print_request_preview(agent, config.DEMO_PROMPT)
        print(f"\nДемо-запрос: {config.DEMO_PROMPT}")
        print(f"Ожидаемая цепочка: mcp/get_task(task_id={config.TASK_ID}) → ответ")
    return 0


async def _run_demo(client, *, scripted: bool) -> int:
    async with TaskMCPClient() as mcp:
        registry = _build_registry(mcp)
        agent = Agent(client=client, declarations=registry.declarations, dispatcher=registry.adispatch)
        print(f"Сервер: {mcp.server_info or '?'} / протокол {mcp.protocol_version or '?'}")
        print(f"Режим: {'scripted (без Gemini)' if scripted else 'реальная модель Gemini'}")
        print(f"Пользователь: {config.DEMO_PROMPT}")
        print_request_preview(agent, config.DEMO_PROMPT)
        result = await agent.arun(config.DEMO_PROMPT)
        print(f"\nАгент: {result.text}")
        print_tool_calls(result)
        problems = check_task(result)
        verdict = "PASS" if not problems else "FAIL: " + "; ".join(problems)
        print(f"\n[{verdict}] итераций: {result.iterations}, "
              f"токены in/out: {result.input_tokens}/{result.output_tokens}")
        return 0 if not problems else 1


async def cmd_demo_scripted() -> int:
    return await _run_demo(ScriptedGeminiClient(), scripted=True)


async def cmd_demo(assume_yes: bool) -> int:
    if not os.environ.get(config.API_KEY_ENV):
        print(f"Не задана переменная окружения {config.API_KEY_ENV}. "
              "Реальная модель не запускается. Используйте --demo-scripted.",
              file=sys.stderr)
        return 2
    print(f"Будет выполнен один запрос к {config.MODEL} (платно, доли цента).")
    if not confirm("Запустить реальный вызов Gemini API?", assume_yes):
        print("Отменено.")
        return 1
    return await _run_demo(make_client(), scripted=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="День 17: собственный MCP get_task")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-api", action="store_true", help="проверить JSONPlaceholder напрямую")
    mode.add_argument("--list-tools", action="store_true", help="показать MCP get_task и схему")
    mode.add_argument("--dry-run", action="store_true", help="MCP + схемы, без Gemini")
    mode.add_argument("--demo-scripted", action="store_true", help="полный цикл без Gemini")
    mode.add_argument("--demo", action="store_true", help="полный цикл с реальной моделью (платно)")
    parser.add_argument("--yes", action="store_true", help="не спрашивать подтверждение платного вызова")
    parser.add_argument("--verbose", action="store_true", help="подробный лог (DEBUG)")
    parser.add_argument("--log-file", action="store_true", help="дублировать лог в logs/")
    args = parser.parse_args(argv)

    log_path = setup_logging(args.verbose, args.log_file)
    if log_path:
        print(f"Лог пишется в {log_path.relative_to(ROOT)}", file=sys.stderr)

    try:
        if args.check_api:
            return asyncio.run(cmd_check_api())
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
