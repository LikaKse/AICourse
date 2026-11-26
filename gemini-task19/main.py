"""CLI Дня 19: детерминированная MCP-цепочка search → summarize → save_to_file."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import config
from chain_agent import ChainAgent, ChainResult, result_json
from mcp_client import MCPConnectionError, ChainMCPClient

SCENARIOS = ("success", "summarize-error", "empty-search")


def setup_logging(verbose: bool) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%H:%M:%S")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
    logging.getLogger("chain").setLevel(logging.DEBUG if verbose else logging.INFO)
    logging.getLogger("mcp").setLevel(logging.DEBUG if verbose else logging.INFO)
    for name in ("httpx2", "mcp.client"):
        logging.getLogger(name).setLevel(logging.WARNING)


def print_chain(result: ChainResult) -> None:
    for index, step in enumerate(result.steps, 1):
        print(f"[{index}/3] {step.name}  status={step.status}")
        print(f"        args: {json.dumps(step.args, ensure_ascii=False, default=str)}")
        if step.result is not None:
            print(f"        result: {json.dumps(step.result, ensure_ascii=False, default=str)}")
        if step.reason:
            print(f"        reason: {step.reason}")

    if result.status == "ok":
        print(f"chain: OK -> {result.saved_path}")
    else:
        print(f"chain: STOPPED at {result.stopped_at}: {result.reason}")
        print(f"completed steps: {result.completed_steps}")
        print("последующие инструменты не вызывались")


async def run_scenario(
    *,
    scenario: str,
    query: str | None,
    filename: str | None,
    api_base: str | None,
    output_dir: Path | None,
) -> ChainResult:
    fail_summarize = scenario == "summarize-error"
    if query is None:
        query = config.EMPTY_QUERY if scenario == "empty-search" else config.DEMO_QUERY
    if filename is None:
        filename = config.DEFAULT_FILENAME if scenario == "success" else config.ERROR_FILENAME

    print(f"\n=== scenario: {scenario} ===")
    print(f"query={query!r} filename={filename!r} fail_summarize={fail_summarize}")
    async with ChainMCPClient(
        api_base=api_base,
        output_dir=output_dir,
        fail_summarize=fail_summarize,
    ) as client:
        result = await ChainAgent(client).run(query, filename)
    print_chain(result)
    print("result:", result_json(result))
    return result


async def cmd_list_tools(api_base: str | None, output_dir: Path | None) -> int:
    async with ChainMCPClient(api_base=api_base, output_dir=output_dir) as client:
        print(f"Сервер: {client.server_info or '?'}")
        print(f"Протокол: {client.protocol_version or '?'}")
        for tool in client.exposed_tools():
            print(f"\n--- {tool.name} ---")
            print(tool.description)
            print(json.dumps(tool.input_schema, ensure_ascii=False, indent=2))
    return 0


async def async_main(args) -> int:
    if args.list_tools:
        return await cmd_list_tools(args.api_base, args.output_dir)

    if args.demo:
        results = []
        for scenario in SCENARIOS:
            results.append(await run_scenario(
                scenario=scenario,
                query=None,
                filename=None,
                api_base=args.api_base,
                output_dir=args.output_dir,
            ))
        ok = sum(result.status == "ok" for result in results)
        stopped = len(results) - ok
        print(f"\nИтого: OK={ok}, STOPPED={stopped}")
        return 0 if ok == 1 and stopped == 2 else 1

    result = await run_scenario(
        scenario=args.scenario,
        query=args.query,
        filename=args.filename,
        api_base=args.api_base,
        output_dir=args.output_dir,
    )
    return 0 if result.status == "ok" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="День 19: MCP-цепочка search → summarize → save_to_file")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-tools", action="store_true", help="показать три MCP-инструмента")
    mode.add_argument("--demo", action="store_true", help="успех + ошибка summarize + пустой search")
    mode.add_argument("--scenario", choices=SCENARIOS, help="один сценарий цепочки")
    parser.add_argument("--query", help="поисковый запрос")
    parser.add_argument("--filename", help="имя файла результата")
    parser.add_argument("--api-base", help="базовый URL JSONPlaceholder/заглушки")
    parser.add_argument("--output-dir", type=Path, help="папка output")
    parser.add_argument("--verbose", action="store_true", help="подробный лог")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        return asyncio.run(async_main(args))
    except MCPConnectionError as exc:
        print(f"Ошибка MCP: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
