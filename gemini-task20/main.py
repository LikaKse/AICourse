"""CLI Дня 20: Orchestration MCP."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

import config
from flow_agent import FlowAgent, FlowResult, result_json
from manager import MultiMCPManager
from server_specs import default_specs


def setup_logging(verbose: bool) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%H:%M:%S")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
    for name in ("manager", "agent"):
        logging.getLogger(name).setLevel(logging.DEBUG if verbose else logging.INFO)
    for name in ("mcp.client",):
        logging.getLogger(name).setLevel(logging.WARNING)


def print_tools(manager: MultiMCPManager) -> None:
    print(f"Доступные серверы: {', '.join(manager.available_servers) or 'нет'}")
    for item in manager.unavailable:
        print(f"[{item.alias}] UNAVAILABLE: {item.reason}")
    for tool in manager.tools:
        print(f"[{tool.server_alias}] {tool.public_name} ← {tool.original_name}")
        print(f"    {tool.description}")


def print_flow(result: FlowResult) -> None:
    for index, step in enumerate(result.steps, 1):
        print(
            f"[{index}] {step.server_alias}/{step.original_name} "
            f"({step.public_name}) status={step.status}"
        )
        print(f"    args: {json.dumps(step.args, ensure_ascii=False, default=str)}")
        if step.result is not None:
            print(f"    result: {json.dumps(step.result, ensure_ascii=False, default=str)}")
        if step.reason:
            print(f"    reason: {step.reason}")
    print(f"\nstatus: {result.status}")
    print(f"final: {result.final_text}")
    if result.unavailable:
        print(f"unavailable: {', '.join(result.unavailable)}")


async def run_scenario(
    query: str,
    *,
    catalog_down: bool = False,
) -> FlowResult:
    specs = default_specs(
        catalog_command="/definitely/missing/python" if catalog_down else None
    )
    manager = MultiMCPManager(specs)
    await manager.start()
    try:
        print(f"\n=== scenario: {'catalog-down' if catalog_down else 'full'} ===")
        print_tools(manager)
        result = await FlowAgent(manager).run(query)
        print_flow(result)
        print("result_json:")
        print(result_json(result))
        return result
    finally:
        await manager.aclose()


async def async_main(args) -> int:
    if args.list_tools:
        manager = MultiMCPManager(default_specs())
        await manager.start()
        try:
            print_tools(manager)
        finally:
            await manager.aclose()
        return 0

    if args.demo:
        full = await run_scenario(args.query or config.DEMO_QUERY)
        partial = await run_scenario(args.query or config.DEMO_QUERY, catalog_down=True)
        print("\n=== demo summary ===")
        print(f"full:   {full.status}")
        print(f"partial: {partial.status}")
        return 0 if full.status == "ok" and partial.status in {"partial", "failed"} else 1

    result = await run_scenario(
        args.query or config.DEMO_QUERY,
        catalog_down=args.scenario == "catalog-down",
    )
    if args.scenario == "catalog-down":
        return 0 if result.status in {"partial", "failed"} else 1
    return 0 if result.status == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="День 20: Orchestration MCP")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-tools", action="store_true", help="показать все MCP-инструменты")
    mode.add_argument("--demo", action="store_true", help="полный флоу и catalog-down")
    mode.add_argument("--scenario", choices=("full", "catalog-down"), help="один сценарий")
    parser.add_argument("--query", help="запрос агента")
    parser.add_argument("--verbose", action="store_true", help="подробный лог")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
