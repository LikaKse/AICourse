"""Фоновый планировщик: снимает данные и сохраняет отчёты через MCP."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import config
from mcp_client import MCPError, WeatherMCPClient
from models import normalize_iso, utc_now_iso

logger = logging.getLogger("scheduler")


class Scheduler:
    def __init__(
        self,
        client: WeatherMCPClient | Any,
        *,
        poll_interval: float = config.POLL_INTERVAL_SECONDS,
        report_every: int = config.REPORT_EVERY_N_OBSERVATIONS,
        report_period: float = config.REPORT_PERIOD_SECONDS,
        max_observations: int = config.MAX_OBSERVATIONS,
        reports_dir: Path | None = config.REPORTS_DIR,
        now_fn: Callable[[], str] = utc_now_iso,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if poll_interval < 0:
            raise ValueError("poll_interval не может быть отрицательным.")
        if report_every < 1:
            raise ValueError("report_every должен быть ≥ 1.")
        if report_period <= 0:
            raise ValueError("report_period должен быть > 0.")
        if max_observations < 0:
            raise ValueError("max_observations не может быть отрицательным.")
        self.client = client
        self.poll_interval = poll_interval
        self.report_every = report_every
        self.report_period = report_period
        self.max_observations = max_observations
        self.reports_dir = Path(reports_dir) if reports_dir else None
        self._now = now_fn
        self._sleep = sleep_fn
        self.saved_count = 0
        self.report_count = 0

    async def run(self) -> int:
        while True:
            try:
                await self.run_observation()
            except MCPError as exc:
                logger.error("Наблюдение не сохранено: %s", exc)
            except Exception as exc:
                logger.exception("Неожиданная ошибка наблюдения: %s", exc)

            if self.saved_count and self.saved_count % self.report_every == 0:
                await self._run_report_safe()

            if self.max_observations and self.saved_count >= self.max_observations:
                break
            await self._sleep(self.poll_interval)

        if self.saved_count and self.saved_count % self.report_every != 0:
            await self._run_report_safe()
        return self.saved_count

    async def _run_report_safe(self) -> None:
        try:
            await self.run_report()
        except Exception as exc:
            logger.error("Отчёт не построен: %s", exc)

    async def run_observation(self) -> dict:
        reading = await self.client.call_tool_structured("fetch_weather", {})
        if not isinstance(reading, dict):
            raise MCPError("fetch_weather вернул не structured-объект.")
        saved = await self.client.call_tool_structured(
            "save_observation",
            {
                "observed_at": reading["observed_at"],
                "temperature_c": reading["temperature_c"],
                "wind_speed_kmh": reading["wind_speed_kmh"],
                "source": reading.get("source", "open-meteo"),
                "raw_json": json.dumps(reading, ensure_ascii=False),
            },
        )
        if not isinstance(saved, dict):
            raise MCPError("save_observation вернул не structured-объект.")
        self.saved_count += 1
        print(
            f"run #{self.saved_count}: observed_at={reading['observed_at']} "
            f"temperature={reading['temperature_c']}°C "
            f"wind={reading['wind_speed_kmh']} km/h "
            f"fetched_at={saved['fetched_at']} saved_id={saved['id']}"
        )
        return saved

    async def run_report(self) -> dict:
        period_end = normalize_iso(self._now(), "period_end")
        end_dt = datetime.fromisoformat(period_end)
        period_start = (end_dt - timedelta(seconds=self.report_period)).astimezone(timezone.utc).isoformat(
            timespec="seconds"
        )
        period = {"period_start": period_start, "period_end": period_end}

        history = await self.client.call_tool_structured("read_observations", period)
        report = await self.client.call_tool_structured("build_report", period)
        if not isinstance(history, dict) or not isinstance(report, dict):
            raise MCPError("read_observations/build_report вернули не structured-объекты.")

        items = history.get("items", [])
        if report.get("record_count") != len(items):
            logger.warning(
                "Расхождение: read_observations=%d, build_report.record_count=%s",
                len(items),
                report.get("record_count"),
            )
        self.report_count += 1
        print(
            f"report #{self.report_count}: period={period_start}..{period_end} "
            f"count={report.get('record_count')} "
            f"ids={report.get('observation_ids')} "
            f"avg={report.get('temperature_avg')} "
            f"min={report.get('temperature_min')} "
            f"max={report.get('temperature_max')}"
        )
        if self.reports_dir is not None:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            path = self.reports_dir / f"report_{report.get('report_id', self.report_count):04d}.json"
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


def setup_logging(verbose: bool, log_file: bool) -> Path | None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%H:%M:%S")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    log_path = None
    if log_file:
        log_dir = config.PROJECT_ROOT / config.LOG_DIR
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"scheduler_{datetime.now():%Y%m%d_%H%M%S}.log"
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(fmt)
    logging.basicConfig(level=logging.WARNING, handlers=handlers, force=True)
    logging.getLogger("scheduler").setLevel(logging.DEBUG if verbose else logging.INFO)
    for name in ("httpx", "httpx2", "mcp.client"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_path


async def cmd_list_tools(args) -> int:
    async with WeatherMCPClient(db_path=args.db, api_base=args.api_base) as client:
        print(f"Сервер: {client.server_info or '?'}")
        print(f"Протокол: {client.protocol_version or '?'}")
        for tool in client.exposed_tools():
            print(f"\n--- {tool.name} ---")
            print(tool.description)
            print(json.dumps(tool.input_schema, ensure_ascii=False, indent=2))
    return 0


async def async_main(args) -> int:
    if args.list_tools:
        return await cmd_list_tools(args)

    if args.demo:
        poll_interval = args.poll_interval if args.poll_interval is not None else config.DEMO_POLL_INTERVAL
        report_every = args.report_every if args.report_every is not None else config.DEMO_REPORT_EVERY
        report_period = args.report_period if args.report_period is not None else config.DEMO_REPORT_PERIOD
        max_observations = (
            args.max_observations if args.max_observations is not None else config.DEMO_MAX_OBSERVATIONS
        )
        db_path = args.db or config.DEMO_DB_PATH
    elif args.once:
        poll_interval = 0.0
        report_every = 1
        report_period = args.report_period if args.report_period is not None else config.REPORT_PERIOD_SECONDS
        max_observations = 1
        db_path = args.db or config.DATABASE_PATH
    else:
        poll_interval = args.poll_interval if args.poll_interval is not None else config.POLL_INTERVAL_SECONDS
        report_every = args.report_every if args.report_every is not None else config.REPORT_EVERY_N_OBSERVATIONS
        report_period = args.report_period if args.report_period is not None else config.REPORT_PERIOD_SECONDS
        max_observations = (
            args.max_observations if args.max_observations is not None else config.MAX_OBSERVATIONS
        )
        db_path = args.db or config.DATABASE_PATH

    async with WeatherMCPClient(db_path=db_path, api_base=args.api_base) as client:
        scheduler = Scheduler(
            client,
            poll_interval=poll_interval,
            report_every=report_every,
            report_period=report_period,
            max_observations=max_observations,
        )
        print(
            f"Планировщик запущен: poll={poll_interval}s report_every={report_every} "
            f"period={report_period}s max={max_observations or '∞'} db={db_path}"
        )
        await scheduler.run()
        print(f"Готово: observations={scheduler.saved_count} reports={scheduler.report_count}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="День 18: планировщик погодных наблюдений через MCP")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="4 наблюдения и 2 отчёта, затем выход")
    mode.add_argument("--once", action="store_true", help="одно наблюдение и один отчёт")
    mode.add_argument("--list-tools", action="store_true", help="показать MCP-инструменты и схемы")
    parser.add_argument("--db", type=Path, help="путь к SQLite")
    parser.add_argument("--api-base", help="базовый URL Open-Meteo/заглушки")
    parser.add_argument("--poll-interval", type=float, help="секунды между наблюдениями")
    parser.add_argument("--report-every", type=int, help="отчёт каждые N наблюдений")
    parser.add_argument("--report-period", type=float, help="окно отчёта в секундах")
    parser.add_argument("--max-observations", type=int, help="остановиться после N наблюдений")
    parser.add_argument("--verbose", action="store_true", help="подробный лог")
    parser.add_argument("--log-file", action="store_true", help="дублировать лог в logs/")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = setup_logging(args.verbose, args.log_file)
    if log_path:
        print(f"Лог пишется в {log_path.relative_to(config.PROJECT_ROOT)}", file=sys.stderr)
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
