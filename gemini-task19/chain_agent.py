"""Детерминированный оркестратор цепочки search → summarize → save_to_file.

Это не LLM: порядок вызовов, проверки и остановки заданы кодом.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

import config
from mcp_client import MCPCallResult
from models import SavedFile, SearchResult, Summary

logger = logging.getLogger("chain")


@dataclass
class StepRecord:
    name: str
    args: dict
    status: str                 # ok | error | empty | invalid
    result: dict | None = None
    reason: str | None = None


@dataclass
class ChainResult:
    status: str                 # ok | stopped
    stopped_at: str | None
    reason: str | None
    steps: list[StepRecord] = field(default_factory=list)
    summary: str | None = None
    saved_path: str | None = None

    @property
    def completed_steps(self) -> list[str]:
        return [step.name for step in self.steps if step.status == "ok"]


class ChainAgent:
    def __init__(self, client: Any) -> None:
        self.client = client

    async def run(self, query: str, filename: str = config.DEFAULT_FILENAME) -> ChainResult:
        steps: list[StepRecord] = []

        search_args = {"query": query}
        search_payload, error = await self._invoke("search", search_args)
        if error is not None:
            steps.append(StepRecord("search", search_args, "error", reason=error))
            return self._stopped("search", error, steps)
        try:
            search_result = SearchResult.model_validate(search_payload)
        except ValidationError as exc:
            reason = f"Некорректный ответ search: {exc.errors()[0].get('msg', str(exc))}"
            steps.append(StepRecord("search", search_args, "invalid", reason=reason))
            return self._stopped("search", reason, steps)
        steps.append(StepRecord("search", search_args, "ok", search_result.model_dump(by_alias=True)))
        if not search_result.items:
            reason = "search вернул пустой список публикаций."
            steps[-1].status = "empty"
            steps[-1].reason = reason
            return self._stopped("search", reason, steps)

        summarize_args = {
            "items": [item.model_dump(by_alias=True) for item in search_result.items],
        }
        summary_payload, error = await self._invoke("summarize", summarize_args)
        if error is not None:
            steps.append(StepRecord("summarize", summarize_args, "error", reason=error))
            return self._stopped("summarize", error, steps)
        try:
            summary = Summary.model_validate(summary_payload)
        except ValidationError as exc:
            reason = f"Некорректный ответ summarize: {exc.errors()[0].get('msg', str(exc))}"
            steps.append(StepRecord("summarize", summarize_args, "invalid", reason=reason))
            return self._stopped("summarize", reason, steps)
        if summary.source_count != len(search_result.items):
            reason = (
                f"summarize вернул source_count={summary.source_count}, "
                f"ожидалось {len(search_result.items)}."
            )
            steps.append(StepRecord("summarize", summarize_args, "invalid", reason=reason))
            return self._stopped("summarize", reason, steps)
        if not summary.summary.strip():
            reason = "summarize вернул пустое резюме."
            steps.append(StepRecord("summarize", summarize_args, "empty", reason=reason))
            return self._stopped("summarize", reason, steps)
        steps.append(StepRecord("summarize", summarize_args, "ok", summary.model_dump()))

        save_args = {"content": summary.summary, "filename": filename}
        save_payload, error = await self._invoke("save_to_file", save_args)
        if error is not None:
            steps.append(StepRecord("save_to_file", save_args, "error", reason=error))
            return self._stopped("save_to_file", error, steps)
        try:
            saved = SavedFile.model_validate(save_payload)
        except ValidationError as exc:
            reason = f"Некорректный ответ save_to_file: {exc.errors()[0].get('msg', str(exc))}"
            steps.append(StepRecord("save_to_file", save_args, "invalid", reason=reason))
            return self._stopped("save_to_file", reason, steps)
        expected_bytes = len(summary.summary.encode("utf-8"))
        if not saved.path or saved.bytes_written != expected_bytes:
            reason = (
                f"save_to_file вернул bytes_written={saved.bytes_written}, "
                f"ожидалось {expected_bytes}."
            )
            steps.append(StepRecord("save_to_file", save_args, "invalid", reason=reason))
            return self._stopped("save_to_file", reason, steps)
        steps.append(StepRecord("save_to_file", save_args, "ok", saved.model_dump()))
        logger.info("Цепочка завершена: %s", saved.path)
        return ChainResult(
            status="ok",
            stopped_at=None,
            reason=None,
            steps=steps,
            summary=summary.summary,
            saved_path=saved.path,
        )

    async def _invoke(self, name: str, args: dict) -> tuple[Any, str | None]:
        try:
            result: MCPCallResult = await self.client.call_tool(name, args)
        except Exception as exc:
            logger.error("%s не выполнен: %s", name, exc)
            return None, f"{type(exc).__name__}: {exc}"
        if result.is_error:
            reason = result.content or f"{name} завершился с ошибкой."
            logger.warning("%s вернул ошибку: %s", name, reason)
            return None, reason
        if result.structured is None:
            return None, f"{name} не вернул structuredContent."
        return result.structured, None

    @staticmethod
    def _stopped(step: str, reason: str, steps: list[StepRecord]) -> ChainResult:
        logger.warning("Цепочка остановлена на %s: %s", step, reason)
        return ChainResult(
            status="stopped",
            stopped_at=step,
            reason=reason,
            steps=steps,
        )


def result_json(result: ChainResult) -> str:
    return json.dumps({
        "status": result.status,
        "stopped_at": result.stopped_at,
        "reason": result.reason,
        "completed_steps": result.completed_steps,
        "summary": result.summary,
        "saved_path": result.saved_path,
    }, ensure_ascii=False, indent=2)
