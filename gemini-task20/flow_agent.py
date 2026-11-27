"""Детерминированный state-machine агент для нескольких MCP-серверов."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

import config
from manager import MultiMCPManager, ToolCallResult
from models import (
    CatalogItem,
    CatalogSearch,
    FileContent,
    Note,
    NotesSearch,
    NotesSummary,
)

logger = logging.getLogger("agent")


@dataclass(frozen=True)
class Intent:
    context: bool
    catalog: bool
    notes: bool


def classify_intent(query: str) -> Intent:
    """Явные правила намерения; не скрытая эвристика в глубине агента."""
    text = query.casefold()
    wants_context = any(word in text for word in ("контекст", "context", "файл", "file", "закупк"))
    wants_notes = any(word in text for word in ("заметк", "note", "notes"))
    explicit_catalog = any(word in text for word in ("каталог", "catalog", "товар", "product", "ноутбук"))
    # "laptop" — поисковый термин; если запрос явно про заметки, не включаем каталог.
    wants_catalog = explicit_catalog or ("laptop" in text and not wants_notes)
    if not (wants_context or wants_catalog or wants_notes):
        # Общий запрос: пробуем оба содержательных источника, без файла.
        wants_catalog = True
        wants_notes = True
    return Intent(context=wants_context, catalog=wants_catalog, notes=wants_notes)


def search_term(query: str) -> str:
    text = query.casefold()
    if "ноутбук" in text or "laptop" in text:
        return "laptop"
    return query.strip()


@dataclass
class AgentState:
    context: str | None = None
    context_done: bool = False
    catalog_items: list[CatalogItem] | None = None
    catalog_done: bool = False
    item: CatalogItem | None = None
    item_done: bool = False
    notes: list[Note] | None = None
    notes_done: bool = False
    note: Note | None = None
    note_done: bool = False
    summary: str | None = None
    summary_done: bool = False
    skipped: list[tuple[str, str]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    attempted: set[str] = field(default_factory=set)


@dataclass
class StepRecord:
    public_name: str
    server_alias: str
    original_name: str
    args: dict
    status: str                 # ok | error | empty | invalid | skipped
    result: Any = None
    reason: str | None = None


@dataclass
class FlowResult:
    status: str                 # ok | partial | failed
    final_text: str
    steps: list[StepRecord]
    completed_tools: list[str]
    unavailable: list[str]


@dataclass(frozen=True)
class Action:
    key: str
    public_name: str
    args: dict


class FlowAgent:
    def __init__(self, manager: MultiMCPManager) -> None:
        self.manager = manager

    async def run(self, query: str) -> FlowResult:
        intent = classify_intent(query)
        state = AgentState()
        steps: list[StepRecord] = []
        term = search_term(query)

        for _ in range(12):
            action = self._choose_action(state, intent, term)
            if action is None:
                break
            result = await self.manager.call_tool(action.public_name, action.args)
            step = self._execute(action, result, state)
            steps.append(step)
            if step.status != "ok":
                logger.warning("Шаг %s: %s", action.public_name, step.reason)
            if step.status in {"error", "invalid"}:
                state.problems.append(f"{step.public_name}: {step.reason}")
            if state.context_done and state.catalog_done and state.item_done and \
                    state.notes_done and state.note_done and state.summary_done:
                # Если все ветки уже завершены, следующая итерация вернёт None.
                pass

        final_text = self._final_text(intent, state)
        requested_done = self._requested_done(intent, state)
        has_data = any((
            state.context is not None,
            state.item is not None,
            state.note is not None,
            state.summary is not None,
        ))
        if not has_data:
            status = "failed"
        elif requested_done and not state.skipped:
            status = "ok"
        else:
            status = "partial"
        return FlowResult(
            status=status,
            final_text=final_text,
            steps=steps,
            completed_tools=[step.public_name for step in steps if step.status == "ok"],
            unavailable=[item.alias for item in self.manager.unavailable],
        )

    def _has(self, state: AgentState, public_name: str, key: str) -> bool:
        if public_name in self.manager.routes:
            return True
        state.attempted.add(key)
        state.skipped.append((public_name, "инструмент недоступен: сервер не подключён"))
        return False

    def _choose_action(self, state: AgentState, intent: Intent, term: str) -> Action | None:
        if intent.context and not state.context_done:
            state.context_done = True
            if not self._has(state, "filesystem__read_text_file", "context"):
                return self._choose_action(state, intent, term)
            return Action("context", "filesystem__read_text_file", {"path": str(config.CONTEXT_FILE)})

        if intent.catalog and not state.catalog_done:
            state.catalog_done = True
            if not self._has(state, "catalog__search", "catalog_search"):
                return self._choose_action(state, intent, term)
            return Action("catalog_search", "catalog__search", {"query": term})

        if state.catalog_items and not state.item_done:
            state.item_done = True
            if not self._has(state, "catalog__get_item", "catalog_item"):
                return self._choose_action(state, intent, term)
            return Action("catalog_item", "catalog__get_item", {"item_id": state.catalog_items[0].id})

        if intent.notes and not state.notes_done:
            state.notes_done = True
            if not self._has(state, "notes__search", "notes_search"):
                return self._choose_action(state, intent, term)
            return Action("notes_search", "notes__search", {"query": term})

        if state.notes and not state.note_done:
            state.note_done = True
            if not self._has(state, "notes__get_note", "notes_note"):
                return self._choose_action(state, intent, term)
            return Action("notes_note", "notes__get_note", {"note_id": state.notes[0].id})

        if state.note is not None and not state.summary_done:
            state.summary_done = True
            if not self._has(state, "notes__summarize", "notes_summary"):
                return self._choose_action(state, intent, term)
            return Action("notes_summary", "notes__summarize", {"note_ids": [state.note.id]})

        return None

    def _execute(self, action: Action, result: ToolCallResult, state: AgentState) -> StepRecord:
        if result.is_error:
            return self._record(action, result, "error", reason=result.error_text)
        payload = result.structured
        if payload is None:
            return self._record(action, result, "invalid", reason="нет structuredContent")
        try:
            if action.key == "context":
                model = FileContent.model_validate(payload)
                state.context = model.content
            elif action.key == "catalog_search":
                model = CatalogSearch.model_validate(payload)
                state.catalog_items = model.items
                if not model.items:
                    return self._record(action, result, "empty", reason="товары не найдены")
            elif action.key == "catalog_item":
                model = CatalogItem.model_validate(payload)
                state.item = model
            elif action.key == "notes_search":
                model = NotesSearch.model_validate(payload)
                state.notes = model.items
                if not model.items:
                    return self._record(action, result, "empty", reason="заметки не найдены")
            elif action.key == "notes_note":
                model = Note.model_validate(payload)
                state.note = model
            elif action.key == "notes_summary":
                model = NotesSummary.model_validate(payload)
                if not model.summary.strip():
                    return self._record(action, result, "empty", reason="пустое резюме")
                state.summary = model.summary
            else:
                return self._record(action, result, "invalid", reason=f"неизвестное действие {action.key}")
        except ValidationError as exc:
            return self._record(action, result, "invalid", reason=f"некорректный ответ: {exc.errors()[0].get('msg', str(exc))}")
        return self._record(action, result, "ok", result_=payload)

    @staticmethod
    def _record(
        action: Action,
        result: ToolCallResult,
        status: str,
        *,
        result_: Any = None,
        reason: str | None = None,
    ) -> StepRecord:
        return StepRecord(
            public_name=action.public_name,
            server_alias=result.server_alias or action.public_name.split("__", 1)[0],
            original_name=result.original_name or action.public_name.split("__", 1)[1],
            args=action.args,
            status=status,
            result=result_ if result_ is not None else result.structured,
            reason=reason,
        )

    def _requested_done(self, intent: Intent, state: AgentState) -> bool:
        checks = []
        if intent.context:
            checks.append(state.context is not None)
        if intent.catalog:
            checks.append(state.item is not None or state.catalog_items == [])
        if intent.notes:
            checks.append(state.summary is not None or state.notes == [])
        return all(checks) if checks else True

    def _final_text(self, intent: Intent, state: AgentState) -> str:
        has_data = any((
            state.context is not None,
            state.item is not None,
            state.note is not None,
            state.summary is not None,
        ))
        parts: list[str] = []
        if not has_data:
            parts.append("Нет доступных данных; результат не выдумывается.")
        if state.context is not None:
            text = state.context.strip() or "(файл пуст)"
            parts.append(f"Контекст: {text}")
        if state.item is not None:
            parts.append(
                f"Товар: {state.item.name} — {state.item.description} "
                f"({state.item.price_usd} USD)"
            )
        elif intent.catalog and state.catalog_items == []:
            parts.append("Каталог: товары не найдены.")
        if state.note is not None:
            parts.append(f"Заметка: {state.note.title} — {state.note.text}")
        elif intent.notes and state.notes == []:
            parts.append("Заметки: не найдены.")
        if state.summary:
            parts.append(f"Сводка: {state.summary}")
        if state.problems:
            parts.append("Ошибки: " + "; ".join(state.problems))
        if state.skipped:
            parts.append("Пропущено: " + "; ".join(f"{name} ({reason})" for name, reason in state.skipped))
        if self.manager.unavailable:
            parts.append(
                "Недоступные серверы: " + ", ".join(item.alias for item in self.manager.unavailable)
            )
        if not parts:
            return "Нет доступных данных; результат не выдумывается."
        return "\n".join(parts)


def result_json(result: FlowResult) -> str:
    return json.dumps({
        "status": result.status,
        "final_text": result.final_text,
        "completed_tools": result.completed_tools,
        "unavailable": result.unavailable,
        "steps": [
            {
                "public_name": step.public_name,
                "server": step.server_alias,
                "tool": step.original_name,
                "args": step.args,
                "status": step.status,
                "reason": step.reason,
            }
            for step in result.steps
        ],
    }, ensure_ascii=False, indent=2)
