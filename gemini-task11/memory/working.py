"""Рабочая память: данные ТЕКУЩЕЙ задачи → state/working_memory.json.

    {"task": {"task_id", "created_at", "values": {поле: запись}, "items": {constraints: [...], results: [...]},
              "history": [...]} | null,
     "archived": [завершённые задачи], "version": N, "journal": [...]}

Запись — операции layer=working, прошедшие rules.check_working_op. Значения обновляются по ходу
задачи (прежние — в history). Завершённая задача (status=done) в промпт не попадает и
НИКОГДА не переносится в долговременную память: такого пути в коде нет.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

from google.genai import types

import config
from memory.base import MemoryLayer
from memory.evidence import same_value
from memory.rules import LIST_FIELDS, SCALAR_FIELDS, check_working_op
from memory.storage import MemoryLoadError, now, text_content

FIELD_TITLES = {
    "goal": "Цель", "deadline": "Срок", "budget": "Бюджет", "status": "Статус", "next_step": "Следующий шаг",
    "constraints": "Ограничения задачи", "results": "Промежуточные результаты",
}
STATUS_TITLES = {"planning": "планирование", "in_progress": "в работе", "blocked": "заблокирована",
                 "done": "завершена"}


def _empty_items() -> dict[str, list[dict]]:
    return {f: [] for f in LIST_FIELDS}


@dataclass
class TaskState:
    task_id: int = 1
    created_at: str = field(default_factory=now)
    values: dict[str, dict] = field(default_factory=dict)         # скалярные поля → {value, kind, evidence, source_message}
    items: dict[str, list[dict]] = field(default_factory=_empty_items)
    history: list[dict] = field(default_factory=list)            # прежние значения скалярных полей

    @property
    def status(self) -> str:
        entry = self.values.get("status")
        return entry["value"] if entry else "planning"

    @property
    def active(self) -> bool:
        return self.status != "done"

    def value(self, fld: str) -> str | None:
        entry = self.values.get(fld)
        return entry["value"] if entry else None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: object) -> "TaskState":
        if not isinstance(d, dict):
            raise MemoryLoadError("working: запись задачи должна быть объектом")
        values, items = d.get("values") or {}, {**_empty_items(), **(d.get("items") or {})}
        if not isinstance(values, dict) or not all(isinstance(v, list) for v in items.values()):
            raise MemoryLoadError("working: повреждённые поля задачи")
        return cls(task_id=int(d.get("task_id", 1)), created_at=d.get("created_at") or now(), values=values,
                   items=items, history=list(d.get("history") or []))

    def public_view(self) -> dict:
        view = {"task_id": self.task_id, "status": self.status}
        view.update({f: self.value(f) for f in SCALAR_FIELDS if f != "status" and self.value(f)})
        view.update({f: [i["value"] for i in self.items[f]] for f in LIST_FIELDS if self.items[f]})
        return view


@dataclass
class WorkingState:
    task: TaskState | None = None
    archived: list[dict] = field(default_factory=list)
    version: int = 0
    journal: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"version": self.version, "task": self.task.to_dict() if self.task else None,
                "archived": self.archived, "journal": self.journal}

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkingState":
        try:
            task = TaskState.from_dict(payload["task"]) if payload.get("task") else None
            return cls(task=task, archived=list(payload.get("archived") or []),
                       version=int(payload.get("version", 0)), journal=list(payload.get("journal") or []))
        except (TypeError, ValueError) as exc:
            raise MemoryLoadError(f"working: повреждённая запись: {exc}") from exc


class WorkingMemory(MemoryLayer):
    name = "working"

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        super().__init__(path)
        self.state = WorkingState()
        self.load()

    # ----- хранилище -----

    def load(self) -> None:
        payload = self._read()
        if payload is not None:
            self.state = WorkingState.from_dict(payload)

    def reset_state(self) -> None:
        self.state = WorkingState()

    def to_dict(self) -> dict:
        return self.state.to_dict()

    @property
    def task(self) -> TaskState | None:
        return self.state.task

    # ----- запись -----

    def _next_task_id(self) -> int:
        ids = [a.get("task_id", 0) for a in self.state.archived] + ([self.task.task_id] if self.task else [])
        return max(ids, default=0) + 1

    def _ensure_task(self, fld: str, action: str) -> TaskState:
        task = self.state.task
        if task is None or (not task.active and fld == "goal" and action == "set"):
            if task is not None:
                self.state.archived.append({**task.to_dict(), "archived_at": now()})
            task = TaskState(task_id=self._next_task_id())
            self.state.task = task
        return task

    def _apply_one(self, op: dict, fld: str, value: str, message_no: int) -> str:
        action = op.get("op")
        task = self._ensure_task(fld, action)
        entry = {"value": value, "kind": op.get("kind") or "user",
                 "evidence": (op.get("evidence") or "").strip(), "source_message": message_no}
        if action == "delete":
            if fld in SCALAR_FIELDS:
                old = task.values.pop(fld)
                task.history.append({"field": fld, "old": old["value"], "new": None, "message": message_no,
                                     "reason": op.get("reason")})
            else:
                task.items[fld] = [i for i in task.items[fld] if not same_value(i["value"], value)]
            return "applied"
        if fld in SCALAR_FIELDS:
            old = task.values.get(fld)
            if old and old["value"] == value:
                return "unchanged"
            if old:
                task.history.append({"field": fld, "old": old["value"], "new": value, "message": message_no})
            task.values[fld] = entry
            return "applied"
        if any(same_value(i["value"], value) for i in task.items[fld]):
            return "unchanged"
        task.items[fld].append(entry)
        return "applied"

    def apply_operations(self, ops: list[dict], *, user_text: str, assistant_text: str | None,
                         message_no: int) -> dict:
        """Проверить и применить операции layer=working. → {"applied", "rejected", "unchanged"}."""
        out: dict[str, list] = {"applied": [], "rejected": [], "unchanged": []}
        for op in ops:
            fld, value, problem = check_working_op(op, self.task, user_text, assistant_text)
            rec = {**op, "key": fld, "value": value}
            if problem:
                out["rejected"].append({**rec, "rejection": problem})
                continue
            out[self._apply_one(op, fld, value, message_no)].append(rec)
        if ops:
            if out["applied"]:
                self.state.version += 1
            self.state.journal.append({"message": message_no, **out})
            self.save()
        return out

    def finish_task(self) -> bool:
        """Команда пользователя «/task done»: статус done. В долговременную память ничего не уходит."""
        task = self.task
        if task is None or not task.active:
            return False
        old = task.values.get("status")
        task.history.append({"field": "status", "old": old["value"] if old else None, "new": "done",
                             "message": None, "reason": "команда /task done"})
        task.values["status"] = {"value": "done", "kind": "user", "evidence": "/task done", "source_message": None}
        self.state.version += 1
        self.state.journal.append({"command": "task_done", "task_id": task.task_id})
        self.save()
        return True

    # ----- чтение -----

    def is_active(self) -> bool:
        return self.task is not None and self.task.active

    def public_view(self) -> dict:
        return self.task.public_view() if self.is_active() else {}

    def render(self) -> str:
        t = self.task
        if t is None:
            return ""
        lines = [f"Задача #{t.task_id}, статус: {STATUS_TITLES.get(t.status, t.status)} ({t.status})"]
        for f in ("goal", "deadline", "budget"):
            if t.value(f):
                lines.append(f"- {FIELD_TITLES[f]}: {t.value(f)}")
        for f in LIST_FIELDS:
            if t.items[f]:
                lines.append(f"- {FIELD_TITLES[f]}: " + "; ".join(i["value"] for i in t.items[f]))
        if t.value("next_step"):
            lines.append(f"- {FIELD_TITLES['next_step']}: {t.value('next_step')}")
        return "\n".join(lines)

    def prompt_contents(self) -> list[types.Content]:
        if not self.is_active():
            return []
        return [text_content("user", f"{config.WORKING_HEADER}\n{self.render()}"),
                text_content("model", config.WORKING_ACK)]

    def is_empty(self) -> bool:
        return self.task is None and not self.state.archived

    def stats(self) -> dict:
        t = self.task
        return {
            "has_task": t is not None,
            "task_id": t.task_id if t else None,
            "status": t.status if t else None,
            "active": self.is_active(),
            "filled_fields": [f for f in SCALAR_FIELDS if t and t.value(f)],
            "constraints": len(t.items["constraints"]) if t else 0,
            "results": len(t.items["results"]) if t else 0,
            "archived_tasks": len(self.state.archived),
            "version": self.state.version,
            "in_prompt": bool(self.prompt_contents()),
        }

    def dump(self) -> dict:
        return {"task": self.task.public_view() if self.task else None, "archived_tasks": len(self.state.archived)}
