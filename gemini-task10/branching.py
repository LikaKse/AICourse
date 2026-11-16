"""Checkpoint и независимые ветки диалога (для любой стратегии).

Структура состояния сессии:

    <root>/
      session.json                    ← стратегия, параметры, текущая ветка
      checkpoints/<id>/               ← НЕИЗМЕНЯЕМАЯ копия файлов стратегии + meta.json
      branches/<name>/                ← своя копия состояния ветки + branch.json
                                        (родительский checkpoint, изменения, журнал конфликтов)

Независимость обеспечивается тем, что у каждой ветки СВОЙ каталог: стратегия ветки читает
и пишет только его. Checkpoint копируется в ветку при создании и при restore, сам не меняется.

Конфликты при продолжении исходного диалога в ветке (сообщения после checkpoint написаны
без знания об изменении в ветке):

1. Sticky Facts — изменённые в ветке ключи помечаются locked; унаследованные сообщения
   (inherited_mode) не могут их перезаписать — операция попадает в журнал конфликтов,
   остаётся значение ветки (facts_store.apply_operations).
2. Любая стратегия — до продолжения каждое унаследованное сообщение проверяется на
   упоминание ПРЕЖНЕГО значения изменённого факта (conflict_patterns). Политика:
   "skip" — сообщение (и ответ на него) в ветку не добавляется, "error" — BranchConflictError.
   Найденные конфликты записываются в branch.json.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from google.genai import types

import config
from context_manager import content_text, text_content
from strategies import ContextStrategy, make_strategy
from utils import normalize_text, write_json

SESSION_FILE = "session.json"
BRANCH_META = "branch.json"
CHECKPOINT_META = "meta.json"
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
MAIN_BRANCH = "main"


class BranchError(Exception):
    """Неверное имя, несуществующая ветка или checkpoint."""


class BranchConflictError(BranchError):
    """Унаследованное сообщение противоречит изменению в ветке (политика "error")."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_name(name: str, what: str) -> str:
    if not NAME_RE.match(name or ""):
        raise BranchError(f"Недопустимое имя {what} {name!r}: латиница, цифры, _ и -, до 40 символов")
    return name


@dataclass
class BranchChange:
    """Изменение данных пользователя в ветке."""
    user: str                         # реплика пользователя с изменением
    model: str | None = None          # заранее написанный ответ (None — спросить модель через Agent)
    keys: tuple[str, ...] = ()        # ключи facts, которые меняются (станут locked)
    conflict_patterns: tuple[str, ...] = ()   # прежнее значение: его упоминание после checkpoint — конфликт


class BranchManager:
    def __init__(self, root: str | Path, *, strategy: str | None = None, strategy_kwargs: dict | None = None,
                 agent: Any = None) -> None:
        """Открыть существующую сессию или создать новую (тогда нужна strategy).

        agent — если задан, стратегия активной ветки всегда подключена к нему (use_strategy).
        """
        self.root = Path(root)
        self.agent = agent
        path = self.root / SESSION_FILE
        if path.is_file():
            self.session = _read(path)
            if strategy and strategy != self.session["strategy"]:
                raise BranchError(f"Сессия {self.root} создана для стратегии {self.session['strategy']!r}, "
                                  f"а не {strategy!r}")
        else:
            name = strategy or config.STRATEGY
            make_strategy(name)            # проверка имени
            self.session = {"format_version": 1, "strategy": name, "strategy_kwargs": strategy_kwargs or {},
                            "current": MAIN_BRANCH, "branches": [MAIN_BRANCH], "created_at": _now()}
            self._branch_dir(MAIN_BRANCH).mkdir(parents=True, exist_ok=True)
            self._write_branch_meta(MAIN_BRANCH, {"name": MAIN_BRANCH, "parent_checkpoint": None,
                                                  "created_at": _now(), "changes": [], "conflicts": []})
            self._save_session()
        self.strategy: ContextStrategy = self._activate(self._open(self.current))

    # ----- пути и метаданные -----

    @property
    def current(self) -> str:
        return self.session["current"]

    @property
    def strategy_name(self) -> str:
        return self.session["strategy"]

    def _branch_dir(self, name: str) -> Path:
        return self.root / "branches" / name

    def _checkpoint_dir(self, cp: str) -> Path:
        return self.root / "checkpoints" / cp

    def _save_session(self) -> None:
        write_json(self.root / SESSION_FILE, self.session)

    def _write_branch_meta(self, name: str, meta: dict) -> None:
        write_json(self._branch_dir(name) / BRANCH_META, meta)

    def branch_meta(self, name: str | None = None) -> dict:
        return _read(self._branch_dir(name or self.current) / BRANCH_META)

    def _open(self, name: str) -> ContextStrategy:
        return make_strategy(self.strategy_name, self._branch_dir(name), **self.session.get("strategy_kwargs", {}))

    def _activate(self, strategy: ContextStrategy) -> ContextStrategy:
        if self.agent is not None:
            self.agent.use_strategy(strategy)
        return strategy

    def attach(self, agent: Any) -> None:
        self.agent = agent
        self._activate(self.strategy)

    def branches(self) -> list[str]:
        return list(self.session["branches"])

    def checkpoints(self) -> list[dict]:
        base = self.root / "checkpoints"
        if not base.is_dir():
            return []
        return [_read(d / CHECKPOINT_META) for d in sorted(base.iterdir()) if (d / CHECKPOINT_META).is_file()]

    # ----- checkpoint -----

    def checkpoint(self, cp_id: str, *, note: str = "") -> dict:
        """Зафиксировать текущее состояние текущей ветки. Повторно с тем же id — ошибка."""
        _check_name(cp_id, "checkpoint")
        target = self._checkpoint_dir(cp_id)
        if target.exists():
            raise BranchError(f"Checkpoint {cp_id!r} уже существует (checkpoint неизменяем)")
        self.strategy.save()
        target.mkdir(parents=True)
        for f in self.strategy.state_files:
            src = self._branch_dir(self.current) / f
            if src.is_file():
                shutil.copy2(src, target / f)
        meta = {"id": cp_id, "strategy": self.strategy_name, "branch": self.current,
                "messages": len(self.strategy.full_history), "note": note, "created_at": _now()}
        write_json(target / CHECKPOINT_META, meta)
        return meta

    def checkpoint_meta(self, cp_id: str) -> dict:
        path = self._checkpoint_dir(cp_id) / CHECKPOINT_META
        if not path.is_file():
            raise BranchError(f"Checkpoint {cp_id!r} не найден")
        return _read(path)

    def _copy_checkpoint_to(self, cp_id: str, branch: str) -> None:
        self.checkpoint_meta(cp_id)
        target = self._branch_dir(branch)
        for f in self.strategy.state_files:          # убрать прежнее состояние ветки целиком
            (target / f).unlink(missing_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        for f in self.strategy.state_files:
            src = self._checkpoint_dir(cp_id) / f
            if src.is_file():
                shutil.copy2(src, target / f)

    # ----- ветки -----

    def create_branch(self, name: str, from_checkpoint: str, *, switch: bool = True) -> ContextStrategy:
        _check_name(name, "ветки")
        if name in self.session["branches"]:
            raise BranchError(f"Ветка {name!r} уже существует")
        self.strategy.save()
        self._copy_checkpoint_to(from_checkpoint, name)
        self._write_branch_meta(name, {"name": name, "parent_checkpoint": from_checkpoint, "created_at": _now(),
                                       "changes": [], "conflicts": []})
        self.session["branches"].append(name)
        self._save_session()
        if switch:
            return self.switch(name)
        return self._open(name)

    def switch(self, name: str) -> ContextStrategy:
        if name not in self.session["branches"]:
            raise BranchError(f"Ветки {name!r} нет; есть: {', '.join(self.session['branches'])}")
        self.strategy.save()
        self.session["current"] = name
        self._save_session()
        self.strategy = self._activate(self._open(name))
        return self.strategy

    def restore(self, cp_id: str, *, branch: str | None = None) -> ContextStrategy:
        """Вернуть ветку (по умолчанию текущую) к состоянию checkpoint. Сам checkpoint не меняется."""
        name = branch or self.current
        if name not in self.session["branches"]:
            raise BranchError(f"Ветки {name!r} нет")
        self._copy_checkpoint_to(cp_id, name)
        meta = self.branch_meta(name)
        meta.setdefault("restores", []).append({"checkpoint": cp_id, "at": _now(),
                                                "discarded_changes": meta.get("changes", [])})
        meta.update(parent_checkpoint=cp_id, changes=[], conflicts=[])
        self._write_branch_meta(name, meta)
        if name == self.current:
            self.strategy = self._activate(self._open(name))
            return self.strategy
        return self._open(name)

    # ----- работа с ветками -----

    def apply_change(self, change: BranchChange, agent: Any = None) -> list:
        """Внести изменение данных пользователя в ТЕКУЩУЮ ветку. Возвращает служебные вызовы."""
        agent = agent if agent is not None else self.agent
        strategy = self.strategy
        user = text_content("user", change.user)
        calls = []
        if change.model is None:
            if agent is None:
                raise BranchError("Для ответа модели нужен agent")
            agent.use_strategy(strategy)
            res = agent.ask(change.user)
            calls = res.service_calls
        else:
            if agent is not None:
                agent.use_strategy(strategy)
            calls += strategy.before_user_message(user)
            calls += strategy.add_exchange(user, text_content("model", change.model))
        locked: list[str] = []
        if hasattr(strategy, "facts"):
            # Блокируем ключи, которые РЕАЛЬНО изменило сообщение ветки (имя ключа выбирает модель),
            # и ключи, указанные явно.
            from facts_store import lock, normalize_key
            msg_no = len(strategy.full_history) - 1
            applied = [op["key"] for e in strategy.facts.journal if e.get("message") == msg_no
                       for op in e.get("applied", [])]
            locked = sorted({normalize_key(k) for k in [*applied, *change.keys] if normalize_key(k)})
            lock(strategy.facts, locked, f"изменено в ветке {self.current}")
            strategy.save()
        meta = self.branch_meta()
        meta["changes"].append({"user": change.user, "keys": list(change.keys), "locked_keys": locked,
                                "conflict_patterns": list(change.conflict_patterns),
                                "at_message": len(strategy.full_history) - 1, "at": _now()})
        self._write_branch_meta(self.current, meta)
        return calls

    def find_conflicts(self, messages: list[types.Content], start_number: int) -> list[dict]:
        """Унаследованные сообщения, где упомянуто прежнее значение изменённого в ветке факта."""
        patterns = [(c["user"], p) for c in self.branch_meta().get("changes", []) for p in c["conflict_patterns"]]
        found = []
        for i, msg in enumerate(messages):
            text = normalize_text(content_text(msg))
            for change, p in patterns:
                if re.search(p, text):
                    found.append({"message": start_number + i, "role": msg.role, "pattern": p,
                                  "change": change, "text": content_text(msg)})
        return found

    def continue_dialogue(self, messages: list[types.Content], *, start_number: int, agent: Any = None,
                          on_conflict: str = "skip") -> dict:
        """Продолжить ТЕКУЩУЮ ветку унаследованными сообщениями исходного диалога (пары user/model).

        start_number — номер первого сообщения в исходном диалоге (с 1), для журнала.
        on_conflict: "skip" — пропустить пару с конфликтом, "error" — остановиться.
        """
        if on_conflict not in ("skip", "error"):
            raise ValueError("on_conflict: skip | error")
        conflicts = self.find_conflicts(messages, start_number)
        if conflicts and on_conflict == "error":
            raise BranchConflictError(f"Конфликт в сообщении {conflicts[0]['message']}: "
                                      f"упомянуто прежнее значение ({conflicts[0]['pattern']})")
        bad_pairs = {(c["message"] - start_number) // 2 for c in conflicts}
        agent = agent if agent is not None else self.agent
        strategy = self.strategy
        if agent is not None:
            agent.use_strategy(strategy)
        calls, added, skipped = [], 0, []
        if hasattr(strategy, "inherited_mode"):
            strategy.inherited_mode = True
        try:
            for k in range(0, len(messages), 2):
                if k // 2 in bad_pairs:
                    skipped.append(start_number + k)
                    continue
                calls += strategy.before_user_message(messages[k])
                calls += strategy.add_exchange(messages[k], messages[k + 1])
                added += 2
        finally:
            if hasattr(strategy, "inherited_mode"):
                strategy.inherited_mode = False
                strategy.save()
        facts_conflicts = []
        if hasattr(strategy, "facts"):
            facts_conflicts = [{"message": e["message"], **c} for e in strategy.facts.journal
                               if e.get("inherited") for c in e.get("conflicts", [])]
        meta = self.branch_meta()
        meta["conflicts"] = meta.get("conflicts", []) + [
            {**c, "resolution": "сообщение пропущено"} for c in conflicts]
        meta["facts_conflicts"] = facts_conflicts
        meta.setdefault("continued", []).append({"from_message": start_number, "messages": len(messages),
                                                 "added": added, "skipped_pairs_from": skipped, "at": _now()})
        self._write_branch_meta(self.current, meta)
        return {"added": added, "skipped": skipped, "text_conflicts": conflicts,
                "facts_conflicts": facts_conflicts, "service_calls": calls}

    def describe(self) -> str:
        lines = [f"Стратегия: {self.strategy_name}; текущая ветка: {self.current}"]
        for b in self.session["branches"]:
            meta = self.branch_meta(b)
            mark = "*" if b == self.current else " "
            s = self.strategy if b == self.current else self._open(b)
            lines.append(f" {mark} {b}: от checkpoint {meta.get('parent_checkpoint') or '—'}, "
                         f"{len(s.full_history)} сообщ., изменений {len(meta.get('changes', []))}")
        for cp in self.checkpoints():
            lines.append(f"   checkpoint {cp['id']}: {cp['messages']} сообщ. (из ветки {cp['branch']})")
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Сценарий ветвления из data/branch_scenario.json
# ---------------------------------------------------------------------

def load_scenario(path: Path = config.BRANCH_SCENARIO_FILE) -> dict:
    data = _read(Path(path))
    for key in ("checkpoint_after", "branches", "questions"):
        if key not in data:
            raise BranchError(f"{path}: нет поля {key!r}")
    return data


def run_scenario(root: Path, strategy: str, dialogue: list[types.Content], agent: Any, scenario: dict,
                 *, on_conflict: str = "skip") -> dict:
    """Исходный диалог до checkpoint → checkpoint → ветки A и B со своими изменениями →
    продолжение исходного диалога в каждой ветке → вопросы в каждой ветке (remember=False)."""
    n = scenario["checkpoint_after"]
    if n % 2 or not 0 < n < len(dialogue):
        raise BranchError("checkpoint_after должен быть чётным и меньше длины диалога")
    if root.exists():
        shutil.rmtree(root)
    bm = BranchManager(root, strategy=strategy, agent=agent)
    for i in range(0, n, 2):
        bm.strategy.before_user_message(dialogue[i])
        bm.strategy.add_exchange(dialogue[i], dialogue[i + 1])
    cp = bm.checkpoint("cp", note=f"после сообщения {n} исходного диалога")
    out = {"checkpoint": cp, "branches": {}}
    for br in scenario["branches"]:
        bm.create_branch(br["name"], "cp")
        bm.apply_change(BranchChange(user=br["change"]["user"], model=br["change"].get("model"),
                                     keys=tuple(br["change"].get("keys", ())),
                                     conflict_patterns=tuple(br["change"].get("conflict_patterns", ()))), agent)
        cont = bm.continue_dialogue(dialogue[n:], start_number=n + 1, agent=agent, on_conflict=on_conflict)
        answers = []
        for q in scenario["questions"]:
            res = agent.ask(q["question"], remember=False)
            exp = q["expected"][br["name"]]
            answers.append({"id": q["id"], "question": q["question"], "answer": res.text,
                            "expected": exp, "correct": bool(re.search(exp, normalize_text(res.text)))})
        out["branches"][br["name"]] = {"continue": {k: v for k, v in cont.items() if k != "service_calls"},
                                       "answers": answers, "messages": len(bm.strategy.full_history)}
    out["manager"] = bm
    return out
