"""Явный CLI управления инвариантами.

Все изменения проходят через ``InvariantCommandService`` и ``InvariantStore``:
проверки моделей, версии, аудит, журнал и атомарность остаются прежними.
Ассистент этот модуль не импортирует.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from invariants.commands import InvariantCommandService
from invariants.models import (
    InvariantCategory,
    InvariantChange,
    InvariantCheck,
    InvariantRule,
)
from invariants.store import InvariantStore

DEFAULT_STATE_DIR = "invariants_state"


class CliError(Exception):
    """Ожидаемая ошибка CLI с понятным сообщением."""


def _default_actor() -> str:
    return os.environ.get("USER") or os.environ.get("USERNAME") or "local-operator"


def _default_state_dir() -> str:
    return os.environ.get("INVARIANTS_STATE_DIR", DEFAULT_STATE_DIR)


def _require_reason(value: str | None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CliError("причина изменения обязательна (--reason)")
    return value.strip()


def _require_actor(value: str | None) -> str:
    actor = value if value is not None else _default_actor()
    if not isinstance(actor, str) or not actor.strip():
        raise CliError("actor обязателен и не должен быть пустым")
    return actor.strip()


def _parse_checks(raw_checks: list[str]) -> tuple[InvariantCheck, ...]:
    if not raw_checks:
        raise CliError("нужна хотя бы одна проверка (--check JSON)")
    checks: list[InvariantCheck] = []
    for raw in raw_checks:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CliError(f"некорректный JSON проверки: {exc}") from exc
        if not isinstance(data, dict):
            raise CliError("каждая проверка должна быть JSON-объектом")
        try:
            checks.append(InvariantCheck.from_dict(data))
        except (TypeError, ValueError) as exc:
            raise CliError(f"некорректная проверка: {exc}") from exc
    return tuple(checks)


def _category(value: str) -> InvariantCategory:
    try:
        return InvariantCategory(value)
    except ValueError as exc:
        raise CliError(f"неизвестная категория {value!r}") from exc


def _format_checks(rule: InvariantRule) -> str:
    parts: list[str] = []
    for check in rule.checks:
        text = f"{check.kind}={check.value!r}"
        if check.key is not None:
            text += f" key={check.key!r}"
        if check.case_sensitive:
            text += " case_sensitive=true"
        parts.append(text)
    return ", ".join(parts) if parts else "-"


def _print_rules(snapshot, as_json: bool) -> None:
    if as_json:
        print(json.dumps(
            snapshot.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(f"revision: {snapshot.revision}")
    if not snapshot.rules:
        print("Инварианты отсутствуют.")
        return
    for rule in snapshot.rules:
        state = "active" if rule.active else "disabled"
        print(
            f"- {rule.id} | {rule.category.value} | v{rule.version} | "
            f"{state} | {rule.text}")
        print(f"  checks: {_format_checks(rule)}")


def _print_history(changes, as_json: bool) -> None:
    if as_json:
        print(json.dumps(
            [change.to_dict() for change in changes],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ))
        return
    if not changes:
        print("История изменений пуста.")
        return
    for change in changes:
        print(
            f"{change.store_revision} | {change.action} | "
            f"{change.invariant_id} | {change.actor} | {change.at} | "
            f"{change.reason}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="invariant_cli.py",
        description="Явное управление инвариантами Day 14",
    )
    parser.add_argument(
        "--state-dir",
        default=_default_state_dir(),
        help="каталог хранилища инвариантов "
             "(по умолчанию INVARIANTS_STATE_DIR или ./invariants_state)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="показать все инварианты")
    list_parser.add_argument("--json", action="store_true", help="вывод JSON")

    add_parser = sub.add_parser("add", help="добавить инвариант")
    add_parser.add_argument(
        "--id", required=True, help="ID нового правила")
    add_parser.add_argument(
        "--category",
        required=True,
        choices=sorted(item.value for item in InvariantCategory),
        help="категория правила",
    )
    add_parser.add_argument(
        "--text", required=True, help="текст правила")
    add_parser.add_argument(
        "--check",
        action="append",
        default=[],
        metavar="JSON",
        help="JSON-объект проверки; можно повторять; нужна хотя бы одна",
    )
    add_parser.add_argument(
        "--actor", default=None,
        help="кто изменяет (по умолчанию системный пользователь)")
    add_parser.add_argument(
        "--reason", default=None, help="причина изменения (обязательна)")

    update_parser = sub.add_parser("update", help="изменить инвариант")
    update_parser.add_argument(
        "--id", required=True,
        help="ID существующего правила; изменить ID нельзя")
    update_parser.add_argument(
        "--version", required=True, type=int,
        help="ожидаемая текущая версия правила (обязательна)")
    update_parser.add_argument(
        "--category",
        default=None,
        choices=sorted(item.value for item in InvariantCategory),
        help="новая категория (если не указана, сохраняется текущая)",
    )
    update_parser.add_argument(
        "--text", required=True, help="новый текст правила")
    update_parser.add_argument(
        "--check",
        action="append",
        default=[],
        metavar="JSON",
        help="JSON-объект проверки; можно повторять; нужна хотя бы одна",
    )
    update_parser.add_argument(
        "--actor", default=None,
        help="кто изменяет (по умолчанию системный пользователь)")
    update_parser.add_argument(
        "--reason", default=None, help="причина изменения (обязательна)")

    disable_parser = sub.add_parser("disable", help="отключить инвариант")
    disable_parser.add_argument(
        "--id", required=True, help="ID существующего правила")
    disable_parser.add_argument(
        "--version", required=True, type=int,
        help="ожидаемая текущая версия правила (обязательна)")
    disable_parser.add_argument(
        "--actor", default=None,
        help="кто изменяет (по умолчанию системный пользователь)")
    disable_parser.add_argument(
        "--reason", default=None, help="причина изменения (обязательна)")

    history_parser = sub.add_parser("history", help="показать историю изменений")
    history_parser.add_argument("--json", action="store_true", help="вывод JSON")

    return parser


def _read_validated_history(
    store: InvariantStore,
) -> tuple[InvariantChange, ...]:
    """Проверить snapshot и прочитать историю в одной критической секции.

    ``load()`` и ``read_history()`` берут блокировку по отдельности, поэтому
    между ними возможна гонка. Здесь используется ``locked()`` и unlocked-
    методы store: сначала проверяется snapshot/история, затем читается
    история без шанса конкурентного изменения.
    """
    with store.locked():
        store._load_unlocked()
        return store._read_history_unlocked()


def _dispatch(args: argparse.Namespace) -> int:
    store = InvariantStore(args.state_dir)

    if args.command == "list":
        snapshot = store.load()
        _print_rules(snapshot, args.json)
        return 0

    if args.command == "history":
        changes = _read_validated_history(store)
        _print_history(changes, args.json)
        return 0

    service = InvariantCommandService(store)

    if args.command == "add":
        reason = _require_reason(args.reason)
        actor = _require_actor(args.actor)
        checks = _parse_checks(args.check)
        rule = InvariantRule(
            id=args.id,
            category=_category(args.category),
            text=args.text,
            checks=checks,
        )
        change = service.add(rule, actor=actor, reason=reason)
        print(
            f"Добавлен инвариант {change.invariant_id!r} "
            f"(revision {change.store_revision}).")
        return 0

    if args.command == "update":
        reason = _require_reason(args.reason)
        actor = _require_actor(args.actor)
        checks = _parse_checks(args.check)
        change = service.update(
            args.id,
            text=args.text,
            checks=checks,
            actor=actor,
            reason=reason,
            expected_version=args.version,
            category=(
                _category(args.category)
                if args.category is not None else None
            ),
        )
        print(
            f"Обновлён инвариант {change.invariant_id!r} до версии "
            f"{change.new['version']} (revision {change.store_revision}).")
        return 0

    if args.command == "disable":
        reason = _require_reason(args.reason)
        actor = _require_actor(args.actor)
        change = service.disable(
            args.id,
            actor=actor,
            reason=reason,
            expected_version=args.version,
        )
        print(
            f"Отключён инвариант {change.invariant_id!r} "
            f"(revision {change.store_revision}).")
        return 0

    raise CliError(f"неизвестная команда {args.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except CliError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Не показываем traceback для ожидаемых ошибок store/model/IO.
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "build_parser"]
