"""Отдельный персистентный профиль предпочтений пользователя (не слой памяти Дня 11)."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from memory.storage import now
from utils import atomic_write_text, redact_secrets

PROFILE_FIELDS = ("style", "length", "code", "format", "constraints")

# ---------------------------------------------------------------------
# Разбор сообщений: вопросы, отрицания и форматы ответа
# ---------------------------------------------------------------------

# Вопрос определяется по «?», по вопросительному слову в начале или по частице «ли».
# Исключения — устойчивые вводные обороты, которые начинаются с вопросительного слова,
# но являются утверждениями.
_QUESTION_START = re.compile(
    r"^(?:что|кто|как|какой|какая|какое|какие|почему|зачем|где|когда|сколько|чем|кем|куда|откуда|"
    r"разве|неужели)(?=\s|$)"
)
_QUESTION_EXCEPTIONS = re.compile(
    r"^(?:что\s+касается|что\s+до|что\s+ж|что\s+важно|как\s+правило|как\s+обычно|как\s+известно|"
    r"как\s+я\s+(?:уже\s+)?(?:говорил|сказал))\b"
)
# Границы смысловой части: знаки конца предложения и запятая перед новой командой/противопоставлением.
_CLAUSE_SPLIT = re.compile(
    r"[.!?;:\n]|,\s*(?=(?:а|но|однако|зато|лучше|отвечай|пиши|используй|показывай|давай)\b)"
)
# Прямая просьба избегать чего-либо: «не пиши», «не отвечай», «не надо», «без», ...
_IMPERATIVE_NEGATION = re.compile(
    r"\b(?:не\s+(?:пиши|пишите|отвечай|отвечайте|используй|используйте|показывай|показывайте|"
    r"надо|нужно|стоит|будь|будьте|добавляй|добавляйте|включай|включайте)|"
    r"перестань|перестаньте|хватит|убери|уберите|забудь|забудьте)\b"
)
# Личное отрицание предпочтения: «я не предпочитаю», «мне не нравится», ...
_STATEMENT_NEGATION = re.compile(
    r"\b(?:я\s+)?не\s+(?:предпочитаю|люблю|хочу|обожаю)|мне\s+не\s+нравится|\bне\s+нравится\b"
)
# Явная отмена уже сохранённой настройки (не путать с отрицанием в просьбе).
_CANCEL_RE = re.compile(r"\b(?:отменяю|отменяйте|больше\s+не|убери|уберите|забудь|забудьте)\b")
_CONSTRAINT_TRIGGER = re.compile(
    r"\b(?:предпочитаю|предпочтительно|всегда|обычно|по\s+умолчанию|в\s+дальнейшем|отныне|"
    r"не\s+предпочитаю|не\s+люблю|не\s+использ\w+|не\s+(?:отвечай|пиши|показывай)|"
    r"без|отменяю|отменяйте|больше\s+не|не\s+нужно|не\s+надо|убери|уберите|забудь|забудьте)\b"
)
_CONSTRAINT_TERMS = ("эмодзи", "смайл", "таблиц", "спис", "markdown", "json", "заголов", "сокращен", "жаргон")
_FORMAT_TERMS = {"таблиц", "спис", "markdown", "json"}
# Сообщение должно содержать действие/пожелание, а не просто описание («это короткий ответ»).
_EXPLICIT_CUE = re.compile(
    r"\b(?:отвечай|отвечайте|пиши|пишите|показыва\w*|приводи\w*|используй|используйте|"
    r"давай|давайте|предпочитаю|предпочтительно|люблю|нравится|хочу|лучше|"
    r"всегда|обычно|по\s+умолчанию|в\s+дальнейшем|отныне|никогда|без|"
    r"не\s+(?:надо|нужно|пиши|отвечай|используй|показывай|будь)|"
    r"отменяю|отменяйте|больше\s+не|убери|уберите|забудь|забудьте|в\s+формате|формат|на\s+ты)\b"
)
_FORMAT_CUE = re.compile(
    r"\b(?:в\s+формате|формат\w*\s+(?:ответа|ответ)|отвечай|отвечайте|пиши|пишите|"
    r"предпочитаю|предпочтительно|использ\w*|выводи|выводите|оформляй|оформляйте|нравится)\b"
)
_FORMAT_VALUES = (
    (re.compile(r"\bсписк\w*|\bперечислен\w*"), "списком"),
    (re.compile(r"\bтаблиц\w*"), "таблицей"),
    (re.compile(r"\bmarkdown\b"), "markdown"),
    (re.compile(r"\bjson\b"), "json"),
)


def _is_question(text: str, low: str) -> bool:
    if "?" in text:
        return True
    if _QUESTION_EXCEPTIONS.match(low):
        return False
    if re.search(r"\bли\b", low):
        return True
    return bool(_QUESTION_START.match(low))


def _clause_prefix(low: str, pos: int) -> str:
    return _CLAUSE_SPLIT.split(low[:pos])[-1]


def _negation_kind(low: str, pos: int) -> str | None:
    """Тип отрицания перед найденным словом: imperative / statement / generic / None."""
    prefix = _clause_prefix(low, pos)
    if _IMPERATIVE_NEGATION.search(prefix) or re.search(r"\bбез(?:\s+\S+)?\s*$", prefix):
        return "imperative"
    if _STATEMENT_NEGATION.search(prefix):
        return "statement"
    if re.search(r"\bне\b", prefix):
        return "generic"
    return None


def _same_value(left, right) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return left.strip().rstrip(".!?").lower() == right.strip().rstrip(".!?").lower()
    return left == right


def _contains_value(items, value) -> bool:
    if not isinstance(items, list):
        items = [items]
    return any(_same_value(item, value) for item in items)


def _imperative_negated_format(low: str):
    """Канонический формат, который просят не использовать («не используй списки», «без таблиц»)."""
    for regex, value in _FORMAT_VALUES:
        for match in regex.finditer(low):
            if _negation_kind(low, match.start()) in ("imperative", "generic"):
                return value
    return None


def _format_match(low: str):
    """(каноническое значение формата, совпадение) или (None, совпадение).

    None при отсутствии формата или если все найденные форматы отрицаются
    («не используй списки»): такую просьбу обрабатывает ветка constraints.
    """
    if not _FORMAT_CUE.search(low):
        return None, None
    candidates = []
    for regex, value in _FORMAT_VALUES:
        for match in regex.finditer(low):
            candidates.append((match.start(), value, match))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0])
    for _, value, match in candidates:
        if _negation_kind(low, match.start()) is None:
            return value, match
    return None, candidates[0][2]


@dataclass
class Preference:
    value: str | list[str]
    source: str
    confidence: float
    updated_at: str = field(default_factory=now)
    evidence: str = ""


@dataclass
class UserProfile:
    preferences: dict[str, Preference] = field(default_factory=dict)
    revision: int = 0

    def to_dict(self) -> dict:
        return {"preferences": {k: asdict(v) for k, v in self.preferences.items()}, "revision": self.revision}

    def render(self) -> str:
        labels = {"style": "Стиль общения", "length": "Длина ответа", "code": "Код",
                  "format": "Формат ответа", "constraints": "Дополнительные ограничения и предпочтения"}
        rows = [f"{labels[k]}: {', '.join(p.value) if isinstance(p.value, list) else p.value} "
                f"(уверенность {p.confidence:.2f})" for k, p in self.preferences.items()]
        return "Профиль пользователя (учитывай вместе с системными инструкциями и остальной памятью):\n" + "\n".join(rows) if rows else ""


class ProfileStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.profile = UserProfile()
        self.pending: dict[str, dict] = {}
        self.load()

    def load(self):
        if self.path and self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.profile = UserProfile({k: Preference(**v) for k, v in raw.get("preferences", {}).items()}, raw.get("revision", 0))
            self.pending = raw.get("pending", {})
        elif self.path is not None:
            # A missing file means an externally cleared profile; never keep stale in-memory values.
            self.profile = UserProfile()
            self.pending = {}

    def save(self):
        if self.path:
            payload = json.dumps({**self.profile.to_dict(), "pending": self.pending}, ensure_ascii=False, indent=2)
            atomic_write_text(self.path, redact_secrets(payload))

    def update(self, key: str, value: str, *, source: str, confidence: float, evidence: str):
        if key not in PROFILE_FIELDS or confidence < .8:
            return False
        if key == "constraints":
            incoming = value if isinstance(value, list) else [value]
            existing = self.profile.preferences.get(key)
            prior = existing.value if existing else []
            prior = prior if isinstance(prior, list) else [prior]
            new_items = []
            for item in incoming:
                if not _contains_value(prior, item) and not _contains_value(new_items, item):
                    new_items.append(item)
            if not new_items:
                return True
            value = [*prior, *new_items]
            if existing:
                source = ",".join(dict.fromkeys([*existing.source.split(","), *source.split(",")]))
                confidence = min(existing.confidence, confidence)
                evidence = "\n".join(dict.fromkeys([*existing.evidence.split("\n"), evidence]))
        self.profile.preferences[key] = Preference(value, source, confidence, evidence=evidence)
        self.profile.revision += 1
        self.save()
        return True

    def cancel(self, key: str, *, source: str, evidence: str):
        if key in self.profile.preferences:
            del self.profile.preferences[key]
            self.profile.revision += 1
            self.save()
            return True
        return False

    def cancel_constraint(self, phrase: str, *, source: str, evidence: str):
        current = self.profile.preferences.get("constraints")
        if not current:
            return False
        items = current.value if isinstance(current.value, list) else [current.value]
        terms = [term for term in _CONSTRAINT_TERMS if term in phrase.lower()]
        remaining = [item for item in items if not any(term in item.lower() for term in terms)]
        if len(remaining) == len(items):
            return False
        if remaining:
            self.profile.preferences["constraints"] = Preference(remaining, current.source, current.confidence,
                                                                  evidence=current.evidence)
            self.profile.revision += 1
            self.save()
        else:
            self.cancel("constraints", source=source, evidence=evidence)
        return True

    def clear(self):
        self.profile = UserProfile()
        self.pending.clear()
        self.save()

    def observe(self, text: str, message_no: int):
        """Записывает только явные, устойчивые формулировки; разовые запросы остаются локальными."""
        low = text.lower()
        # Вопрос — не предпочтение: не меняем профиль и не создаём кандидатов.
        if _is_question(text, low):
            return []
        persistent = bool(re.search(
            r"\b(всегда|обычно|предпочитаю|предпочтительно|мне нравится|в дальнейшем|по умолчанию|отныне|никогда)\b",
            low))
        one_off = bool(re.search(r"\b(сейчас|для этого ответа|в этот раз|только сегодня|на этот вопрос)\b", low))
        if one_off and not persistent:
            return []
        if not (persistent or _EXPLICIT_CUE.search(low)):
            return []

        found: list[dict] = []
        structured: set[str] = set()

        def consider(key: str, value: str):
            source = f"user_message:{message_no}"
            existing = self.profile.preferences.get(key)
            if key == "constraints":
                prior = existing.value if existing else []
                prior = prior if isinstance(prior, list) else ([prior] if existing else [])
                if existing and _contains_value(prior, value):
                    return
            elif existing and _same_value(existing.value, value):
                if key != "constraints":
                    structured.add(key)
                return
            candidate = self.pending.get(key)
            if persistent:
                confidence = .95
            elif candidate and _same_value(candidate.get("value"), value):
                confidence = .85
                source = f"{candidate['source']},{source}"
            else:
                self.pending[key] = {"value": value, "source": source, "evidence": text, "confidence": .55}
                self.save()
                if key != "constraints":
                    structured.add(key)
                found.append({"category": key, "action": "candidate", "confidence": .55})
                return
            self.pending.pop(key, None)
            self.update(key, value, source=source, confidence=confidence, evidence=text)
            if key != "constraints":
                structured.add(key)
            found.append({"category": key, "action": "set", "confidence": confidence})

        def cancel_key(key: str):
            had_pending = self.pending.pop(key, None) is not None
            removed = self.cancel(key, source=f"user_message:{message_no}", evidence=text)
            if had_pending and not removed:
                self.save()
            if removed or had_pending:
                found.append({"category": key, "action": "cancel"})

        patterns = (
            ("style", r"(?:неформальн\w*|дружеск\w*|\bна ты\b)", "неформальный",
             r"(?<!не)формальн\w*", "формальный"),
            ("length", r"(?:кратк\w*|коротк\w*|сжато|лаконичн\w*)", "кратко",
             r"(?:подробн\w*|развёрнуто|детально)", "подробно"),
            ("code", r"(?:показыва(?:й|йте|ть|ют|ет|ете|ешь|ем)|приводи\w*).{0,20}код", "показывать",
             r"не\s+показыва(?:й|йте|ть).{0,20}код|без\s+кода", "не показывать"),
        )
        for key, yes, yes_val, no, no_val in patterns:
            match = re.search(no, low) if no else None
            side = "no"
            if match is None:
                match = re.search(yes, low)
                side = "yes"
            if match is None:
                continue

            value = no_val if side == "no" else yes_val
            if _CANCEL_RE.search(low):
                # Явная отмена сильнее любой полярности: «отменяю», «больше не», «убери».
                cancel_key(key)
                continue

            negation = _negation_kind(low, match.start())
            if negation == "imperative":
                # «Не пиши формально» → неформальный; «не отвечай кратко» → подробно.
                # Противоположную настройку снимаем сразу, не дожидаясь повтора.
                existing = self.profile.preferences.get(key)
                if existing and _same_value(existing.value, value):
                    cancel_key(key)
                value = yes_val if side == "no" else no_val
            elif negation is not None:
                # «Я не предпочитаю краткие ответы» — не сохраняем противоположность.
                # Если такая настройка уже была, снимаем именно её, а не любую другую.
                existing = self.profile.preferences.get(key)
                if existing and _same_value(existing.value, value):
                    cancel_key(key)
                continue
            consider(key, value)

        format_value, _ = _format_match(low)
        if _CANCEL_RE.search(low) and any(term in low for term in _FORMAT_TERMS):
            # «Убери списки» / «отменяю формат таблицей» — снять текущий формат.
            cancel_key("format")
        elif format_value is not None:
            consider("format", format_value)
        else:
            negated_format = _imperative_negated_format(low)
            existing_format = self.profile.preferences.get("format")
            if negated_format and existing_format and _same_value(existing_format.value, negated_format):
                # «Не используй списки» при уже выбранном формате списков.
                cancel_key("format")

        terms = [term for term in _CONSTRAINT_TERMS if term in low]
        if terms and _CONSTRAINT_TRIGGER.search(low):
            if _CANCEL_RE.search(low):
                if self.cancel_constraint(low, source=f"user_message:{message_no}", evidence=text):
                    found.append({"category": "constraints", "action": "cancel"})
            else:
                # «Предпочитаю ответы списком» — это format, а не ещё и constraint.
                only_format = all(term in _FORMAT_TERMS for term in terms)
                if not (only_format and "format" in structured):
                    consider("constraints", text.strip())
        return found
