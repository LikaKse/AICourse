"""Детерминированный программный валидатор инвариантов.

Валидатор не вызывает LLM и не изменяет snapshot/историю. Ошибки загрузки
или проверки приводят к исключению ``InvariantValidationError``: вызывающий
код обязан отказать в ответе (fail-closed), а не продолжать как при PASS.
"""
from __future__ import annotations

import ast
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .errors import InvariantStoreError, InvariantValidationError
from .models import (
    CHECK_KINDS,
    InvariantCheck,
    InvariantRule,
    InvariantSnapshot,
    Violation,
)
from .store import InvariantStore

_CODE_FENCE_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<code>.*?)```", re.S)
_DIRECT_IMPORT_RE = re.compile(
    r"(?m)^[ \t]*(?:"
    r"from[ \t]+[A-Za-z_][A-Za-z0-9_.]*[ \t]+import[ \t]+[^\n]+|"
    r"import[ \t]+[^\n]+"
    r")[ \t]*(?:#.*)?$")
_DYNAMIC_IMPORT_LINE_RE = re.compile(
    r"(?m)^[ \t]*(?:[A-Za-z_][A-Za-z0-9_.]*\.)?"
    r"(?:__import__|import_module)[ \t]*\(")
_PYTHON_HINT_RE = re.compile(
    r"(?m)^[ \t]*(?:from[ \t]+|import[ \t]+|exec[ \t]*\(|eval[ \t]*\(|"
    r"(?:[A-Za-z_][A-Za-z0-9_.]*\.)?(?:__import__|import_module)[ \t]*\()"
)
_SAFE_PREFIX_RE = re.compile(
    r"(?:\b(?:do\s+not|don't|never|not)\s+"
    r"(?:use|using|recommend|permit|allow|choose|adopt)\s+"
    r"|\b(?:cannot|can't|should\s+not|must\s+not)\s+"
    r"(?:use|using|recommend|permit|allow|choose|adopt)\s+"
    r"|без|without|вместо|instead\s+of|rather\s+than|"
    r"нельзя(?:\s+(?:использовать|применять))?|"
    r"не\s+(?:используй|использовать|применяй|применять|"
    r"рекомендуй|допускай|используем|применяем|используется|"
    r"применяется|рекомендуется|допускается)|"
    r"no)\s*$",
    re.I,
)
_EPISTEMIC_RE = re.compile(
    r"\b(?:think|believe|sure|suppose|guess|claim|say|doubt|"
    r"perhaps|maybe|probably|true|false|incorrect|wrong)\b",
    re.I,
)
_SAFE_POST_RE = re.compile(
    r"^\s*(?:is\s+)?(?:prohibited|forbidden|not\s+allowed|"
    r"not\s+recommended|not\s+used|not\s+a\s+good|not\s+suitable|"
    r"запрещ|недопустим|не\s+используется|не\s+подходит|"
    r"не\s+рекомендуется|не\s+применяется)",
    re.I,
)
#: Отрицательные глаголы/обороты, которые делают упоминание безопасным.
#: Они не являются «разрешением по слову»: после них всё равно проверяется
#: остаток текста на положительную инструкцию.
_SAFE_NEGATIVE_PREFIX_RE = re.compile(
    r"(?:\b(?:avoid|avoids|avoiding|forbid|forbids|forbidding|"
    r"prohibit|prohibits|prohibiting|refrain\s+from|"
    r"steer\s+clear\s+of|stay\s+away\s+from)\b[^.!?;]*|"
    r"\b(?:избегай|избегать|избегая|запрещ\w*|"
    r"не\s+допуска\w*)\b[^.!?;]*)\s*$",
    re.I,
)
_SAFE_NEGATIVE_POST_RE = re.compile(
    r"^\s*(?:(?:is|are|was|were|has\s+been|have\s+been)\s+)?"
    r"(?:not\s+(?:used|allowed|recommended|permitted|supported)|"
    r"(?:should|must)\s+not\s+be\s+used|"
    r"is\s+not\s+to\s+be\s+used|"
    r"запрещ\w*|не\s+используется|не\s+допускается|"
    r"не\s+рекомендуется|не\s+должен\s+использоваться)",
    re.I,
)
_FORCE_UNSAFE_RE = re.compile(
    r"\b(?:ignore\s+(?:all\s+)?(?:previous|prior)|"
    r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|system)\s+"
    r"(?:instructions?|rules?|prompts?)|"
    r"forget\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|system)\s+"
    r"(?:instructions?|rules?|prompts?)|"
    r"system\s+prompt|"
    r"developer\s+mode|act\s+as|pretend|role[- ]?play|"
    r"make\s+(?:an\s+)?exception|only\s+once|one\s+time|"
    r"игнорируй\w*|проигнорируй\w*|забудь\w*|"
    r"представь(?:\s+себя)?|ты\s+теперь|"
    r"сделай\s+исключение|только\s+один\s+раз)\b",
    re.I,
)
#: Явное объяснение/цитата опасной фразы не является попыткой обхода.
_FORCE_UNSAFE_EXPLANATION_RE = re.compile(
    r"\b(?:phrase|example|prompt\s+injection|known\s+attack\s+pattern|"
    r"attack\s+pattern|means|refers\s+to|"
    r"is\s+(?:a|an|the)\s+(?:technique|example|phrase|term|concept|"
    r"verb|word)|"
    r"фраз\w*|пример\w*|означа\w*|термин\w*|инъекц\w*)\b",
    re.I,
)
_FORCE_UNSAFE_DEFINITION_RE = re.compile(
    r"^\s*(?:role[- ]?play|pretend|act\s+as|system\s+prompt|"
    r"developer\s+mode|make\s+(?:an\s+)?exception|only\s+once|one\s+time|"
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?|"
    r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|system)\s+"
    r"(?:instructions?|rules?|prompts?)|"
    r"forget\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|system)\s+"
    r"(?:instructions?|rules?|prompts?))\s+"
    r"(?:is|are)\s+(?:a|an|the)\s+\w+",
    re.I,
)
#: Отрицание непосредственно перед опасной фразой делает её запретом,
#: а не попыткой. В отличие от _NEGATION_RE не срабатывает на «No, ...».
_FORCE_UNSAFE_NEGATION_RE = re.compile(
    r"\b(?:do\s+not|don't|never|not|cannot|can't|should\s+not|"
    r"must\s+not|no|не|нельзя|без)\s*$",
    re.I,
)
_EXPLANATION_START_RE = re.compile(
    r"^\s*(?:explain|why|what|is\s+it|can\s+you\s+explain|"
    r"could\s+you\s+explain|tell\s+me\s+about|describe|"
    r"почему|объясни(?:те)?|что|можно\s+ли|опиши(?:те)?|"
    r"расскажи(?:те)?)\b",
    re.I,
)
_ANALYSIS_START_RE = re.compile(
    r"^\s*(?:analysis|analyze|проанализ\w*|анализ\w*)", re.I)
_ANALYSIS_NEGATIVE_RE = re.compile(
    r"(?:\b(?:if|если)\b.*\b(?:prohibited|forbidden|not\s+allowed|"
    r"нельзя|запрещ\w*|недопустим\w*|наруш\w*|потенциальн\w*|"
    r"risk|риск\w*|проблем\w*)|\b(?:prohibited|forbidden|нельзя|"
    r"запрещ\w*|недопустим\w*|наруш\w*|потенциальн\w*|risk|риск\w*|"
    r"проблем\w*)\b.*\b(?:if|если)\b)",
    re.I,
)
_HYPOTHETICAL_START_RE = re.compile(
    r"^\s*(?:for\s+example|example|например)\b", re.I)
_HYPOTHETICAL_NEGATIVE_RE = re.compile(
    r"\b(?:if|если)\b.*\b(?:prohibited|forbidden|not\s+allowed|"
    r"нельзя|запрещ\w*|недопустим\w*|наруш\w*|потенциальн\w*|"
    r"risk|риск\w*|проблем\w*|fail\w*|ошиб\w*)",
    re.I,
)
_UNSAFE_STRUCTURE_RE = re.compile(
    r"(?:[,;:](?!\s*(?:because|потому\s+что|так\s+как|"
    r"since|поскольку))|[-–—/|()]|\b(?:and|then|but|но|и|а|однако)\b)",
    re.I,
)
#: Для объяснений список/тире/«and» сами по себе не инструкция.
#: Двоеточие и точка с запятой остаются небезопасными: они вводят
#: утверждение, которое может быть рекомендацией.
_EXPLANATION_STRUCTURE_RE = re.compile(r"[:;]")
#: Глаголы, которые в положительном контексте означают инструкцию
#: использовать запрещённую технологию. Список нужен только для поиска
#: таких инструкций; безопасность упоминания им не определяется.
_IMPERATIVE_VERBS = (
    r"use|using|run|deploy|install|adopt|choose|switch|migrate|move|"
    r"containerize|apply|enable|start|go\s+with|stick\s+with|"
    r"rely\s+on|build\s+with|work\s+with|recommend|"
    r"используй\w*|использовать|примен\w*|выбер\w*|выбрать|"
    r"установ\w*|запуст\w*|разверн\w*|мигрир\w*|перейд\w*|"
    r"перевести|переведи\w*|подключ\w*|замени\w*|рекоменд\w*"
)
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|without|don't|doesn't|didn't|cannot|can't|"
    r"won't|wouldn't|shouldn't|couldn't|mustn't|"
    r"не|нельзя|без)\b",
    re.I,
)
_CONDITIONAL_RE = re.compile(
    r"\b(?:if|when|whether|если|когда|ли)\b", re.I)
_NEGATIVE_OUTCOME_RE = re.compile(
    r"\b(?:prohibited|forbidden|not\s+allowed|not\s+recommended|"
    r"not\s+a\s+good|not\s+suitable|bad\s+choice|"
    r"нельзя|запрещ\w*|недопустим\w*|наруш\w*|"
    r"потенциальн\w*|risk|риск\w*|проблем\w*|ошиб\w*)\b",
    re.I,
)
_CLAUSE_CONJUNCTION_RE = re.compile(
    r"\b(?:and|but|then|however|но|затем|однако|и|а)\b", re.I)
_POSITIVE_PRONOUN_STATEMENT_RE = re.compile(
    r"\b(?:and|but|then|но|затем|и|а)\s+"
    r"(?:it|this|это|он|она)\s+(?:is|are|was|were)\s+(?:the\s+)?"
    r"(?:best|better|selected|recommended|preferred|required|chosen|"
    r"the\s+architecture|a\s+good\s+choice)\b",
    re.I,
)
#: Короткое положительное предложение с местоимением-подлежащим.
#: Применяется только к предложению, у которого предыдущее предложение
#: содержит проверяемый термин, поэтому любое «it» ссылкой не считается.
_ANAPHORA_INTRO = (
    r"(?:(?:in\s+fact|indeed|actually|moreover|furthermore|also|"
    r"besides|therefore|thus|so|например|на\s+самом\s+деле|"
    r"фактически|более\s+того|кроме\s+того)\s*,?\s*)?"
)
_ANAPHORA_ADVERB = (
    r"(?:clearly|strongly|highly|really|very|also|still|definitely|"
    r"absolutely|particularly|especially|generally|widely|commonly|"
    r"often|usually|typically|probably|perhaps|maybe|"
    r"явно|особенно|широко|обычно|как\s+правило)\s+"
)
_ANAPHORA_PREDICATE = (
    r"(?:the\s+)?(?:best(?:\s+(?:option|choice|solution))?|only\s+way|"
    r"recommended|mandatory|standard|preferred|way\s+to\s+go|"
    r"a\s+good\s+choice|better|"
    r"лучш\w*(?:\s+вариант\w*)?|единственн\w*(?:\s+способ\w*)?|"
    r"обязательн\w*|стандарт\w*|предпочтительн\w*|рекомендуется)"
)
_POSITIVE_ANAPHORA_RE = re.compile(
    r"^" + _ANAPHORA_INTRO +
    r"(?:"
    r"(?:it(?:'s|\s+is)|this\s+is|that\s+is|он|она|оно)\s+"
    r"(?:" + _ANAPHORA_ADVERB + r")*"
    + _ANAPHORA_PREDICATE +
    r"|"
    r"(?:it|he|she|он|она|оно)\s+"
    r"(?:should|must|ought\s+to|needs?\s+to)\s+be\s+used"
    r")"
    r"\s*[.!?]?\s*$",
    re.I,
)
#: «Everyone recommends it» — тоже рекомендация, но с третьим лицом.
#: Проверяется только вместе с антецедентом в предыдущем предложении.
_POSITIVE_RECOMMENDS_IT_RE = re.compile(
    r"^\s*(?:everyone|everybody|people|they|the\s+team|"
    r"most\s+people|many\s+people|experts?)\s+"
    r"recommends?\s+it\s*[.!?]?\s*$",
    re.I,
)
_MARKDOWN_SPAN_RE = re.compile(
    r"\[[^\]]*\]\([^)]*\)|\*\*[^*]+\*\*|__[^_]+__|`[^`]+`")
_NEXT_SAFE_START_RE = re.compile(
    r"^\s*(?:because|потому|так\s+как|since|it\s+is|this\s+is|"
    r"это|the\s+reason|причина|no\b|not\b|for\s+example|например)",
    re.I,
)
_EXPLANATION_TAIL_SAFE_RE = re.compile(
    r"^\s*(?:$|[?.!]|[,;:—–-]|\b(?:and|or|и|или|also|также|"
    r"is|are|was|were|as|in|on|for|with|about|of|to|that|which|"
    r"who|whom|whose)\b)",
    re.I,
)


def _next_sentence_safe(text: str, end: int) -> bool:
    # Точки внутри URL и Markdown-ссылок не являются границей предложения.
    scan = _scan_text(text)
    match = re.search(r"[.!?]\s*([^.!?]+)", scan[end:])
    if not match:
        return True
    return bool(_NEXT_SAFE_START_RE.match(match.group(1)))


def _quote_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    for quote in ('"', "'", "`"):
        left = text.rfind(quote, 0, start)
        right = text.find(quote, end)
        if left != -1 and right != -1:
            return left + 1, right
    return None


def _markdown_span_at(text: str, start: int) -> re.Match[str] | None:
    for match in _MARKDOWN_SPAN_RE.finditer(text):
        if match.start() <= start < match.end():
            return match
    return None


def _markdown_unsafe(text: str, start: int, end: int) -> bool:
    term = text[start:end].strip()
    match = _markdown_span_at(text, start)
    if match is None:
        return False
    span = match.group(0)
    if span.startswith("["):
        inner = span[1:span.find("]")]
    elif span.startswith("**"):
        inner = span[2:-2]
    elif span.startswith("__"):
        inner = span[2:-2]
    elif span.startswith("`"):
        inner = span[1:-1]
    else:
        inner = span
    inner = inner.strip()
    if inner.casefold() == term.casefold():
        return False
    position = inner.casefold().find(term.casefold())
    if position == -1:
        return True
    # Форматирование не должно скрывать инструкцию: оцениваем внутренний
    # текст теми же safe-правилами.
    return not _is_safe_mention(inner, position, position + len(term))


def _strip_outer_markup(sentence: str) -> str:
    text = sentence.strip()
    changed = True
    while changed:
        changed = False
        for prefix in ("> ", ">", "- ", "* ", "+ "):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
                changed = True
        # Markdown-ссылка в начале: использовать её текст, не URL.
        link = _MARKDOWN_SPAN_RE.match(text)
        if link is not None and text.startswith("["):
            span = link.group(0)
            inner = span[1:span.find("]")].strip()
            text = (inner + text[link.end():]).strip()
            changed = True
            continue
        # Снять открывающий/закрывающий маркер выделения, даже если после
        # него есть обычное продолжение предложения.
        for marker in ("**", "__", "`"):
            if text.startswith(marker):
                text = text[len(marker):].lstrip()
                changed = True
            if text.endswith(marker) and len(text) >= len(marker):
                text = text[:-len(marker)].rstrip()
                changed = True
    return text


def _safe_quote(
    text: str, start: int, end: int, span: tuple[int, int]
) -> bool:
    quote_left, quote_right = span
    content = text[quote_left:quote_right]
    relative_start = start - quote_left
    relative_end = end - quote_left
    prefix = content[:relative_start]
    suffix = content[relative_end:]
    if _UNSAFE_STRUCTURE_RE.search(content):
        return False
    if (_SAFE_PREFIX_RE.search(prefix)
            or _SAFE_NEGATIVE_PREFIX_RE.search(prefix)
            or _SAFE_POST_RE.search(suffix)
            or _SAFE_NEGATIVE_POST_RE.search(suffix)):
        return True
    # Цитата с объяснением безопасна, если сама цитата не скрывает
    # положительную инструкцию.
    if _safe_explanation(content) or _safe_analysis(content):
        return _is_safe_mention(content, relative_start, relative_end)
    return False


def _structural_text(sentence: str) -> str:
    """Убрать markdown-ссылки/выделения/URL перед проверкой разделителей."""
    text = _MARKDOWN_SPAN_RE.sub("", sentence)
    return re.sub(r"https?://\S+", "", text)


def _safe_explanation(sentence: str) -> bool:
    clean = _strip_outer_markup(sentence)
    return bool(_EXPLANATION_START_RE.match(clean)) and (
        _EXPLANATION_STRUCTURE_RE.search(_structural_text(clean)) is None)


def _safe_analysis(sentence: str) -> bool:
    clean = _strip_outer_markup(sentence)
    if not _ANALYSIS_START_RE.match(clean):
        return False
    if _ANALYSIS_NEGATIVE_RE.search(clean):
        return True
    return _UNSAFE_STRUCTURE_RE.search(_structural_text(clean)) is None


def _safe_hypothetical(sentence: str) -> bool:
    return bool(
        _HYPOTHETICAL_START_RE.match(sentence)
        and _HYPOTHETICAL_NEGATIVE_RE.search(sentence)
    )


def _short(value: str, limit: int = 80) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[:limit - 1] + "…"


#: Частые визуальные омографы. Это не полная Unicode-нормализация
#: безопасности, а практическая защита от очевидных подмен.
_HOMOGLYPHS = str.maketrans({
    "а": "a", "А": "a", "е": "e", "Е": "e", "о": "o", "О": "o",
    "с": "c", "С": "c", "р": "p", "Р": "p", "х": "x", "Х": "x",
    "к": "k", "К": "k", "у": "y", "У": "y", "і": "i", "І": "i",
    "ј": "j", "Ј": "j",
    "ѕ": "s", "Ѕ": "s", "ԁ": "d", "ԛ": "q", "ԝ": "w",
    "α": "a", "Α": "a", "ε": "e", "Ε": "e", "ι": "i", "Ι": "i",
    "κ": "k", "Κ": "k", "ν": "v", "Ν": "v", "ο": "o", "Ο": "o",
    "ρ": "p", "Ρ": "p", "τ": "t", "Τ": "t", "υ": "u", "Υ": "u",
    "χ": "x", "Χ": "x", "ω": "w", "Ω": "w",
})
_ZERO_WIDTH = frozenset("\u200b\u200c\u200d\u200e\u200f\u2060\ufeff")


def _fold_char(char: str, *, case_sensitive: bool) -> str:
    normalized = unicodedata.normalize("NFKC", char)
    normalized = normalized.translate(_HOMOGLYPHS)
    return normalized if case_sensitive else normalized.casefold()


def _compact(text: str, *, case_sensitive: bool) -> tuple[str, list[int]]:
    """Оставить буквы/цифры и запомнить позиции в исходном тексте.

    Пробелы, дефисы, точки, запятые, звёздочки, en dash, zero-width и другие
    не-алфанумерические разделители не участвуют в сравнении, но границы
    слова затем проверяются по исходному тексту.
    """
    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        if char in _ZERO_WIDTH:
            continue
        for folded in _fold_char(char, case_sensitive=case_sensitive):
            if folded.isalnum():
                chars.append(folded)
                positions.append(index)
    return "".join(chars), positions


def _term_matches(
    term: str, text: str, *, case_sensitive: bool
) -> list[tuple[int, int]]:
    compact_term, _ = _compact(term, case_sensitive=case_sensitive)
    if not compact_term:
        raise InvariantValidationError(f"пустой терм для проверки {term!r}")
    compact_text, position_map = _compact(text, case_sensitive=case_sensitive)
    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        found = compact_text.find(compact_term, start)
        if found == -1:
            break
        original_start = position_map[found]
        original_end = position_map[found + len(compact_term) - 1] + 1
        before = text[original_start - 1] if original_start else ""
        after = text[original_end] if original_end < len(text) else ""
        if not ((before.isalnum() or before == "_")
                or (after.isalnum() or after == "_")):
            matches.append((original_start, original_end))
        start = found + 1
    return matches


def _code_ranges(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _CODE_FENCE_RE.finditer(text)]


def _inside_ranges(position: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in ranges)


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    separators = (",", ";", "\n", ".", "?", "!")
    left = max(text.rfind(item, 0, start) for item in separators)
    right_candidates = [index for index in (
        text.find(item, end) for item in separators) if index != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return (left + 1 if left != -1 else 0), right


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    separators = (".", "?", "!", "\n")
    left = max(text.rfind(item, 0, start) for item in separators)
    right_candidates = [index for index in (
        text.find(item, end) for item in separators) if index != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return (left + 1 if left != -1 else 0), right


def _scan_text(text: str) -> str:
    """Убрать разметку/URL, сохранив длину и позиции исходного текста.

    Нужен для поиска инструкций: ``**use it**`` и ``[use it](url)`` должны
    оставаться видимыми, а URL — нет.
    """
    chars = list(text)
    for match in re.finditer(r"\[([^\]]*)\]\(([^)]*)\)", text):
        for index in range(match.start(), match.end()):
            chars[index] = " "
        inner_start = match.start() + 1
        for offset, char in enumerate(match.group(1)):
            chars[inner_start + offset] = char
    for index, char in enumerate(text):
        if char in "*`":
            chars[index] = " "
    # Двойное подчёркивание — markdown-выделение; одиночное оставляем,
    # чтобы не ломать термины вроде my_tool.
    for match in re.finditer(r"__", text):
        chars[match.start()] = " "
        chars[match.start() + 1] = " "
    for match in re.finditer(r"https?://[^\s)]+", text):
        for index in range(match.start(), min(match.end(), len(chars))):
            chars[index] = " "
    return "".join(chars)


def _last_clause_boundary(text: str, position: int) -> int:
    left = max(
        (text.rfind(item, 0, position)
         for item in (".", "?", "!", ",", ";", "\n")),
        default=-1,
    )
    for match in _CLAUSE_CONJUNCTION_RE.finditer(text, 0, position):
        left = max(left, match.end() - 1)
    return left


def _is_question_context(text: str, position: int) -> bool:
    left, right = _sentence_bounds(text, position, position)
    if right < len(text) and text[right] == "?":
        return True
    return text[left:right].rstrip().endswith("?")


def _conditional_outcome_is_negative(text: str, position: int) -> bool:
    left, right = _sentence_bounds(text, position, position)
    if right < len(text) and text[right] == "?":
        return True
    return bool(_NEGATIVE_OUTCOME_RE.search(text[left:right]))


def _imperative_pattern(term: str) -> re.Pattern[str]:
    compact, _ = _compact(term, case_sensitive=False)
    objects = [
        "it", "its", "them", "его", "её", "ее", "их",
        "это", "этот", "эту", "него", "неё", "нее", "них",
    ]
    if compact:
        objects.append(compact)
    if term.strip():
        objects.append(unicodedata.normalize("NFKC", term))
    object_re = "|".join(
        re.escape(item)
        for item in sorted(set(objects), key=len, reverse=True)
    )
    return re.compile(
        rf"\b(?:{_IMPERATIVE_VERBS})\b(?!\s+of\b)"
        rf"(?:\s+\S+){{0,4}}?\s+(?:{object_re})\b",
        re.I,
    )


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Разбить текст на предложения по ``.?!`` и переводу строки."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?\n]", text):
        spans.append((start, match.end()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _has_positive_instruction(text: str, start: int, end: int) -> bool:
    """Есть ли в тексте положительная инструкция использовать терм.

    Безопасный фрагмент не должен делать безопасным весь остальной ответ,
    поэтому проверяются все предложения и клаузы, а не только окрестность
    упоминания. Отрицания, вопросы и гипотетические условия с отрицательным
    исходом инструкцией не считаются.
    """
    term = text[start:end]
    scan = _scan_text(text)
    pattern = _imperative_pattern(term)
    previous_end = 0
    for match in pattern.finditer(scan):
        matched_text = scan[match.start():match.end()]
        if re.search(
                r"\b(?:instead\s+of|rather\s+than|вместо)\b",
                matched_text, re.I):
            # «Use Podman instead of Docker» — это не использование Docker.
            previous_end = match.end()
            continue
        if _is_question_context(scan, match.start()):
            previous_end = match.end()
            continue
        window_start = max(
            previous_end, _last_clause_boundary(scan, match.start()) + 1)
        window = scan[window_start:match.start()]
        if _NEGATION_RE.search(window):
            previous_end = match.end()
            continue
        if (_CONDITIONAL_RE.search(window)
                and _conditional_outcome_is_negative(
                    scan, match.start())):
            previous_end = match.end()
            continue
        return True

    compact, _ = _compact(term, case_sensitive=False)
    names = {re.escape(unicodedata.normalize("NFKC", term))}
    if compact:
        names.add(re.escape(compact))
    statement_re = re.compile(
        r"\b(?:" + "|".join(sorted(names, key=len, reverse=True)) + r")\b"
        r"\s+(?:is|are)\s+(?:the\s+)?"
        r"(?:best|better|selected|recommended|preferred|required|chosen|"
        r"the\s+architecture|a\s+good\s+choice)\b",
        re.I,
    )
    for match in statement_re.finditer(scan):
        if not _is_question_context(scan, match.start()):
            return True
    for match in _POSITIVE_PRONOUN_STATEMENT_RE.finditer(scan):
        if not _is_question_context(scan, match.start()):
            return True

    # «Do not use Docker. It is recommended.»: короткая положительная
    # рекомендация с местоимением относится к термину, если он был в
    # непосредственно предыдущем предложении. Без такого антецедента
    # местоимение ничего не блокирует.
    spans = [
        (left, right)
        for left, right in _sentence_spans(scan)
        if scan[left:right].strip()
    ]
    for index, (left, right) in enumerate(spans):
        if index == 0:
            continue
        raw_sentence = scan[left:right].strip()
        if raw_sentence.endswith("?"):
            continue
        sentence = _strip_outer_markup(raw_sentence).strip()
        if not (_POSITIVE_ANAPHORA_RE.fullmatch(sentence)
                or _POSITIVE_RECOMMENDS_IT_RE.fullmatch(sentence)):
            continue
        previous = scan[spans[index - 1][0]:spans[index - 1][1]]
        compact_term, _ = _compact(term, case_sensitive=False)
        if not compact_term:
            continue
        if re.search(
                r"\b(?:instead\s+of|rather\s+than|вместо)\b",
                previous, re.I):
            # «Use Podman instead of Docker. It is recommended.» — «it»
            # относится к альтернативе, а не к запрещённому термину.
            continue
        if _term_matches(term, previous, case_sensitive=False):
            return True
    return False


def _has_force_unsafe(text: str) -> bool:
    """Найти известные попытки обхода во всём тексте, а не в одном предложении.

    Цитаты, определения и отрицания опасной фразы не считаются попыткой.
    """
    ranges = _code_ranges(text)
    for match in _FORCE_UNSAFE_RE.finditer(text):
        left, right = _sentence_bounds(text, match.start(), match.end())
        sentence = text[left:right]
        clean = _strip_outer_markup(sentence)
        if _EXPLANATION_START_RE.match(clean):
            continue
        if _FORCE_UNSAFE_DEFINITION_RE.match(clean):
            continue
        prefix = text[left:match.start()]
        if (_FORCE_UNSAFE_NEGATION_RE.search(prefix)
                or _SAFE_PREFIX_RE.search(prefix)
                or _SAFE_NEGATIVE_PREFIX_RE.search(prefix)):
            continue
        if re.search(
                r"\b(?:is|are)\s+(?:not\s+)?(?:prohibited|forbidden|"
                r"not\s+allowed|запрещ\w*|недопустим\w*)\b",
                sentence, re.I):
            continue
        if re.search(
                r"\b(?:(?:must|should|shall|may|can|could|would)\s+not\s+be|"
                r"cannot\s+be|can't\s+be)\s+"
                r"(?:overridden|ignored|used|allowed|permitted|applied|"
                r"followed|obeyed|trusted|changed|disabled|forgotten|"
                r"neglected|disregarded)\b",
                sentence, re.I):
            continue
        quoted = (
            _quote_span(text, match.start(), match.end()) is not None
            or _inside_ranges(match.start(), ranges)
        )
        if quoted and _FORCE_UNSAFE_EXPLANATION_RE.search(sentence):
            continue
        return True
    return False


def _is_safe_mention(text: str, start: int, end: int) -> bool:
    """Эвристика безопасного упоминания: объяснение/запрет, не рекомендация.

    Это именно эвристика, а не семантический анализ. Она намеренно не
    считает слово «например» достаточным основанием для пропуска.
    """
    ranges = _code_ranges(text)
    if _inside_ranges(start, ranges):
        return False

    sentence_left, sentence_right = _sentence_bounds(text, start, end)
    markdown_span = _markdown_span_at(text, start)
    if markdown_span is not None and markdown_span.end() > sentence_right:
        # URL внутри Markdown-ссылки содержит точки; не обрезаем ссылку
        # по точке в URL.
        sentence_right = markdown_span.end()
    sentence = text[sentence_left:sentence_right]

    if _has_force_unsafe(text):
        return False
    if _markdown_unsafe(text, start, end):
        return False
    if _has_positive_instruction(text, start, end):
        # Безопасный фрагмент не должен делать безопасным остальной ответ:
        # положительная инструкция в любом предложении/клаузе запрещает PASS.
        return False

    quote = _quote_span(text, start, end)
    if quote is not None:
        # Цитата безопасна только при отрицании/запрете внутри неё и
        # отсутствии дополнительной инструкции вокруг.
        return (
            _safe_quote(text, start, end, quote)
            and _UNSAFE_STRUCTURE_RE.search(_structural_text(sentence)) is None
        )

    # Гипотетический/аналитический разбор с явным отрицательным выводом
    # проверяем до общей проверки разделителей.
    if _safe_hypothetical(sentence):
        return True
    if _safe_analysis(sentence):
        return True

    # Объяснение: список/тире/«and» сами по себе не инструкция. Двоеточие
    # и «;» остаются небезопасными (см. _safe_explanation).
    if _safe_explanation(sentence):
        tail_end = end
        markdown = _markdown_span_at(text, start)
        if markdown is not None:
            tail_end = markdown.end()
        tail = text[tail_end:sentence_right]
        if not _EXPLANATION_TAIL_SAFE_RE.match(tail):
            return False
        return _next_sentence_safe(text, end)

    # Снять внешние Markdown/list/blockquote обёртки до проверки разделителей:
    # сама разметка не является инструкцией.
    clean_sentence = _strip_outer_markup(sentence)
    if _UNSAFE_STRUCTURE_RE.search(_structural_text(clean_sentence)):
        return False

    left, right = _clause_bounds(text, start, end)
    clause = text[left:right]
    relative_start = start - left
    relative_end = end - left
    prefix = clause[:relative_start]
    suffix = clause[relative_end:]

    if _EPISTEMIC_RE.search(prefix):
        # «I do not think/sure/believe ... is prohibited» — это не запрет,
        # а сомнение или отрицание самого утверждения.
        return False
    if (_SAFE_PREFIX_RE.search(prefix)
            or _SAFE_NEGATIVE_PREFIX_RE.search(prefix)):
        return True
    if (_SAFE_POST_RE.search(suffix)
            or _SAFE_NEGATIVE_POST_RE.search(suffix)):
        return True
    return False


#: Единственный допустимый числовой формат max_value:
#: знак, целая часть, при необходимости точка и дробная часть. Никаких
#: тысяч, валют, единиц, экспоненты и внутренних пробелов.
_STRICT_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)")
#: Число с ровно тремя знаками после точки без других признаков неоднозначно
#: (1.000 — это 1 или 1000?); такой формат отклоняется.
_AMBIGUOUS_THREE_DECIMAL_RE = re.compile(r"[+-]?\d{1,3}\.\d{3}")
_TABLE_ROW_RE = re.compile(r"(?m)^[ \t]*\|.*\|[ \t]*$")
_BARE_NUMERIC_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*[.;,]?\s*$")
_LOOSE_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?|"
    r"\.\d+(?:[eE][-+]?\d+)?)")
#: Денежный/бюджетный контекст: только такие независимые числа считаются
#: относящимися к бюджету. Обычные «Python 3», «Version: 150», «2026 год»
#: сюда не попадают.
_MONEY_CONTEXT_RE = re.compile(
    r"(?:\b(?:budget|cost|costs|price|prices|amount|total|sum|fee|fees|"
    r"usd|eur|rub|gbp|jpy|cny|dollars?|euros?|rubles?|"
    r"бюджет\w*|стоимост\w*|цен[аыу]\w*|сумм\w*|итог\w*|"
    r"расход\w*|затрат\w*|доллар\w*|евро|рубл\w*)\b|[$€£¥₽])",
    re.I,
)


def _nfkc_with_map(text: str) -> tuple[str, list[int]]:
    """NFKC-нормализовать текст и сохранить карту обратно в исходные позиции.

    ``norm_map[i]`` — индекс исходного символа, из которого получен
    нормализованный символ ``i``; последний элемент указывает на конец
    исходной строки.
    """
    chars: list[str] = []
    mapping: list[int] = []
    for index, char in enumerate(text):
        if char in _ZERO_WIDTH:
            continue
        normalized = unicodedata.normalize("NFKC", char)
        for folded in normalized.translate(_HOMOGLYPHS):
            chars.append(folded)
            mapping.append(index)
    mapping.append(len(text))
    return "".join(chars), mapping


def _normalize_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(_HOMOGLYPHS)
    return normalized.casefold()


def _parse_strict_number(raw: str) -> float | None:
    """Разобрать одно число по строгой грамматике или вернуть None.

    Допустимо: ``50``, ``1.5``, ``-150``, ``+3``, ``.5``. Не допустимо:
    тысячные разделители, валюты, единицы, экспонента, диапазоны, несколько
    чисел, внутренние пробелы и форма ``1.000`` (неоднозначные 3 знака).
    """
    normalized = unicodedata.normalize("NFKC", raw).strip()
    if not normalized or not _STRICT_NUMBER_RE.fullmatch(normalized):
        return None
    if _AMBIGUOUS_THREE_DECIMAL_RE.fullmatch(normalized):
        return None
    try:
        value = float(normalized)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _number_is_continued(rest: str, token: str) -> bool:
    """Похоже ли, что после token число продолжается (1,000 / 1e3 / 50USD)."""
    tail = rest[len(token):]
    if not tail:
        return False
    next_char = tail[0]
    if next_char in ",." and len(tail) > 1 and tail[1].isdigit():
        return True
    return next_char.isalnum() or next_char in "$€£%_"


def _exceeding_number(text: str, limit: float) -> str | None:
    """Вернуть число из текста, которое превышает лимит или неоднозначно.

    Числа внутри обычной прозы (``Python 3``) допустимы, если они не больше
    лимита. Валюты, разделители тысяч и несколько точек неоднозначны и
    трактуются как нарушение (fail-closed).
    """
    normalized = unicodedata.normalize("NFKC", text)
    for match in _LOOSE_NUMBER_RE.finditer(normalized):
        raw = match.group(0)
        value = _parse_strict_number(raw)
        if value is None or value > limit:
            return raw
    return None


def _has_money_number(text: str, limit: float) -> bool:
    """Есть ли в тексте денежный/бюджетный контекст с числом сверх лимита."""
    return bool(_MONEY_CONTEXT_RE.search(text)) and (
        _exceeding_number(text, limit) is not None)


def _max_following_lines(
    text: str, line_end: int, key: str, limit: float
) -> str | None:
    """Проверить строки после значения.

    Блокируем голое число и денежный контекст. Независимые «Version: 150»,
    «420 pages» или «The year is 2026» проходят.
    """
    if line_end == -1:
        return None
    normalized_key = _normalize_key(key)
    for line in text[line_end + 1:].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|"):
            # Табличные строки проверяет отдельный разбор таблиц; чужая
            # строка таблицы не должна блокироваться из-за своего числа.
            continue
        if normalized_key in _normalize_key(stripped):
            # Повторный ключ проверяется отдельным совпадением основного
            # шаблона; не считаем его «чужим полем с числом».
            continue
        if _BARE_NUMERIC_LINE_RE.fullmatch(stripped):
            return stripped
        if _has_money_number(stripped, limit):
            return stripped
    return None


@dataclass(frozen=True)
class ValidationDecision:
    target: str
    allowed: bool
    violations: tuple[Violation, ...]
    checked_rules: int
    checked_checks: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "allowed": self.allowed,
            "violations": [item.to_dict() for item in self.violations],
            "checked_rules": self.checked_rules,
            "checked_checks": self.checked_checks,
            "error": self.error,
        }


class InvariantValidator:
    """Проверка запроса и готового ответа по активным инвариантам."""

    def __init__(self, store: InvariantStore | None = None):
        self._store = store

    # ----- публичные проверки -----

    def check_request(
        self, text: str, *, snapshot: InvariantSnapshot | None = None
    ) -> ValidationDecision:
        return self._check(text, target="request", snapshot=snapshot)

    def check_response(
        self, text: str, *, snapshot: InvariantSnapshot | None = None
    ) -> ValidationDecision:
        return self._check(text, target="response", snapshot=snapshot)

    # ----- внутренняя проверка -----

    def _check(
        self,
        text: str,
        *,
        target: str,
        snapshot: InvariantSnapshot | None,
    ) -> ValidationDecision:
        if not isinstance(text, str):
            raise InvariantValidationError("проверяемый текст должен быть строкой")
        snapshot = self._resolve_snapshot(snapshot)
        if not isinstance(snapshot.rules, (tuple, list)):
            raise InvariantValidationError(
                "snapshot.rules должен быть кортежем/списком правил")
        violations: list[Violation] = []
        checked_rules = 0
        checked_checks = 0

        for rule in snapshot.rules:
            if not isinstance(rule, InvariantRule):
                raise InvariantValidationError(
                    "snapshot.rules содержит объект неожиданного типа")
            if type(rule.active) is not bool:
                raise InvariantValidationError(
                    f"правило {rule.id}: active должен быть bool")
            if not isinstance(rule.checks, (tuple, list)):
                raise InvariantValidationError(
                    f"правило {rule.id}: checks должен быть списком")
            # Структуру проверяем у каждого правила до пропуска неактивного:
            # повреждённый snapshot не должен становиться PASS только потому,
            # что правило выключено.
            for check in rule.checks:
                if not isinstance(check, InvariantCheck):
                    raise InvariantValidationError(
                        f"правило {rule.id}: checks содержит объект "
                        "неожиданного типа")
                if not isinstance(check.kind, str) or check.kind not in CHECK_KINDS:
                    raise InvariantValidationError(
                        f"правило {rule.id}: неизвестный тип проверки {check.kind!r}")
                if not isinstance(check.value, (str, int, float)) or isinstance(
                        check.value, bool):
                    raise InvariantValidationError(
                        f"правило {rule.id}: некорректный тип value "
                        f"{type(check.value).__name__}")
                if check.key is not None and not isinstance(check.key, str):
                    raise InvariantValidationError(
                        f"правило {rule.id}: некорректный тип key")
                if type(check.case_sensitive) is not bool:
                    raise InvariantValidationError(
                        f"правило {rule.id}: case_sensitive должен быть bool")
            if not rule.active:
                continue
            if not rule.checks:
                # Модель не позволяет создать такое правило, но snapshot
                # мог быть мутирован после создания; fail-closed.
                raise InvariantValidationError(
                    f"правило {rule.id}: активное правило не содержит checks")
            checked_rules += 1
            for check in rule.checks:
                if target == "request" and check.kind in {"required_regex", "max_value"}:
                    # Это требования к ответу; запрос не обязан их содержать.
                    continue
                checked_checks += 1
                try:
                    violation = self._apply_check(rule, check, text)
                except InvariantValidationError:
                    raise
                except Exception as exc:
                    raise InvariantValidationError(
                        f"правило {rule.id}: ошибка проверки {check.kind}: {exc}") from exc
                if violation is not None:
                    violations.append(violation)

        return ValidationDecision(
            target=target,
            allowed=not violations,
            violations=tuple(violations),
            checked_rules=checked_rules,
            checked_checks=checked_checks,
        )

    def _resolve_snapshot(
        self, snapshot: InvariantSnapshot | None
    ) -> InvariantSnapshot:
        if snapshot is not None:
            if not isinstance(snapshot, InvariantSnapshot):
                raise InvariantValidationError(
                    "snapshot должен быть InvariantSnapshot")
            return snapshot
        if self._store is None:
            raise InvariantValidationError(
                "не передан snapshot и не подключено хранилище")
        try:
            return self._store.load()
        except InvariantStoreError as exc:
            raise InvariantValidationError(
                f"не удалось загрузить инварианты: {exc}") from exc

    # ----- отдельные проверки -----

    def _apply_check(
        self, rule: InvariantRule, check: InvariantCheck, text: str
    ) -> Violation | None:
        if check.kind == "forbidden_term":
            return self._forbidden_term(rule, check, text)
        if check.kind == "forbidden_regex":
            return self._forbidden_regex(rule, check, text)
        if check.kind == "required_regex":
            return self._required_regex(rule, check, text)
        if check.kind == "forbidden_import":
            return self._forbidden_import(rule, check, text)
        if check.kind == "max_value":
            return self._max_value(rule, check, text)
        raise InvariantValidationError(
            f"правило {rule.id}: неизвестный тип проверки {check.kind!r}")

    def _term_violation(
        self, rule: InvariantRule, check: InvariantCheck, text: str
    ) -> Violation | None:
        term = str(check.value)
        for start, end in _term_matches(
                term, text, case_sensitive=check.case_sensitive):
            if not _is_safe_mention(text, start, end):
                return Violation(
                    invariant_id=rule.id,
                    check_kind=check.kind,
                    message=f"обнаружен запрещённый термин {term!r}: "
                            f"{_short(text[start:end])!r}",
                )
        return None

    def _forbidden_term(
        self, rule: InvariantRule, check: InvariantCheck, text: str
    ) -> Violation | None:
        return self._term_violation(rule, check, text)

    def _forbidden_regex(
        self, rule: InvariantRule, check: InvariantCheck, text: str
    ) -> Violation | None:
        flags = 0 if check.case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(str(check.value), flags)
        except re.error as exc:
            raise InvariantValidationError(
                f"правило {rule.id}: некорректный regex: {exc}") from exc
        for match in pattern.finditer(text):
            if not _is_safe_mention(text, match.start(), match.end()):
                return Violation(
                    invariant_id=rule.id,
                    check_kind=check.kind,
                    message=f"обнаружено запрещённое выражение: "
                            f"{_short(match.group(0))!r}",
                )
        return None

    def _required_regex(
        self, rule: InvariantRule, check: InvariantCheck, text: str
    ) -> Violation | None:
        flags = 0 if check.case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(str(check.value), flags)
        except re.error as exc:
            raise InvariantValidationError(
                f"правило {rule.id}: некорректный regex: {exc}") from exc
        if pattern.search(text) is None:
            return Violation(
                invariant_id=rule.id,
                check_kind=check.kind,
                message="ответ не содержит обязательного выражения",
            )
        return None

    def _forbidden_import(
        self, rule: InvariantRule, check: InvariantCheck, text: str
    ) -> Violation | None:
        target = str(check.value)
        code_blocks: list[tuple[str, bool]] = []
        for match in _CODE_FENCE_RE.finditer(text):
            language = match.group("lang").strip().lower()
            code = match.group("code")
            python_label = language in {"", "python", "py", "python3"}
            if not python_label and not _PYTHON_HINT_RE.search(code):
                continue
            code_blocks.append((code, python_label))

        # Целый текст без fence может быть Python-фрагментом с вложенными
        # конструкциями (if/for/with). Парсим только если это действительно
        # корректный Python; иначе работаем со строками ниже.
        if not _CODE_FENCE_RE.search(text) and _PYTHON_HINT_RE.search(text):
            try:
                ast.parse(text.strip())
            except SyntaxError:
                pass
            else:
                code_blocks.append((text.strip(), False))

        # Импорт может быть приведён вне fenced-блока.
        for match in _DIRECT_IMPORT_RE.finditer(text):
            code_blocks.append((match.group(0), False))
        for match in _DYNAMIC_IMPORT_LINE_RE.finditer(text):
            line_end = text.find("\n", match.start())
            code_blocks.append((
                text[match.start():line_end if line_end != -1 else len(text)],
                False,
            ))

        for code, strict in code_blocks:
            try:
                tree = ast.parse(code)
            except SyntaxError as exc:
                if strict:
                    raise InvariantValidationError(
                        f"правило {rule.id}: невозможно разобрать Python-код "
                        f"для проверки импорта: {exc}") from exc
                # Строка вне явного Python-контекста может быть прозой.
                continue
            imported = self._imports(tree)
            if any(self._module_matches(name, target) for name in imported):
                return Violation(
                    invariant_id=rule.id,
                    check_kind=check.kind,
                    message=f"обнаружен запрещённый импорт {target!r}",
                )
        return None

    @staticmethod
    def _imports(tree: ast.AST) -> list[str]:
        names: list[str] = []
        aliases: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.append(node.module)
                if node.module == "importlib":
                    aliases.update(
                        alias.asname or alias.name for alias in node.names)
            elif isinstance(node, ast.Call):
                names.extend(
                    InvariantValidator._dynamic_import_names(node, aliases))
                function = node.func
                if (isinstance(function, ast.Name)
                        and function.id in {"exec", "eval"}
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)):
                    try:
                        nested = ast.parse(node.args[0].value)
                    except SyntaxError:
                        continue
                    names.extend(InvariantValidator._imports(nested))
        return names

    @staticmethod
    def _dynamic_import_names(node: ast.Call, aliases: set[str] | None = None) -> list[str]:
        """Имена из __import__/importlib.import_module с literal-строкой."""
        aliases = aliases or set()
        function = node.func
        is_dynamic = False
        if isinstance(function, ast.Name):
            is_dynamic = function.id in {"__import__", "import_module"} | aliases
        elif isinstance(function, ast.Attribute):
            is_dynamic = function.attr in {"__import__", "import_module"}
        if not is_dynamic:
            return []

        argument: ast.AST | None = node.args[0] if node.args else None
        if argument is None:
            for keyword in node.keywords:
                if keyword.arg in {"name", "module"}:
                    argument = keyword.value
                    break
        if (argument is not None
                and isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)):
            return [argument.value]
        return []

    @staticmethod
    def _module_matches(module: str, target: str) -> bool:
        return module == target or module.startswith(target + ".")

    def _max_violation(
        self, rule: InvariantRule, check: InvariantCheck,
        raw: str, value: float | None = None,
    ) -> Violation:
        if value is None:
            message = (f"{check.key}: неоднозначное или неподдерживаемое "
                       f"значение {_short(raw)!r}")
        else:
            message = (f"{check.key}={_short(raw)} превышает "
                       f"максимум {check.value}")
        return Violation(
            invariant_id=rule.id,
            check_kind=check.kind,
            message=message,
        )

    def _max_value(
        self, rule: InvariantRule, check: InvariantCheck, text: str
    ) -> Violation | None:
        if check.key is None:
            raise InvariantValidationError(
                f"правило {rule.id}: max_value требует key")
        key = str(check.key)
        limit = float(check.value)
        normalized_key = _normalize_key(key)

        table_ranges: list[tuple[int, int]] = []
        # Markdown-таблицы: ключ и значение находятся в одной строке.
        for row_match in _TABLE_ROW_RE.finditer(text):
            table_ranges.append((row_match.start(), row_match.end()))
            row = row_match.group(0).strip()
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            for index, cell in enumerate(cells):
                normalized_cell = _normalize_key(cell)
                if normalized_cell == normalized_key:
                    if index + 1 >= len(cells):
                        return self._max_violation(rule, check, row)
                    raw_value = cells[index + 1]
                    extra_start = index + 2
                else:
                    # Ячейка вида «budget_usd: 150» (в том числе с текстом
                    # перед ключом).
                    inline = re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(normalized_key)}"
                        rf"(?![A-Za-z0-9_])\s*[:=]\s*(.*)$",
                        normalized_cell,
                        re.IGNORECASE,
                    )
                    if inline is None:
                        continue
                    raw_value = inline.group(1).strip()
                    extra_start = index + 1
                value = _parse_strict_number(raw_value)
                if value is None:
                    return self._max_violation(rule, check, raw_value)
                if value > limit:
                    return self._max_violation(
                        rule, check, raw_value, value)
                # Числа в соседних ячейках проверяем, но независимое поле
                # с числом в пределах лимита не блокируем.
                for extra in cells[extra_start:]:
                    if _exceeding_number(extra, limit) is not None:
                        return self._max_violation(rule, check, row)
                line_end = text.find("\n", row_match.end())
                following = _max_following_lines(
                    text, line_end, key, limit)
                if following is not None:
                    return self._max_violation(rule, check, following)

        normalized, norm_map = _nfkc_with_map(text)
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(normalized_key)}"
            r"(?![A-Za-z0-9_])[ \t]*[:=][ \t]*([^\n]*)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(normalized):
            # Строку таблицы уже проверил отдельный разбор.
            if _inside_ranges(norm_map[match.start()], table_ranges):
                continue
            group_start, group_end = match.start(1), match.end(1)
            value_start = norm_map[group_start]
            if group_end > group_start:
                value_end = norm_map[group_end - 1] + 1
            else:
                value_end = value_start
            line_end = text.find("\n", value_end)
            if line_end == -1:
                line_end = len(text)
            rest = text[value_start:line_end].strip()
            token_match = re.match(_STRICT_NUMBER_RE, rest)
            if token_match is None:
                return self._max_violation(rule, check, rest)
            token = token_match.group(0)
            if _number_is_continued(rest, token):
                return self._max_violation(rule, check, rest)
            value = _parse_strict_number(token)
            if value is None:
                return self._max_violation(rule, check, rest)
            if value > limit:
                return self._max_violation(rule, check, token, value)

            tail = rest[len(token):]
            if not self._max_tail_allows_prose(tail, key, limit):
                return self._max_violation(rule, check, rest)

            following = _max_following_lines(text, line_end, key, limit)
            if following is not None:
                return self._max_violation(rule, check, following)
        return None

    @staticmethod
    def _max_tail_allows_prose(
        tail: str, key: str, limit: float
    ) -> bool:
        """Разрешить обычный текст после значения, но не второе число."""
        if not tail.strip():
            return True
        normalized_key = _normalize_key(key)
        if normalized_key in _normalize_key(tail):
            return False
        stripped = tail.lstrip()
        if stripped[0] not in ".?!,":
            return False
        after = stripped[1:]
        if normalized_key in _normalize_key(after):
            return False
        after_stripped = after.lstrip()
        # Второе голое число после разделителя — это продолжение поля.
        if (_BARE_NUMERIC_LINE_RE.fullmatch(after_stripped)
                and _exceeding_number(after_stripped, limit) is not None):
            return False
        # Денежный контекст («project costs 150») тоже относится к бюджету.
        if _has_money_number(after, limit):
            return False
        return True


__all__ = ["InvariantValidator", "ValidationDecision"]
