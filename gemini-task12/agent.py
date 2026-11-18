"""Agent: ответы Gemini с трёхслойной памятью (День 11).

Один вопрос пользователя (Agent.ask, remember=True):

    update_memory(вопрос)          ← 1 вызов Gemini «менеджер памяти»: операции для working / long_term,
                                     код проверяет их правилами memory/rules.py и пишет в свои файлы
    blocks = memory.prompt_blocks() ← правила чтения: LT → WM (если задача активна) → последние N сообщений
    count_tokens по каждому блоку   ← токены каждого слоя отдельно + всего промпта (бесплатно)
    generate_content(...)           ← usage_metadata: фактические токены
    memory.record_exchange(...)     ← краткосрочная память: только успешная пара

remember=False (контрольные вопросы) — память не читается на запись и не меняется.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from google.genai import errors, types

import config
from memory import PROMPT_ORDER, MemoryManager, PromptBlock
from memory.rules import OPERATIONS_SCHEMA, MemoryFormatError, parse_operations
from memory.storage import content_text, text_content
from token_stats import TokenCounter, Usage, compare_count_with_usage, sum_known, usage_from_response
from utils import redact_secrets

logger = logging.getLogger("agent")


class AgentAPIError(Exception):
    """Ошибка обращения к Gemini (уже с замаскированным ключом)."""


@dataclass
class CallRecord:
    """Служебный вызов generate_content (обновление памяти), не ответ пользователю."""
    kind: str
    ok: bool
    usage: Usage = field(default_factory=Usage)
    request_tokens: int | None = None
    error: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usage"] = self.usage.to_dict()
        return d


@dataclass
class AgentResult:
    text: str
    status: str                          # ok | api_error | empty_response
    usage: Usage = field(default_factory=Usage)
    tokens: dict = field(default_factory=dict)        # count_tokens: по слоям, вопрос, весь промпт
    prompt_layers: list[str] = field(default_factory=list)
    request_messages: int = 0
    error: str | None = None
    service_calls: list[CallRecord] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def request_tokens(self) -> int | None:
        return self.tokens.get("total")


def build_memory_request(view: dict, assistant_text: str | None, user_text: str, message_no: int) -> str:
    """Запрос менеджеру памяти: ТОЛЬКО текущие LT и WM, предыдущий ответ ассистента и новое сообщение.

    Профиль пользователя сюда намеренно не передаётся: это отдельное хранилище настроек,
    которое менеджер памяти не должен читать или изменять.
    """
    def as_json(d: dict) -> str:
        return json.dumps(d, ensure_ascii=False, indent=1) if d else "{}"

    prev = f"[{message_no - 1}] Ассистент: {assistant_text}" if assistant_text else "(нет)"
    return (
        f"Долговременная память (JSON):\n{as_json(view.get('long_term') or {})}\n\n"
        f"Рабочая память текущей задачи (JSON):\n{as_json(view.get('working') or {})}\n\n"
        "Предыдущее сообщение ассистента (только контекст: его предложения — НЕ решения пользователя, "
        f"пока пользователь явно их не принял):\n{prev}\n\n"
        f"Новое сообщение пользователя:\n[{message_no}] Пользователь: {user_text}\n\n"
        "Верни операции записи в память по правилам (пустой список, если записывать нечего)."
    )


class Agent:
    def __init__(
        self,
        client: Any,
        memory: MemoryManager | None = None,
        *,
        model: str = config.MODEL,
        system_instruction: str = config.SYSTEM_INSTRUCTION,
        memory_system_instruction: str = config.MEMORY_SYSTEM_INSTRUCTION,
        answer_thinking_level: str | None = config.ANSWER_THINKING_LEVEL,
        memory_thinking_level: str | None = config.MEMORY_THINKING_LEVEL,
        max_output_tokens: int = config.MAX_OUTPUT_TOKENS,
        memory_max_output_tokens: int = config.MEMORY_MAX_OUTPUT_TOKENS,
        count_tokens: bool = config.COUNT_TOKENS,
    ) -> None:
        self.client = client
        self.memory = memory if memory is not None else MemoryManager()
        self.model = model
        self.system_instruction = system_instruction
        self.memory_system_instruction = memory_system_instruction
        self.answer_thinking_level = answer_thinking_level
        self.memory_thinking_level = memory_thinking_level
        self.max_output_tokens = max_output_tokens
        self.memory_max_output_tokens = memory_max_output_tokens
        self.counter = TokenCounter(client, model, enabled=count_tokens)
        from profile import ProfileStore
        self.profile = ProfileStore(self.memory.state_dir / "user_profile.json" if self.memory.state_dir else None)

    # ----- низкоуровневый вызов -----

    def _config(self, system_instruction: str, thinking: str | None, max_out: int) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(thinking_level=thinking) if thinking else None,
            max_output_tokens=max_out,
        )

    def _generate(self, contents: list[types.Content], gen_config: types.GenerateContentConfig) -> tuple[str | None, Usage, Any]:
        """(текст или None, usage, finish_reason). Ошибки API/сети → AgentAPIError с понятным текстом."""
        try:
            response = self.client.models.generate_content(model=self.model, contents=contents, config=gen_config)
        except errors.APIError as exc:
            msg = redact_secrets(f"{exc.code} {exc.status or ''}: {exc.message or exc}".strip())
            hint = {
                400: "некорректный запрос", 401: "неверный API-ключ", 403: "нет доступа (проверьте ключ)",
                404: f"модель {self.model} не найдена", 429: "превышен лимит запросов, подождите",
            }.get(exc.code, "временная ошибка сервера" if (exc.code or 0) >= 500 else "ошибка API")
            raise AgentAPIError(f"Ошибка Gemini API ({hint}): {msg}") from exc
        except Exception as exc:  # сеть, таймаут
            raise AgentAPIError(f"Не удалось связаться с Gemini: {redact_secrets(f'{type(exc).__name__}: {exc}')}") from exc

        usage = usage_from_response(response)
        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate else None
        finish_reason = candidate.finish_reason if candidate else getattr(response, "prompt_feedback", None)
        text = content_text(content) if content and content.parts else ""
        return (text or None), usage, finish_reason

    # ----- запись в память -----

    def update_memory(self, user_text: str) -> CallRecord:
        """Один вызов менеджера памяти → операции → проверка и запись по слоям.

        Ошибка вызова или формата — ни один слой не меняется, CallRecord(ok=False).
        """
        st = self.memory.short_term
        message_no = st.next_message_no()
        assistant_text = st.last_assistant_text()
        request = [text_content("user", build_memory_request(self.memory.extraction_view(), assistant_text,
                                                             user_text, message_no))]
        counted, count_err = self.counter.count(request)
        gen_config = self._config(self.memory_system_instruction, self.memory_thinking_level,
                                  self.memory_max_output_tokens)
        gen_config.response_mime_type = "application/json"
        gen_config.response_json_schema = OPERATIONS_SCHEMA
        meta: dict = {"message": message_no, "request_count_error": count_err}
        started = time.monotonic()

        def fail(error: str, usage: Usage | None = None) -> CallRecord:
            logger.warning("Память не обновлена (сообщение %d): %s", message_no, error)
            return CallRecord("memory_update", False, usage or Usage(), counted, error, meta)

        try:
            text, usage, finish_reason = self._generate(request, gen_config)
        except AgentAPIError as exc:
            return fail(str(exc))
        meta.update(count_vs_usage=compare_count_with_usage(counted, usage),
                    finish_reason=str(finish_reason) if finish_reason is not None else None,
                    seconds=round(time.monotonic() - started, 2))
        if not text:
            return fail(f"модель вернула пустой ответ (finish_reason={finish_reason})", usage)
        try:
            ops = parse_operations(text)
        except MemoryFormatError as exc:
            meta["raw"] = text[:500]
            return fail(str(exc), usage)
        decisions = self.memory.apply_operations(ops, user_text=user_text, assistant_text=assistant_text,
                                                 message_no=message_no)
        for layer in ("working", "long_term"):
            for r in decisions[layer]["rejected"]:
                logger.info("%s: операция отклонена (%s): %s", layer, r["rejection"], r.get("key"))
        meta.update(proposed=len(ops), decisions=decisions)
        return CallRecord("memory_update", True, usage, counted, None, meta)

    # ----- токены по слоям -----

    def token_breakdown(self, blocks: list[PromptBlock], question: types.Content) -> dict:
        """count_tokens каждого слоя отдельно, вопроса и всего промпта (contents; без system instruction)."""
        by_layer = {b.layer: b for b in blocks}
        layers, errors_ = {}, []
        for name in PROMPT_ORDER:
            if name in by_layer:
                n, err = self.counter.count(by_layer[name].contents)
                if err:
                    errors_.append(f"{name}: {err}")
            else:
                n = 0
            layers[name] = n
        profile_block = by_layer.get("profile")
        profile_tokens, profile_err = self.counter.count(profile_block.contents) if profile_block else (0, None)
        if profile_err:
            errors_.append(f"profile: {profile_err}")
        q, q_err = self.counter.count([question])
        total, t_err = self.counter.count([c for b in blocks for c in b.contents] + [question])
        errors_ += [e for e in (q_err, t_err) if e]
        return {"layers": layers, "profile": profile_tokens, "question": q, "total": total,
                "sum_of_parts": sum_known([*layers.values(), profile_tokens, q]), "errors": errors_}

    # ----- вопрос пользователя -----

    def ask(self, question: str, *, remember: bool = True) -> AgentResult:
        calls: list[CallRecord] = []
        self.profile.load()
        if remember:
            self.profile.observe(question, self.memory.short_term.next_message_no())
            calls.append(self.update_memory(question))
        user = text_content("user", question)
        blocks = self.memory.prompt_blocks()
        profile_text = self.profile.profile.render()
        if profile_text:
            from memory.base import PromptBlock
            blocks = [PromptBlock("profile", [text_content("user", profile_text)]), *blocks]
        contents = [c for b in blocks for c in b.contents] + [user]
        tokens = self.token_breakdown(blocks, user)
        gen_config = self._config(self.system_instruction, self.answer_thinking_level, self.max_output_tokens)

        def result(text: str, status: str, usage: Usage | None = None, error: str | None = None) -> AgentResult:
            return AgentResult(text, status, usage or Usage(), tokens, [b.layer for b in blocks], len(contents),
                               error, calls)

        try:
            text, usage, finish_reason = self._generate(contents, gen_config)
        except AgentAPIError as exc:
            logger.error("%s", exc)
            return result(f"{exc}. Попробуйте ещё раз позже.", "api_error", error=str(exc))
        if not text:
            msg = f"Модель не вернула ответ (причина: {finish_reason})."
            return result(msg, "empty_response", usage, error=msg)
        if remember:
            self.memory.record_exchange(user, text_content("model", text))
        return result(text, "ok", usage)

    def play_exchange(self, user_text: str, model_text: str) -> CallRecord:
        """Проигрывание сценария: обновить память по реплике и записать заранее написанный ответ."""
        call = self.update_memory(user_text)
        self.memory.record_exchange(text_content("user", user_text), text_content("model", model_text))
        return call
