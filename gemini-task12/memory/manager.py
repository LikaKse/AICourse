"""MemoryManager: три независимых слоя, маршрутизация записи и сборка промпта.

    state_dir/short_term.json      ← ShortTermMemory   (история диалога)
    state_dir/working_memory.json  ← WorkingMemory     (текущая задача)
    state_dir/long_term.json       ← LongTermMemory    (факты, решения, знания; без настроек ответа)

Порядок промпта (правила чтения — memory/rules.READ_RULES):

    [долговременная] → [рабочая, если задача активна] → [последние N сообщений] → вопрос

При создании менеджер загружает все три файла — это и есть восстановление после перезапуска.
"""

from __future__ import annotations

import os
from pathlib import Path

from google.genai import types

import config
from memory.base import MemoryLayer, PromptBlock
from memory.long_term import LongTermMemory
from memory.rules import LAYERS, MODEL_WRITABLE_LAYERS
from memory.short_term import ShortTermMemory
from memory.storage import content_text
from memory.working import WorkingMemory

PROMPT_ORDER = ("long_term", "working", "short_term")
LAYER_ALIASES = {
    "short": "short_term", "st": "short_term", "short_term": "short_term", "short-term": "short_term",
    "working": "working", "work": "working", "wm": "working",
    "long": "long_term", "lt": "long_term", "long_term": "long_term", "long-term": "long_term",
}
ROLE_NAMES = {"user": "Пользователь", "model": "Ассистент"}
FILE_NAMES = {"short_term": config.SHORT_TERM_FILE, "working": config.WORKING_FILE,
              "long_term": config.LONG_TERM_FILE}


def resolve_layer(name: str) -> str:
    layer = LAYER_ALIASES.get((name or "").strip().lower())
    if layer is None:
        raise ValueError(f"Неизвестный слой {name!r}; допустимо: short, working, long")
    return layer


class MemoryManager:
    def __init__(
        self,
        state_dir: str | os.PathLike | None = None,
        *,
        window: int = config.SHORT_TERM_WINDOW,
        render_evidence: bool = config.RENDER_EVIDENCE,
        include_unconfirmed: bool = config.INCLUDE_UNCONFIRMED_PROPOSALS,
    ) -> None:
        self.state_dir = Path(state_dir) if state_dir is not None else None
        self.short_term = ShortTermMemory(self.file_path("short_term"), window=window)
        self.working = WorkingMemory(self.file_path("working"))
        self.long_term = LongTermMemory(self.file_path("long_term"), render_evidence=render_evidence,
                                        include_unconfirmed=include_unconfirmed)

    def file_path(self, layer: str) -> Path | None:
        return self.state_dir / FILE_NAMES[layer] if self.state_dir else None

    @property
    def layers(self) -> dict[str, MemoryLayer]:
        return {"short_term": self.short_term, "working": self.working, "long_term": self.long_term}

    # ----- запись -----

    def extraction_view(self) -> dict:
        """Что видит менеджер памяти (LLM): LT без legacy-настроек профиля и working, без журналов."""
        return {"long_term": self.long_term.public_view(), "working": self.working.public_view()}

    def apply_operations(self, ops: list[dict], *, user_text: str, assistant_text: str | None,
                         message_no: int) -> dict:
        """Разложить операции модели по слоям; каждый слой проверяет их своими правилами."""
        by_layer: dict[str, list[dict]] = {name: [] for name in MODEL_WRITABLE_LAYERS}
        unrouted = []
        for op in ops:
            layer = op.get("layer")
            if layer in by_layer:
                by_layer[layer].append(op)
            elif layer == "short_term":
                unrouted.append({**op, "rejection": "краткосрочную память пишет только код"})
            else:
                unrouted.append({**op, "rejection": f"неизвестный слой {layer!r}"})
        kw = {"user_text": user_text, "assistant_text": assistant_text, "message_no": message_no}
        return {"working": self.working.apply_operations(by_layer["working"], **kw),
                "long_term": self.long_term.apply_operations(by_layer["long_term"], **kw),
                "unrouted": unrouted}

    def record_exchange(self, user: types.Content, model: types.Content) -> None:
        """Правило записи краткосрочной памяти: успешная пара целиком (вызывает Agent)."""
        self.short_term.add_exchange(user, model)

    # ----- чтение -----

    def prompt_blocks(self) -> list[PromptBlock]:
        blocks = []
        for name in PROMPT_ORDER:
            contents = self.layers[name].prompt_contents()
            if contents:
                blocks.append(PromptBlock(name, contents))
        return blocks

    def build_contents(self, new_message: types.Content | None = None) -> list[types.Content]:
        contents = [c for b in self.prompt_blocks() for c in b.contents]
        if new_message is not None:
            contents.append(new_message)
        return contents

    def context_text(self) -> str:
        """Текст всего, что уходит в модель перед вопросом («Роль: текст» по строкам)."""
        return "\n".join(f"{ROLE_NAMES.get(c.role, c.role)}: {content_text(c)}" for c in self.build_contents())

    # ----- очистка -----

    def clear(self, name: str) -> list[str]:
        """Очистить один слой ('short' | 'working' | 'long') или все ('all'). → очищенные слои."""
        names = list(LAYERS) if (name or "").strip().lower() == "all" else [resolve_layer(name)]
        for layer in names:
            self.layers[layer].clear()
        return names

    # ----- сведения -----

    def stats(self) -> dict:
        return {name: layer.stats() for name, layer in self.layers.items()}

    def dump(self) -> dict:
        return {name: layer.dump() for name, layer in self.layers.items()}
