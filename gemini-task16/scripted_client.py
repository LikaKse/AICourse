"""Скриптованный «клиент модели» для бесплатной демонстрации.

Он не обращается к Gemini. Его задача — детерминированно провести тот же путь, что и
настоящая модель: MCP read_text_file → локальный calculate → финальный ответ.
"""

from __future__ import annotations

import re

from google.genai import types

import config


class ScriptedGeminiClient:
    """Минимальный async-клиент с интерфейсом client.aio.models.generate_content."""

    def __init__(self, file_path: str | None = None) -> None:
        self.file_path = file_path or str(config.DEMO_FILE)
        self.step = 0
        self.aio = self
        self.models = self
        self.requests: list[dict] = []

    async def generate_content(self, *, model: str, contents, config: types.GenerateContentConfig):
        self.requests.append({"model": model, "contents": contents, "config": config})
        if self.step == 0:
            self.step += 1
            return _call_response(
                "read_text_file",
                {"path": self.file_path},
                call_id="scripted-read",
                signature=b"scripted-signature-read",
            )
        if self.step == 1:
            self.step += 1
            content = _last_function_response_text(contents)
            numbers = _parse_numbers(content)
            expression = (
                f"{numbers['quantity']} * {numbers['unit_price']} "
                f"* (1 + {numbers['markup_percent']} / 100)"
            )
            return _call_response(
                "calculate",
                {"expression": expression},
                call_id="scripted-calc",
                signature=b"scripted-signature-calc",
            )
        self.step += 1
        return _text_response(
            "Итоговая стоимость: 330 000. "
            "Файл прочитан через MCP read_text_file, расчёт выполнен локальным calculate."
        )


def _parse_numbers(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for key in ("quantity", "unit_price", "markup_percent"):
        match = re.search(rf"{key}\s*=\s*(\d+)", text)
        if not match:
            raise AssertionError(f"В ответе MCP нет {key}: {text!r}")
        result[key] = int(match.group(1))
    return result


def _last_function_response_text(contents) -> str:
    for content in reversed(contents):
        for part in reversed(getattr(content, "parts", []) or []):
            response = getattr(part, "function_response", None)
            if not response:
                continue
            payload = response.response or {}
            if isinstance(payload, dict):
                if isinstance(payload.get("content"), str):
                    return payload["content"]
                result = payload.get("result")
                if isinstance(result, dict) and isinstance(result.get("content"), str):
                    return result["content"]
    raise AssertionError("В истории нет FunctionResponse с содержимым файла.")


def _wrap(parts) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(role="model", parts=parts),
            finish_reason="STOP",
        )],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=5,
        ),
    )


def _call_response(name: str, args: dict, *, call_id: str, signature: bytes):
    return _wrap([
        types.Part(
            function_call=types.FunctionCall(id=call_id, name=name, args=args),
            thought_signature=signature,
        )
    ])


def _text_response(text: str):
    return _wrap([types.Part(text=text, thought_signature=b"scripted-signature-final")])
