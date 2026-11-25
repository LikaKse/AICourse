"""Преобразование MCP-инструментов в FunctionDeclaration Gemini.

MCP отдаёт JSON Schema (draft 2020-12). Gemini принимает подмножество JSON Schema,
поэтому убираем служебные ключи вроде $schema и нормализуем типы.
"""

from __future__ import annotations

from typing import Any, Iterable

from google.genai import types

# Ключи, которые не нужны/не поддерживаются в parameters_json_schema Gemini.
_DROP_KEYS = {
    # Служебные ключи JSON Schema.
    "$schema",
    "$id",
    "$comment",
    "$anchor",
    "$defs",
    "$ref",
    "examples",
    "deprecated",
    # Ключи, которых нет в поддерживаемом Gemini подмножестве Schema.
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "uniqueItems",
    "minContains",
    "maxContains",
    "patternProperties",
    "propertyNames",
    "dependentRequired",
    "dependentSchemas",
    "if",
    "then",
    "else",
    "not",
    "allOf",
    "oneOf",
    "contentEncoding",
    "contentMediaType",
}


def sanitize_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Возвращает безопасную копию JSON Schema для Gemini."""
    cleaned = _clean(schema or {})
    if not isinstance(cleaned, dict):
        cleaned = {}
    if cleaned.get("type") != "object":
        cleaned["type"] = "object"
    properties = cleaned.get("properties")
    if not isinstance(properties, dict):
        cleaned["properties"] = {}
    required = cleaned.get("required")
    if isinstance(required, list):
        cleaned["required"] = [x for x in required if isinstance(x, str)]
    return cleaned


def _clean(node: Any) -> Any:
    if isinstance(node, dict):
        result: dict[str, Any] = {}
        for key, value in node.items():
            if key in _DROP_KEYS:
                continue
            if key == "type" and isinstance(value, list):
                non_null = [x for x in value if x != "null"]
                if len(non_null) == 1:
                    result["type"] = non_null[0]
                if "null" in value:
                    result["nullable"] = True
                continue
            if key == "const":
                result["enum"] = [value]
                continue
            result[key] = _clean(value)
        return {k: v for k, v in result.items() if v is not None}
    if isinstance(node, list):
        return [_clean(x) for x in node]
    return node


def unique_public_name(name: str, used: Iterable[str], prefix: str = "mcp_fs_") -> str:
    """Не даёт MCP-инструменту перекрыть локальный инструмент с тем же именем."""
    used_set = set(used)
    if name not in used_set:
        return name
    candidate = f"{prefix}{name}"
    index = 2
    while candidate in used_set:
        candidate = f"{prefix}{name}_{index}"
        index += 1
    return candidate


def to_gemini_declaration(
    tool: Any,
    *,
    public_name: str | None = None,
    source_label: str = "MCP filesystem",
) -> types.FunctionDeclaration:
    """MCP Tool (name/description/input_schema) → types.FunctionDeclaration."""
    name = public_name or tool.name
    description = (tool.description or tool.name).strip()
    if source_label:
        description = f"[{source_label}] {description}"
    return types.FunctionDeclaration(
        name=name,
        description=description,
        parameters_json_schema=sanitize_schema(getattr(tool, "input_schema", None)),
    )
