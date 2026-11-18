"""Таблицы правил, схема ответа и разбор операций."""

import pytest

from memory.rules import (
    LAYERS, OPERATIONS_SCHEMA, READ_RULES, WRITE_RULES, MemoryFormatError, check_evidence, normalize_field,
    parse_operations,
)
from memory.evidence import numbers_supported, quote_in, quote_is_question


def test_rules_described_for_every_layer():
    assert set(WRITE_RULES) == set(LAYERS) == set(READ_RULES)
    assert all(WRITE_RULES[layer] and READ_RULES[layer] for layer in LAYERS)


def test_schema_does_not_allow_model_to_write_short_term():
    item = OPERATIONS_SCHEMA["properties"]["operations"]["items"]
    assert item["properties"]["layer"]["enum"] == ["working", "long_term"]
    assert "layer" in item["required"]


def test_parse_operations():
    assert parse_operations('```json\n{"operations": []}\n```') == []
    with pytest.raises(MemoryFormatError):
        parse_operations("не json")
    with pytest.raises(MemoryFormatError):
        parse_operations('{"ops": []}')


def test_evidence_helpers():
    assert quote_in("никакого Docker — ни для разработки", "Никакого Docker — ни для разработки, ни для запуска.")
    assert not quote_in("ab", "abc")                                  # слишком короткая цитата
    assert numbers_supported("18 500 000 VND", "бюджет 18500000")
    assert not numbers_supported("21 ноября", "до 20 ноября")
    assert quote_is_question("Какой фреймворк посоветуешь", "Привет. Какой фреймворк посоветуешь?")
    assert not quote_is_question("Давай pytest", "Давай pytest. А дальше?")


def test_check_evidence_by_kind():
    base = {"value": "pytest", "evidence": "Давай pytest", "proposal_evidence": "советую pytest"}
    assert check_evidence({**base, "kind": "accepted_proposal"}, "Давай pytest.", "Я советую pytest.") is None
    assert check_evidence({**base, "kind": "accepted_proposal"}, "Давай pytest.", None) == "нет цитаты предложения ассистента"
    assert check_evidence({**base, "kind": "assistant_proposal"}, "Давай pytest.", "Иное.") == "цитаты нет в сообщении ассистента"
    assert check_evidence({**base, "kind": "boss"}, "", "") == "неизвестный kind 'boss'"


def test_field_aliases():
    assert normalize_field("Constraint") == "constraints"
    assert normalize_field("next step") == "next_step"
    assert normalize_field("goal") == "goal"
