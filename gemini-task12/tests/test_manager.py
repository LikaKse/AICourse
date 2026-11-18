"""MemoryManager: маршрутизация записи, порядок промпта, независимая очистка, перезапуск."""

import pytest

import experiments
from memory import MemoryManager
from memory.storage import content_text, text_content

USER = "Меня зовут Лика. Нужно сделать отчёт за неделю."


def ops():
    return [
        {"layer": "long_term", "op": "set", "key": "name", "value": "Лика", "kind": "user", "category": "name",
         "evidence": "Меня зовут Лика"},
        {"layer": "working", "op": "set", "key": "goal", "value": "отчёт за неделю", "kind": "user",
         "category": "other", "evidence": "Нужно сделать отчёт за неделю"},
    ]


def filled(tmp_path, window=6):
    m = MemoryManager(tmp_path, window=window)
    m.apply_operations(ops(), user_text=USER, assistant_text=None, message_no=1)
    m.record_exchange(text_content("user", USER), text_content("model", "Принято."))
    return m


def test_prompt_order_long_term_working_short_term(tmp_path):
    m = filled(tmp_path)
    assert [b.layer for b in m.prompt_blocks()] == ["long_term", "working", "short_term"]
    texts = [content_text(c) for c in m.build_contents(text_content("user", "Вопрос?"))]
    assert texts[0].startswith("Долговременная память") and texts[2].startswith("Рабочая память")
    assert texts[4:] == [USER, "Принято.", "Вопрос?"]


def test_empty_layers_are_not_in_prompt():
    m = MemoryManager()
    assert m.prompt_blocks() == [] and m.context_text() == ""


def test_routing_unknown_and_short_term_layers_rejected(tmp_path):
    m = MemoryManager(tmp_path)
    out = m.apply_operations([{**ops()[0], "layer": "short_term"}, {**ops()[0], "layer": "episodic"}],
                             user_text=USER, assistant_text=None, message_no=1)
    assert [r["rejection"] for r in out["unrouted"]] == ["краткосрочную память пишет только код",
                                                         "неизвестный слой 'episodic'"]
    assert m.long_term.state.facts == {} and not any(tmp_path.iterdir())


@pytest.mark.parametrize("layer, name", [("short", "short_term"), ("working", "working"), ("long", "long_term")])
def test_clear_changes_exactly_one_file(tmp_path, layer, name):
    m = filled(tmp_path)
    before = experiments.file_digests(tmp_path)
    assert m.clear(layer) == [name]
    after = experiments.file_digests(tmp_path)
    assert [k for k in before if before[k] != after[k]] == [name]
    reloaded = MemoryManager(tmp_path)
    assert reloaded.layers[name].prompt_contents() == []
    for other in {"short_term", "working", "long_term"} - {name}:
        assert reloaded.layers[other].prompt_contents()


def test_clear_all_and_unknown_layer(tmp_path):
    m = filled(tmp_path)
    assert m.clear("all") == ["short_term", "working", "long_term"]
    assert MemoryManager(tmp_path).prompt_blocks() == []
    with pytest.raises(ValueError):
        m.clear("episodic")


def test_restart_restores_all_layers(tmp_path):
    m = filled(tmp_path, window=2)
    again = MemoryManager(tmp_path, window=2)
    assert again.dump() == m.dump() and again.context_text() == m.context_text()


def test_working_memory_never_becomes_long_term(tmp_path):
    m = filled(tmp_path)
    lt_before = m.long_term.dump()
    m.working.finish_task()
    assert m.long_term.dump() == lt_before
    m.clear("working")
    assert MemoryManager(tmp_path).long_term.dump() == lt_before
    assert "отчёт" not in str(lt_before)
