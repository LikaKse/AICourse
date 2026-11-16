"""Краткосрочная память: запись пар, окно чтения, хранение и очистка."""

import json

import pytest

from memory import MemoryLoadError, ShortTermMemory
from memory.storage import content_text, text_content


def add(st, n):
    for i in range(1, n + 1):
        st.add_exchange(text_content("user", f"вопрос {i}"), text_content("model", f"ответ {i}"))


@pytest.mark.parametrize("window, expected", [(0, []), (2, ["вопрос 3", "ответ 3"]),
                                              (3, ["вопрос 3", "ответ 3"]),     # ответ в начале окна отброшен
                                              (6, ["вопрос 1", "ответ 1", "вопрос 2", "ответ 2", "вопрос 3", "ответ 3"]),
                                              (50, ["вопрос 1", "ответ 1", "вопрос 2", "ответ 2", "вопрос 3", "ответ 3"])])
def test_window_controls_what_is_read(window, expected):
    st = ShortTermMemory(window=window)
    add(st, 3)
    assert [content_text(c) for c in st.prompt_contents()] == expected
    assert st.stats()["stored_messages"] == 6          # хранится всё, окно влияет только на чтение


def test_negative_window_rejected():
    with pytest.raises(ValueError):
        ShortTermMemory(window=-1)


def test_add_requires_user_model_pair():
    with pytest.raises(ValueError):
        ShortTermMemory().add_exchange(text_content("model", "a"), text_content("user", "b"))


def test_persistence_and_counter_survives_clear(tmp_path):
    path = tmp_path / "short_term.json"
    st = ShortTermMemory(path, window=4)
    add(st, 2)
    assert st.next_message_no() == 5 and st.last_assistant_text() == "ответ 2"

    again = ShortTermMemory(path, window=4)
    assert [content_text(c) for c in again.messages] == ["вопрос 1", "ответ 1", "вопрос 2", "ответ 2"]
    again.clear()
    reloaded = ShortTermMemory(path)
    assert reloaded.is_empty() and reloaded.prompt_contents() == []
    assert reloaded.next_message_no() == 5             # номера сообщений не начинаются заново
    assert reloaded.last_assistant_text() is None
    add(reloaded, 1)
    assert reloaded.dump()["messages"][0]["no"] == 5


def test_file_header_and_wrong_layer(tmp_path):
    path = tmp_path / "short_term.json"
    add(ShortTermMemory(path), 1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["layer"] == "short_term" and payload["format_version"] == 1
    payload["layer"] = "long_term"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MemoryLoadError, match="принадлежит слою"):
        ShortTermMemory(path)


def test_corrupted_file_raises(tmp_path):
    path = tmp_path / "short_term.json"
    path.write_text("{не json", encoding="utf-8")
    with pytest.raises(MemoryLoadError, match="некорректный JSON"):
        ShortTermMemory(path)
