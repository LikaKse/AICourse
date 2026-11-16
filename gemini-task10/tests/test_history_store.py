"""Сохранение и восстановление полной истории и summary после перезапуска."""

import json

import pytest

import config
from context_manager import ContextManager, StateError, content_text, text_content
from history_store import HistoryLoadError, HistoryStore, SummaryState, SummaryStore
from test_context_manager import RecordingSummarizer, fill


def test_history_roundtrip(tmp_path):
    store = HistoryStore(tmp_path / "h.json")
    history = [text_content("user", "Привет, я Лика"), text_content("model", "Здравствуйте!")]
    store.save(history)
    loaded = store.load()
    assert [c.role for c in loaded] == ["user", "model"]
    assert [content_text(c) for c in loaded] == ["Привет, я Лика", "Здравствуйте!"]


def test_missing_files_mean_empty_state(tmp_path):
    cm = ContextManager.from_dir(tmp_path / "state")
    assert cm.full_history == [] and cm.summary.text == "" and cm.summary.summarized_upto == 0


def test_summary_and_history_survive_restart(tmp_path):
    cm = ContextManager.from_dir(tmp_path, summarizer=RecordingSummarizer())
    fill(cm, 13)                                    # 26 сообщений, 2 версии summary
    text, upto = cm.summary.text, cm.summary.summarized_upto

    restarted = ContextManager.from_dir(tmp_path, summarizer=RecordingSummarizer())
    assert restarted.summary.text == text
    assert restarted.summary.summarized_upto == upto == 20
    assert restarted.summary.version == 2
    assert len(restarted.full_history) == 26
    assert [content_text(c) for c in restarted.active] == [content_text(c) for c in cm.active]
    # после перезапуска сжатие не повторяется
    assert restarted.maybe_compress() is None
    assert restarted.summarizer.calls == []


def test_summary_stored_separately_from_full_history(tmp_path):
    cm = ContextManager.from_dir(tmp_path, summarizer=RecordingSummarizer())
    fill(cm, 8)
    history_file = json.loads((tmp_path / config.FULL_HISTORY_FILE).read_text(encoding="utf-8"))
    summary_file = json.loads((tmp_path / config.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert len(history_file["history"]) == 16            # сжатые сообщения остались в полной истории
    assert "history" not in summary_file
    assert summary_file["summarized_upto"] == 10 and summary_file["text"]


def test_corrupted_history_raises_clear_error(tmp_path):
    path = tmp_path / "h.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(HistoryLoadError, match="некорректный JSON"):
        HistoryStore(path).load()
    backup = HistoryStore(path).backup_corrupted()
    assert backup.exists() and not path.exists()


def test_invalid_summary_file(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"format_version": 1, "text": "", "summarized_upto": 4}), encoding="utf-8")
    with pytest.raises(HistoryLoadError, match="текст summary пуст"):
        SummaryStore(path).load()
    path.write_text(json.dumps({"format_version": 99}), encoding="utf-8")
    with pytest.raises(HistoryLoadError, match="версия"):
        SummaryStore(path).load()


def test_inconsistent_state_detected(tmp_path):
    HistoryStore(tmp_path / config.FULL_HISTORY_FILE).save([text_content("user", "a"), text_content("model", "b")])
    SummaryStore(tmp_path / config.SUMMARY_FILE).save(SummaryState(text="x", summarized_upto=10, version=1))
    with pytest.raises(StateError, match="не согласованы"):
        ContextManager.from_dir(tmp_path)


def test_api_key_not_written_to_files(tmp_path, monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "SECRET-KEY-123456")
    store = HistoryStore(tmp_path / "h.json")
    store.save([text_content("user", "мой ключ SECRET-KEY-123456")])
    assert "SECRET-KEY-123456" not in (tmp_path / "h.json").read_text(encoding="utf-8")
