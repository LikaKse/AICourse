"""Сжатие: создание summary, объединение, последние 6 сообщений, отсутствие повторного сжатия."""

import pytest

from context_manager import (
    ContextManager, SummarizationError, SummaryOutput, content_text, text_content,
)


class RecordingSummarizer:
    """Фейковый summarizer: запоминает, что ему передали, и «объединяет» summary."""

    def __init__(self, fail_times: int = 0):
        self.calls = []
        self.fail_times = fail_times

    def __call__(self, previous, messages):
        self.calls.append((previous, [content_text(m) for m in messages]))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise SummarizationError("503 UNAVAILABLE")
        new = " ".join(content_text(m) for m in messages)
        return SummaryOutput(text=(previous + " | " if previous else "") + new, meta={"n": len(messages)})


def pairs(n, start=0):
    return [(text_content("user", f"u{i}"), text_content("model", f"m{i}")) for i in range(start, start + n)]


def fill(cm, n, start=0):
    events = []
    for u, m in pairs(n, start):
        events += cm.add_exchange(u, m)
    return events


def test_no_compression_below_threshold():
    s = RecordingSummarizer()
    cm = ContextManager(summarizer=s)
    fill(cm, 7)                      # 14 сообщений < 16
    assert s.calls == []
    assert cm.summary.text == ""
    assert len(cm.active) == 14


def test_summary_created_at_threshold_and_last_six_kept():
    s = RecordingSummarizer()
    cm = ContextManager(summarizer=s)
    events = fill(cm, 8)             # 16 сообщений → сжатие
    assert len(s.calls) == 1
    assert len(events) == 1 and events[0].ok
    assert (events[0].start, events[0].end, events[0].messages) == (0, 10, 10)
    assert s.calls[0][1] == [f"{r}{i}" for i in range(5) for r in ("u", "m")]
    assert cm.summary.summarized_upto == 10 and cm.summary.version == 1
    # последние 6 сообщений — те же объекты, без изменений
    assert len(cm.active) == 6
    assert cm.active == cm.full_history[-6:]
    assert [content_text(c) for c in cm.active] == ["u5", "m5", "u6", "m6", "u7", "m7"]
    # полная история не потеряна
    assert len(cm.full_history) == 16


def test_merge_previous_and_new_summary():
    s = RecordingSummarizer()
    cm = ContextManager(summarizer=s)
    fill(cm, 13)                     # 26 сообщений → два сжатия
    assert len(s.calls) == 2
    first_summary = s.calls[1][0]
    assert first_summary.startswith("u0 m0")          # во второй вызов передано предыдущее summary
    assert s.calls[1][1][0] == "u5"                    # и только новые сообщения
    assert cm.summary.version == 2
    assert cm.summary.text.startswith(first_summary + " | ")
    assert [(u["start"], u["end"]) for u in cm.summary.updates] == [(0, 10), (10, 20)]
    # каждая версия summary хранится целиком — видно, на каком шаге теряются детали
    assert [u["text"] for u in cm.summary.updates] == [first_summary, cm.summary.text]


def test_no_repeated_summarization():
    s = RecordingSummarizer()
    cm = ContextManager(summarizer=s)
    fill(cm, 25)                     # 50 сообщений
    for _ in range(5):               # повторные вызовы без новых сообщений ничего не делают
        assert cm.maybe_compress() is None
    assert len(s.calls) == 4
    # каждый фрагмент суммирован ровно один раз, фрагменты не пересекаются и идут подряд
    ranges = [(u["start"], u["end"]) for u in cm.summary.updates]
    assert ranges == [(0, 10), (10, 20), (20, 30), (30, 40)]
    seen = [m for _, msgs in s.calls for m in msgs]
    assert len(seen) == len(set(seen)) == 40
    assert len(cm.active) == 10 and cm.active[-6:] == cm.full_history[-6:]


def test_request_contains_summary_and_recent_not_full_history():
    cm = ContextManager(summarizer=RecordingSummarizer())
    fill(cm, 8)
    contents = cm.build_contents(text_content("user", "вопрос"))
    texts = [content_text(c) for c in contents]
    assert texts[0].startswith("Краткое содержание предыдущей части диалога:")
    assert contents[1].role == "model"
    assert texts[2:] == ["u5", "m5", "u6", "m6", "u7", "m7", "вопрос"]
    assert "u0" not in texts                      # сжатое сообщение не передаётся как есть
    assert len(contents) == 2 + 6 + 1


def test_disabled_compression_keeps_full_history_in_request():
    s = RecordingSummarizer()
    cm = ContextManager(summarizer=s, compression_enabled=False)
    fill(cm, 25)
    assert s.calls == []
    assert len(cm.build_contents()) == 50


def test_failed_summarization_leaves_state_unchanged_and_retries_once_per_call():
    s = RecordingSummarizer(fail_times=1)
    cm = ContextManager(summarizer=s)
    events = fill(cm, 8)
    assert len(events) == 1 and not events[0].ok and "503" in events[0].error
    assert cm.summary.summarized_upto == 0 and cm.summary.text == ""
    assert len(cm.active) == 16                  # ничего не потеряно
    events = fill(cm, 1, start=8)                # следующая пара — одна повторная попытка
    assert len(events) == 1 and events[0].ok
    assert (events[0].start, events[0].end) == (0, 12)
    assert len(cm.active) == 6
    assert len(s.calls) == 2


def test_empty_summary_is_rejected():
    cm = ContextManager(summarizer=lambda prev, msgs: SummaryOutput(text="  "))
    events = fill(cm, 8)
    assert not events[0].ok and cm.summary.summarized_upto == 0


def test_compression_without_summarizer_raises():
    cm = ContextManager()
    with pytest.raises(RuntimeError):
        fill(cm, 8)


def test_invalid_settings():
    with pytest.raises(ValueError):
        ContextManager(summary_every=0)
    with pytest.raises(ValueError):
        ContextManager(keep_last=-1)
