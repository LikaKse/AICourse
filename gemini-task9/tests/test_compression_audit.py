"""Офлайн-аудит summary: сохранённые, потерянные и искажённые детали, авторство, версии (без API)."""

import copy
import hashlib
import json

import pytest

import compression_audit as ca
import config
import experiments
from history_store import HistoryStore, SummaryState, SummaryStore


@pytest.fixture(scope="module")
def dialogue():
    return experiments.load_dialogue()


@pytest.fixture(scope="module")
def catalog():
    return ca.load_catalog()


def detail(catalog, did):
    return next(d for d in catalog["details"] if d["id"] == did)


def item(catalog, aid):
    return next(a for a in catalog["attributions"] if a["id"] == aid)


# ----- каталог согласован с диалогом -----

def test_catalog_backed_by_dialogue(catalog, dialogue):
    ca.validate_catalog(catalog, dialogue)
    assert len(catalog["details"]) >= 30
    kinds = {d["kind"] for d in catalog["details"]}
    assert {"intermediate", "reason", "preference", "alternative", "constraint", "advice"} <= kinds


@pytest.mark.parametrize("mutate,error", [
    (lambda c: c["details"][0]["evidence"][0].update(quote="Меня зовут Марина"), "не найдена"),
    (lambda c: c["details"][0]["evidence"][0].update(message=500), "нет в диалоге"),
    (lambda c: c["attributions"][2].update(pattern=r"docker"), "упоминает пользователь"),
    (lambda c: c["attributions"][2].update(assistant_messages=[11]), "нет «"),
    (lambda c: c["attributions"][7].update(user_acceptance_quote="Возьмём requests"), "принимающей"),
])
def test_invalid_catalog_rejected(catalog, dialogue, mutate, error):
    bad = copy.deepcopy(catalog)
    mutate(bad)
    with pytest.raises(ca.AuditError, match=error):
        ca.validate_catalog(bad, dialogue)


# ----- одна деталь -----

def test_preserved_lost_distorted(catalog):
    summary = "- Время запуска: **09:30** (заменило прежнее 08:00)\n- Бюджет: 15 000 000 VND\n- Имя: Марина"
    assert ca.check_detail(detail(catalog, "D29"), summary)["status"] == "preserved"
    assert ca.check_detail(detail(catalog, "D21"), summary)["status"] == "preserved"
    assert ca.check_detail(detail(catalog, "D30"), summary)["status"] == "lost"         # причина переноса
    assert ca.check_detail(detail(catalog, "D08"), summary)["status"] == "preserved"    # 15 млн есть
    distorted = ca.check_detail(detail(catalog, "D26"), summary)                        # но выдан за итог
    assert distorted["status"] == "distorted" and "предварительный" in distorted["note"]
    assert ca.check_detail(detail(catalog, "D01"), summary)["status"] == "distorted"    # другое имя


def test_old_value_next_to_final_value_is_not_distortion(catalog):
    summary = "- Хотела запуск в 08:00.\n- Перенесли на 09:30."
    assert ca.check_detail(detail(catalog, "D29"), summary)["status"] == "preserved"
    assert ca.check_detail(detail(catalog, "D29"), "- Запуск: 08:00")["status"] == "distorted"


def test_residence_vs_weather_city(catalog):
    weather_only = "- Город для прогноза погоды: Нячанг."
    assert ca.check_detail(detail(catalog, "D05"), weather_only)["status"] == "lost"
    assert ca.check_detail(detail(catalog, "D06"), weather_only)["status"] == "preserved"
    assert ca.check_detail(detail(catalog, "D05"), "- Пользователь живёт в Нячанге")["status"] == "preserved"


def test_line_and_text_rules(catalog):
    d17 = detail(catalog, "D17")
    assert ca.check_detail(d17, "- Модель flash-lite: важнее скорость и цена")["status"] == "preserved"
    assert ca.check_detail(d17, "- Модель flash-lite\n- скорость\n- цена")["status"] == "lost"   # не в одной строке
    d18 = detail(catalog, "D18")
    assert ca.check_detail(d18, "- рассматривали CSV\n- и SQLite")["status"] == "preserved"
    assert ca.check_detail(d18, "- рассматривали CSV")["status"] == "lost"


# ----- авторство -----

OLD_SUMMARY_LINE = ("  * Ограничения: Docker не использовать вообще (ни для разработки, ни для запуска); "
                    "используется виртуальное окружение `venv` и системный планировщик (`launchd`).")


def test_assistant_proposal_recorded_as_decision(catalog):
    res = ca.check_attribution(item(catalog, "A03"), OLD_SUMMARY_LINE)
    assert res["status"] == "as_decision"
    assert ca.check_attribution(item(catalog, "A04"), OLD_SUMMARY_LINE)["status"] == "as_decision"


def test_attribution_statuses(catalog):
    assert ca.check_attribution(item(catalog, "A03"), "- Ассистент предложил venv")["status"] == "ok_assistant"
    assert ca.check_attribution(item(catalog, "A06"),
                                "- Структура проекта: src/, tests/, scripts/")["status"] == "unattributed"
    assert ca.check_attribution(item(catalog, "A08"), "- Выбрана библиотека httpx")["status"] == "ok_user_accepted"
    assert ca.check_attribution(item(catalog, "A01"), "- ничего о погодных API")["status"] == "absent"
    # худший статус по всем строкам
    both = "- Ассистент предложил launchd\n- Для launchd используется StartCalendarInterval"
    assert ca.check_attribution(item(catalog, "A04"), both)["status"] == "as_decision"


# ----- аудит целиком -----

def test_versions_show_where_detail_was_lost(catalog):
    versions = [
        {"version": 1, "end": 10, "text": "- Лика"},
        {"version": 2, "end": 20, "text": "- Лика\n- CSV, SQLite рассматривали; выбран JSON"},
        {"version": 3, "end": 30, "text": "- Лика\n- выбран JSON"},
        {"version": 4, "end": 40, "text": "- Лика\n- выбран JSON"},
    ]
    audit = ca.audit_summary(catalog, versions[-1]["text"], summarized_upto=40, versions=versions)
    rows = {r["id"]: r for r in audit["details"]}
    d18 = rows["D18"]                                           # CSV/SQLite из сообщения 17
    assert [h["status"] for h in d18["versions"]] == [None, "preserved", "lost", "lost"]
    assert d18["first_problem_version"] == {"version": 3, "status": "lost", "after_preserved": True}
    assert rows["D01"]["first_problem_version"] is None         # имя сохранено во всех версиях
    d30 = rows["D30"]                                           # причина переноса (сообщение 33) — не попала
    assert d30["first_problem_version"] == {"version": 4, "status": "lost", "after_preserved": False}
    report = ca.markdown_report(audit)
    assert "| v1 | v2 | v3 | v4 | Проблема с |" in report


def test_not_compressed_and_not_comparable(catalog):
    stored = [("user", "Привет! Меня зовут Марина, я хочу сделать небольшой учебный проект на Python.")]
    audit = ca.audit_summary(catalog, "- Имя: Марина\n- учебный проект на Python", summarized_upto=20,
                             mismatched_messages={1}, stored_messages=stored)
    rows = {r["id"]: r for r in audit["details"]}
    assert rows["D01"]["status"] == "not_comparable"            # имя в прогоне было другое
    assert rows["D02"]["status"] == "preserved"                 # цитата про Python в прогоне не менялась
    assert rows["D29"]["status"] == "not_compressed"            # сообщение 33 ещё не сжималось
    assert audit["totals"]["checked"] == audit["totals"]["preserved"] + audit["totals"]["lost"] + audit["totals"]["distorted"]


# ----- сохранённый прогон -----

def _write_run(tmp_path, dialogue, summary_text, *, versions=None, stored_dialogue=None):
    exp = {"started_at": "2026-11-13T01:52:10", "model": config.MODEL, "final_summary": summary_text,
           "context": {"summarized_messages": 40}}
    if versions is not None:
        exp["summary_versions"] = versions
    (tmp_path / "experiment_with_compression.json").write_text(json.dumps(exp, ensure_ascii=False), encoding="utf-8")
    HistoryStore(tmp_path / "state_B" / config.FULL_HISTORY_FILE).save(stored_dialogue or dialogue)
    SummaryStore(tmp_path / "state_B" / config.SUMMARY_FILE).save(
        SummaryState(text=summary_text, summarized_upto=40, version=4,
                     updates=[{"version": 4, "start": 30, "end": 40, "messages": 10}]))


def _digest(path):
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(path.rglob("*.json"))}


def test_old_run_without_versions_on_old_dialogue(tmp_path, dialogue):
    from context_manager import text_content
    old = [text_content("user", content.parts[0].text.replace("Лика", "Марина")) if i < 2 and content.role == "user"
           else (text_content("model", content.parts[0].text.replace("Лика", "Марина")) if i < 2 else content)
           for i, content in enumerate(dialogue)]
    _write_run(tmp_path, dialogue, OLD_SUMMARY_LINE + "\n  * Имя пользователя: Марина.", stored_dialogue=old)
    before = _digest(tmp_path)
    audit = ca.audit_saved_run(tmp_path, dialogue)
    assert _digest(tmp_path) == before                          # аудит только читает
    assert any("промежуточных версий" in w for w in audit["warnings"])
    assert any("[1, 2]" in w for w in audit["warnings"])
    rows = {r["id"]: r for r in audit["details"]}
    assert rows["D01"]["status"] == "not_comparable"
    assert rows["D12"]["status"] == "preserved" and rows["D30"]["status"] == "lost"
    assert "A03" in audit["attribution_issues"]
    assert audit["versions_available"] == []


def test_new_run_with_versions(tmp_path, dialogue):
    versions = [{"version": 1, "start": 0, "end": 10, "text": "- Лика живёт в Нячанге"},
                {"version": 2, "start": 10, "end": 20, "text": "- Лика"}]
    _write_run(tmp_path, dialogue, "- Лика", versions=versions)
    audit = ca.audit_saved_run(tmp_path, dialogue)
    assert audit["versions_available"] == [1, 2]
    assert not any("другой версии" in w for w in audit["warnings"])
    d05 = {r["id"]: r for r in audit["details"]}["D05"]
    assert d05["first_problem_version"] == {"version": 2, "status": "lost", "after_preserved": True}


def test_missing_run_gives_clear_error(tmp_path, dialogue):
    with pytest.raises(ca.AuditError, match="Нет результатов эксперимента B"):
        ca.audit_saved_run(tmp_path, dialogue)


def test_fingerprint_changes_with_dialogue(dialogue):
    from context_manager import text_content
    changed = [text_content("user", "другое")] + list(dialogue[1:])
    assert ca.dialogue_fingerprint(dialogue) != ca.dialogue_fingerprint(changed)
    assert ca.mismatched_messages(ca.dialogue_texts(changed), dialogue) == {1}


# ----- регрессии из финального аудита (прогон 13.11.2026) -----

BUDGET_V2 = "* **Бюджет:** Предварительный бюджет на платные API и хостинг — 15 000 000 VND (требует уточнения)."
MODEL_V2 = ("* **Технологии и модели:** Для анализа ответов используется модель `gemini-3.5-flash-lite` "
            "(выбрана вместо `gemini-3.5-flash` из-за скорости и цены).")
ENV_LINE = ("* **Окружение и инструменты:** Операционная система — macOS (инструкции для терминала macOS, zsh, "
            "Homebrew). Без использования Docker (ограничение: не использовать Docker вообще, ни для разработки, "
            "ни для запуска). Для HTTP-запросов выбрана библиотека `httpx`. API-ключи хранятся в файле `.env` "
            "(добавлен в `.gitignore`), читаются через `python-dotenv`.")


def test_d15_not_matched_by_utochneniya(catalog):
    d15 = detail(catalog, "D15")
    assert ca.check_detail(d15, BUDGET_V2 + "\n" + MODEL_V2)["status"] == "lost"   # «уточнения» ≠ «точнее»
    assert ca.check_detail(d15, "- Полная модель точнее на сложных задачах")["status"] == "preserved"
    assert ca.check_detail(d15, "- Точность выше у gemini-3.5-flash")["status"] == "preserved"


def test_d15_history_never_preserved(catalog):
    versions = [{"version": 1, "end": 10, "text": "- Лика"},
                {"version": 2, "end": 20, "text": BUDGET_V2 + "\n" + MODEL_V2},
                {"version": 3, "end": 30, "text": MODEL_V2},
                {"version": 4, "end": 40, "text": MODEL_V2}]
    audit = ca.audit_summary(catalog, MODEL_V2, summarized_upto=40, versions=versions)
    d15 = {r["id"]: r for r in audit["details"]}["D15"]
    assert [h["status"] for h in d15["versions"]] == [None, "lost", "lost", "lost"]
    assert d15["first_problem_version"] == {"version": 2, "status": "lost", "after_preserved": False}


def test_summary_sentences_split():
    sentences = ca.summary_sentences(ENV_LINE)
    assert len(sentences) == 4
    assert sentences[0].endswith("zsh, homebrew).")
    assert "gemini-3.5-flash-lite" in ca.summary_sentences(MODEL_V2)[0]     # точки внутри имени не режут


def test_a02_long_line_is_unattributed_not_decision(catalog):
    res = ca.check_attribution(item(catalog, "A02"), ENV_LINE)
    assert res["status"] == "unattributed"
    assert res["lines"] == ["окружение и инструменты: операционная система — macos (инструкции для терминала macos, "
                            "zsh, homebrew)."]
    # в той же строке: .env подан как решение, httpx — принят пользователем
    assert ca.check_attribution(item(catalog, "A07"), ENV_LINE)["status"] == "as_decision"
    assert ca.check_attribution(item(catalog, "A08"), ENV_LINE)["status"] == "ok_user_accepted"
