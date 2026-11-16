"""Checkpoint и ветки: независимость, переключение, восстановление, конфликты (fake-клиент, без API)."""

import json
import re

import pytest
from fakes import FakeClient, all_text, echo_responder, facts_json, is_facts_request, request_user_message, text_response

import config
import experiments
from agent import Agent
from branching import (
    BranchChange, BranchConflictError, BranchError, BranchManager, load_scenario, run_scenario,
)
from context_manager import content_text, text_content


@pytest.fixture(scope="module")
def dialogue():
    return experiments.load_dialogue()


@pytest.fixture(scope="module")
def scenario():
    return load_scenario()


def tree_text(path):
    """Весь текст всех файлов каталога (нормализованный) — для проверки утечек."""
    from utils import normalize_text
    return normalize_text("\n".join(p.read_text(encoding="utf-8") for p in path.rglob("*.json")))


def pair(u, m):
    return text_content("user", u), text_content("model", m)


def add(bm, u, m):
    user, model = pair(u, m)
    bm.strategy.before_user_message(user)
    bm.strategy.add_exchange(user, model)


# ----- данные сценария -----

def test_scenario_is_consistent_with_dialogue(dialogue, scenario):
    n = scenario["checkpoint_after"]
    assert n == 30 and n % 2 == 0
    names = [b["name"] for b in scenario["branches"]]
    assert names == ["A", "B"]
    a, b = (br["change"]["user"] for br in scenario["branches"])
    assert a != b
    # Унаследованная часть исходного диалога не содержит прежних значений → конфликтов нет.
    bm_like = [(p, content_text(m)) for br in scenario["branches"] for p in br["change"]["conflict_patterns"]
               for m in dialogue[n:]]
    from utils import normalize_text
    assert not any(re.search(p, normalize_text(t)) for p, t in bm_like)
    # …а до checkpoint прежние значения есть (то есть изменения в ветках действительно меняют факт).
    before = normalize_text(" ".join(content_text(m) for m in dialogue[:n]))
    for br in scenario["branches"]:
        assert any(re.search(p, before) for p in br["change"]["conflict_patterns"])


# ----- checkpoint / ветки / переключение / restore -----

@pytest.mark.parametrize("strategy", config.STRATEGIES)
def test_branches_are_independent_and_switchable(tmp_path, strategy):
    agent = Agent(FakeClient(responder=echo_responder()))
    bm = BranchManager(tmp_path, strategy=strategy, agent=agent)
    add(bm, "Меня зовут Лика", "Привет, Лика")
    cp = bm.checkpoint("cp1")
    assert cp["messages"] == 2

    bm.create_branch("A", "cp1")
    add(bm, "Бюджет 22 000 000 VND", "Принято, 22 млн")
    bm.create_branch("B", "cp1")
    add(bm, "Живу в Дананге", "Принято, Дананг")
    add(bm, "Ещё сообщение B", "ок")

    assert bm.current == "B" and len(bm.strategy.full_history) == 6 and agent.strategy is bm.strategy
    a = bm.switch("A")
    assert bm.current == "A" and len(a.full_history) == 4
    texts_a = " ".join(content_text(c) for c in a.full_history)
    assert "22 000 000" in texts_a and "Дананг" not in texts_a
    b = bm.switch("B")
    texts_b = " ".join(content_text(c) for c in b.full_history)
    assert "Дананг" in texts_b and "22 000 000" not in texts_b
    main = bm.switch("main")
    assert len(main.full_history) == 2

    # Файлы веток не пересекаются.
    assert "дананг" not in tree_text(tmp_path / "branches" / "A")
    assert "22000000" not in tree_text(tmp_path / "branches" / "B")
    assert "дананг" not in tree_text(tmp_path / "checkpoints" / "cp1")
    assert "22000000" not in tree_text(tmp_path / "checkpoints" / "cp1")


def test_requests_of_one_branch_do_not_contain_other_branch(tmp_path):
    client = FakeClient(responder=echo_responder())
    agent = Agent(client)
    bm = BranchManager(tmp_path, strategy="facts", agent=agent)
    add(bm, "Меня зовут Лика", "Привет")
    bm.checkpoint("cp")
    bm.create_branch("A", "cp")
    add(bm, "Бюджет 22 000 000 VND", "ок")
    bm.create_branch("B", "cp")
    add(bm, "Живу в Дананге", "ок")

    start = len(client.requests)
    bm.switch("A")
    agent.ask("Какой бюджет?", remember=False)
    bm.switch("B")
    agent.ask("Где я живу?", remember=False)
    req_a, req_b = client.requests[start:]
    assert "22 000 000" in all_text(req_a["contents"]) and "Дананг" not in all_text(req_a["contents"])
    assert "Дананг" in all_text(req_b["contents"]) and "22 000 000" not in all_text(req_b["contents"])
    assert "Лика" in all_text(req_a["contents"]) and "Лика" in all_text(req_b["contents"])   # общее до checkpoint


def test_state_survives_reopen_of_session(tmp_path):
    agent = Agent(FakeClient(responder=echo_responder()))
    bm = BranchManager(tmp_path, strategy="summary", agent=agent)
    add(bm, "u1", "m1")
    bm.checkpoint("cp")
    bm.create_branch("A", "cp")
    add(bm, "uA", "mA")
    reopened = BranchManager(tmp_path)
    assert reopened.current == "A" and reopened.strategy_name == "summary"
    assert [content_text(c) for c in reopened.strategy.full_history] == ["u1", "m1", "uA", "mA"]
    assert reopened.branches() == ["main", "A"]
    with pytest.raises(BranchError, match="создана для стратегии"):
        BranchManager(tmp_path, strategy="facts")


@pytest.mark.parametrize("strategy", config.STRATEGIES)
def test_restore_to_checkpoint_and_checkpoint_is_immutable(tmp_path, strategy):
    agent = Agent(FakeClient(responder=echo_responder()))
    bm = BranchManager(tmp_path, strategy=strategy, agent=agent)
    add(bm, "u1", "m1")
    bm.checkpoint("cp")
    before = tree_text(tmp_path / "checkpoints" / "cp")
    bm.create_branch("A", "cp")
    for i in range(9):                                  # хватает и для сжатия summary (≥16 сообщений)
        add(bm, f"Бюджет {i} 22 000 000 VND", f"m{i}")
    assert len(bm.strategy.full_history) == 20
    assert tree_text(tmp_path / "checkpoints" / "cp") == before      # checkpoint не изменился

    restored = bm.restore("cp")
    assert agent.strategy is restored
    assert [content_text(c) for c in restored.full_history] == ["u1", "m1"]
    from utils import normalize_text
    for f in restored.state_files:                      # в файлах состояния ветки — снова только checkpoint
        path = tmp_path / "branches" / "A" / f
        if path.is_file():
            assert "22000000" not in normalize_text(path.read_text(encoding="utf-8")), f
    if strategy == "summary":
        assert restored.context.summary.summarized_upto == 0 and restored.context.summary.text == ""
    if strategy == "facts":
        assert all("22" not in f.value for f in restored.facts.facts.values())
    meta = bm.branch_meta("A")
    assert meta["restores"][0]["checkpoint"] == "cp" and meta["changes"] == []
    add(bm, "после restore", "ок")
    assert len(bm.strategy.full_history) == 4


def test_errors(tmp_path):
    bm = BranchManager(tmp_path, strategy="sliding")
    bm.checkpoint("cp")
    with pytest.raises(BranchError, match="неизменяем"):
        bm.checkpoint("cp")
    with pytest.raises(BranchError, match="не найден"):
        bm.create_branch("A", "nope")
    bm.create_branch("A", "cp")
    with pytest.raises(BranchError, match="уже существует"):
        bm.create_branch("A", "cp")
    with pytest.raises(BranchError, match="нет"):
        bm.switch("Z")
    with pytest.raises(BranchError, match="Недопустимое имя"):
        bm.create_branch("../evil", "cp")
    with pytest.raises(BranchError, match="Недопустимое имя"):
        bm.checkpoint("a b")


# ----- конфликты при продолжении исходного диалога -----

def budget_extractor(contents, cfg):
    """Фейковое извлечение: любая сумма в VND — ключ budget; город после «в» — ключ city."""
    if is_facts_request(cfg):
        n, text = request_user_message(all_text(contents))
        ops = []
        m = re.search(r"(\d[\d ]*\d) VND", text)
        if m:
            ops.append({"key": "budget", "value": f"{m.group(1)} VND", "evidence": m.group(0), "category": "budget"})
        return text_response(facts_json(*ops), prompt=10, candidates=5)
    return text_response("ok")


def test_inherited_message_cannot_overwrite_branch_fact(tmp_path):
    """Унаследованное сообщение с прежним бюджетом: facts ветки не перезаписываются; конфликт записан."""
    agent = Agent(FakeClient(responder=budget_extractor))
    bm = BranchManager(tmp_path, strategy="facts", agent=agent)
    add(bm, "Бюджет 15 000 000 VND", "ок")
    bm.checkpoint("cp")
    bm.create_branch("A", "cp")
    bm.apply_change(BranchChange(user="Теперь бюджет 22 000 000 VND", model="Принято"), agent)
    assert bm.strategy.facts.locked == {"budget": "изменено в ветке A"}
    inherited = list(pair("Напоминаю: бюджет 18 500 000 VND", "Хорошо"))
    out = bm.continue_dialogue(inherited, start_number=3, agent=agent)
    assert bm.strategy.facts.facts["budget"].value == "22 000 000 VND"       # значение ветки сохранено
    assert out["facts_conflicts"][0]["key"] == "budget"
    assert out["facts_conflicts"][0]["resolution"] == "оставлено значение ветки"
    assert bm.branch_meta()["facts_conflicts"]
    # Новое (не унаследованное) сообщение в ветке по-прежнему может менять факт.
    add(bm, "Окончательно: 25 000 000 VND", "ок")
    assert bm.strategy.facts.facts["budget"].value == "25 000 000 VND"


@pytest.mark.parametrize("strategy", config.STRATEGIES)
def test_text_conflict_skip_and_error(tmp_path, strategy):
    agent = Agent(FakeClient(responder=echo_responder()))
    bm = BranchManager(tmp_path, strategy=strategy, agent=agent)
    add(bm, "Живу в Нячанге", "ок")
    bm.checkpoint("cp")
    bm.create_branch("B", "cp")
    bm.apply_change(BranchChange(user="Переехала в Дананг", model="Понял", conflict_patterns=("нячанг",)), agent)
    inherited = [*pair("Погода в Нячанге сегодня?", "Солнечно"), *pair("Сколько тестов?", "24")]
    with pytest.raises(BranchConflictError, match="сообщении 3"):
        bm.continue_dialogue(inherited, start_number=3, agent=agent, on_conflict="error")
    assert len(bm.strategy.full_history) == 4                                # ничего не добавлено
    out = bm.continue_dialogue(inherited, start_number=3, agent=agent, on_conflict="skip")
    assert out["skipped"] == [3] and out["added"] == 2
    texts = [content_text(c) for c in bm.strategy.full_history]
    assert "Погода в Нячанге сегодня?" not in texts and "Сколько тестов?" in texts
    assert bm.branch_meta()["conflicts"][0]["resolution"] == "сообщение пропущено"


# ----- полный сценарий (fake-клиент) -----

@pytest.mark.parametrize("strategy", ["facts", "summary"])
def test_full_scenario_branches_a_and_b(tmp_path, dialogue, scenario, strategy):
    client = FakeClient(responder=echo_responder())
    agent = Agent(client)
    out = run_scenario(tmp_path / "s", strategy, dialogue, agent, scenario)
    a, b = out["branches"]["A"], out["branches"]["B"]
    assert out["checkpoint"]["messages"] == 30
    assert a["messages"] == b["messages"] == 30 + 2 + 20                      # checkpoint + изменение + 31–50
    assert a["continue"]["text_conflicts"] == [] and b["continue"]["text_conflicts"] == []
    assert a["continue"]["skipped"] == [] and b["continue"]["skipped"] == []
    ok = {name: {x["id"]: x["correct"] for x in br["answers"]} for name, br in out["branches"].items()}
    assert ok["A"][2] and ok["B"][2]                                          # Нячанг в A, Дананг в B
    if strategy == "facts":
        assert all(ok["A"].values()) and all(ok["B"].values())
    root = tmp_path / "s"
    assert "дананг" not in tree_text(root / "branches" / "A")
    assert "22000000" not in tree_text(root / "branches" / "B")
    assert "дананг" not in tree_text(root / "checkpoints" / "cp")
    bm = out["manager"]
    assert bm.branches() == ["main", "A", "B"] and bm.current == "B"
    assert "checkpoint cp: 30 сообщ." in bm.describe()
