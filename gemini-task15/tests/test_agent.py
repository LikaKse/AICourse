"""Интеграционные тесты памяти с агентом и fake ModelClient."""

import json
import re

from memory.agent import MemoryAwareAgent
from memory.reader import MemoryReader
from memory.service import MemoryService
from memory.store import MemoryStore
from tests.fakes import FrozenClock, at


class ContextClient:
    """Fake-клиент: извлекает актуальное значение ключа из prompt."""

    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        user = prompt.rsplit("[USER]\n", 1)[-1]
        candidates = re.findall(r"[a-z0-9][a-z0-9._-]{2,}", user)
        key = next((item for item in candidates if "." in item), "")
        for line in prompt.splitlines():
            text = line.strip()
            if text.startswith(f"{key} ="):
                raw = text.split("=", 1)[1].split("|", 1)[0].strip()
                return json.loads(raw)
        return "UNKNOWN"


class RecordingClient:
    def __init__(self, response="ok"):
        self.response = response
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.response


def setup(tmp_path):
    store = MemoryStore(tmp_path)
    service = MemoryService(store, clock=FrozenClock(at(1)))
    reader = MemoryReader(store)
    client = ContextClient()
    agent = MemoryAwareAgent(reader, client)
    return service, reader, agent, client


def test_agent_sees_current_values_and_no_stale_mentions(tmp_path):
    service, reader, agent, client = setup(tmp_path)
    stale_values: set[str] = set()
    prompt_stale = 0
    response_stale = 0

    def ask(text, *, task_id=None):
        nonlocal prompt_stale, response_stale
        response = agent.ask(text, task_id=task_id)
        prompt = client.prompts[-1]
        prompt_stale += sum(
            1 for value in stale_values if value in prompt)
        response_stale += sum(
            1 for value in stale_values if value in str(response))
        return response

    service.set_long_term(
        "project.deadline", "DEADLINE-1", recorded_at=at(1), source="user")
    assert ask("project.deadline?") == "DEADLINE-1"

    stale_values.add("DEADLINE-1")
    service.set_long_term(
        "project.deadline", "DEADLINE-2", recorded_at=at(2), source="user")
    assert ask("project.deadline?") == "DEADLINE-2"
    assert "DEADLINE-1" not in client.prompts[-1]

    service.set_long_term(
        "project.requirement", "REQUIREMENT-OLD",
        recorded_at=at(1), source="user")
    assert ask("project.requirement?") == "REQUIREMENT-OLD"

    stale_values.add("REQUIREMENT-OLD")
    service.delete_long_term(
        "project.requirement", recorded_at=at(2), source="user")
    assert ask("project.requirement?") == "UNKNOWN"
    assert "REQUIREMENT-OLD" not in client.prompts[-1]

    service.set_long_term(
        "project.status", "STATUS-1", recorded_at=at(1), source="user")
    assert ask("project.status?") == "STATUS-1"

    stale_values.add("STATUS-1")
    service.set_long_term(
        "project.status", "STATUS-2", recorded_at=at(2), source="user")
    assert ask("project.status?") == "STATUS-2"
    assert "STATUS-1" not in client.prompts[-1]

    service.set_long_term(
        "stack.approach", "MONOLITH", recorded_at=at(1), source="user")
    assert ask("stack.approach?") == "MONOLITH"

    stale_values.add("MONOLITH")
    service.set_long_term(
        "stack.approach", "MICROSERVICES",
        recorded_at=at(2), source="user")
    assert ask("stack.approach?") == "MICROSERVICES"
    assert "MONOLITH" not in client.prompts[-1]

    assert prompt_stale == 0
    assert response_stale == 0
    assert "journal" not in client.prompts[-1].lower()
    assert "tombstone" not in client.prompts[-1].lower()


def test_agent_cumulative_decisions_are_in_prompt(tmp_path):
    service, reader, agent, client = setup(tmp_path)
    recording = RecordingClient("ok")
    cumulative_agent = MemoryAwareAgent(reader, recording)
    service.append_decision(
        "Use PostgreSQL", recorded_at=at(1), source="assistant")
    service.append_decision(
        "Use Redis", recorded_at=at(2), source="assistant")

    response = cumulative_agent.ask("what are the decisions?")

    assert response == "ok"
    prompt = recording.prompts[-1]
    assert "decisions:" in prompt
    assert "Use PostgreSQL" in prompt
    assert "Use Redis" in prompt
    assert "journal" not in prompt.lower()


def test_agent_temporary_facts_are_scoped_and_cleared(tmp_path):
    service, reader, agent, client = setup(tmp_path)
    service.set_temporary(
        "task-1", "task.step", "STEP-1", recorded_at=at(1), source="user")
    service.set_temporary(
        "task-2", "task.step", "STEP-2", recorded_at=at(1), source="user")

    assert agent.ask("task.step?", task_id="task-1") == "STEP-1"
    assert "STEP-2" not in client.prompts[-1]
    assert agent.ask("task.step?", task_id="task-2") == "STEP-2"
    assert "STEP-1" not in client.prompts[-1]

    service.complete_task("task-1", completed_at=at(2), source="system")
    assert agent.ask("task.step?", task_id="task-1") == "UNKNOWN"
    assert "STEP-1" not in client.prompts[-1]
    assert agent.ask("task.step?", task_id="task-2") == "STEP-2"


def test_agent_has_no_memory_write_api(tmp_path):
    service, reader, agent, client = setup(tmp_path)

    for name in (
        "set_long_term", "append_decision", "set_temporary",
        "delete_long_term", "delete_temporary", "complete_task",
        "reopen_task", "history", "store",
    ):
        assert not hasattr(agent, name)
        assert not hasattr(reader, name)


def test_agent_prompt_is_deterministic(tmp_path):
    service, reader, agent, client = setup(tmp_path)
    service.set_long_term(
        "project.deadline", "D", recorded_at=at(1), source="user")
    service.append_decision(
        "decision", recorded_at=at(1), source="assistant")

    first = agent.ask("project.deadline?")
    second = agent.ask("project.deadline?")

    assert first == second == "D"
    assert client.prompts[-1] == client.prompts[-2]
