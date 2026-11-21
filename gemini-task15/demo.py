"""Офлайн-демонстрация устаревания и конфликтов памяти Дня 15.

Сценарий работает во временном state-dir, использует только fake ModelClient
и не обращается к Gemini API.
"""
from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from memory.agent import MemoryAwareAgent
from memory.reader import MemoryReader
from memory.service import MemoryService
from memory.store import MemoryStore

UTC = timezone.utc
T0 = datetime(2026, 12, 1, 9, 0, tzinfo=UTC)


def t(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


class DemoClock:
    def __init__(self, moment: datetime):
        self.moment = moment

    def now(self) -> datetime:
        return self.moment

    def set(self, moment: datetime) -> None:
        self.moment = moment


class FactLookupClient:
    """Fake client: возвращает актуальное значение ключа из prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
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
    def __init__(self, response: str = "ok") -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


@dataclass(frozen=True)
class DemoStep:
    name: str
    question: str
    expected: str
    actual: str
    prompt: str
    passed: bool


@dataclass(frozen=True)
class DemoReport:
    steps: tuple[DemoStep, ...]
    stale_prompt: int
    stale_response: int
    stale_values: tuple[str, ...]
    long_term: dict[str, Any]
    decisions: tuple[Any, ...]
    remaining_temporary: dict[str, dict[str, Any]]
    cumulative_prompt: str
    temporary_prompt_task_1: str
    temporary_prompt_task_2: str
    temporary_prompt_after_complete: str
    passed: bool


def run_demo(*, verbose: bool = False) -> DemoReport:
    with tempfile.TemporaryDirectory(prefix="task15-demo-") as state_dir:
        store = MemoryStore(state_dir)
        clock = DemoClock(t(0))
        service = MemoryService(store, clock=clock)
        reader = MemoryReader(store)
        fact_client = FactLookupClient()
        recording_client = RecordingClient("ok")

        steps: list[DemoStep] = []
        stale_values: set[str] = set()
        stale_prompt = 0
        stale_response = 0

        def ask(
            client,
            name: str,
            question: str,
            expected: str,
            *,
            task_id: str | None = None,
        ) -> DemoStep:
            nonlocal stale_prompt, stale_response
            agent = MemoryAwareAgent(reader, client)
            response = agent.ask(question, task_id=task_id)
            prompt = client.prompts[-1]
            actual = str(response)
            stale_prompt += sum(
                prompt.count(value) for value in stale_values)
            stale_response += sum(
                actual.count(value) for value in stale_values)
            step = DemoStep(
                name=name,
                question=question,
                expected=expected,
                actual=actual,
                prompt=prompt,
                passed=actual == expected,
            )
            steps.append(step)
            if verbose:
                print(f"\n=== {name} ===")
                print(f"USER: {question}")
                print(f"AGENT: {actual}")
                print(f"EXPECTED: {expected}")
                print(f"STEP: {'PASS' if step.passed else 'FAIL'}")
                print("PROMPT:")
                print(prompt)
            return step

        # 1. Срок
        clock.set(t(0))
        service.set_long_term(
            "project.deadline", "DEADLINE-1",
            recorded_at=t(0), source="user", reason="initial")
        ask(
            fact_client, "deadline-1",
            "What is project.deadline?", "DEADLINE-1")

        stale_values.add("DEADLINE-1")
        clock.set(t(1))
        service.set_long_term(
            "project.deadline", "DEADLINE-2",
            recorded_at=t(1), source="user", reason="deadline shifted")
        ask(
            fact_client, "deadline-2",
            "What is project.deadline?", "DEADLINE-2")

        # 2. Требование и его отмена
        clock.set(t(2))
        service.set_long_term(
            "project.requirement", "REQUIREMENT-OLD",
            recorded_at=t(2), source="user", reason="old requirement")
        ask(
            fact_client, "requirement-old",
            "What is project.requirement?", "REQUIREMENT-OLD")

        stale_values.add("REQUIREMENT-OLD")
        clock.set(t(3))
        service.delete_long_term(
            "project.requirement",
            recorded_at=t(3), source="user", reason="requirement cancelled")
        ask(
            fact_client, "requirement-deleted",
            "What is project.requirement?", "UNKNOWN")

        # 3. Статус
        clock.set(t(4))
        service.set_long_term(
            "project.status", "STATUS-1",
            recorded_at=t(4), source="user", reason="initial status")
        ask(
            fact_client, "status-1",
            "What is project.status?", "STATUS-1")

        stale_values.add("STATUS-1")
        clock.set(t(5))
        service.set_long_term(
            "project.status", "STATUS-2",
            recorded_at=t(5), source="user", reason="status changed")
        ask(
            fact_client, "status-2",
            "What is project.status?", "STATUS-2")

        # 4. Технический подход
        clock.set(t(6))
        service.set_long_term(
            "stack.approach", "MONOLITH",
            recorded_at=t(6), source="user", reason="initial approach")
        ask(
            fact_client, "approach-1",
            "What is stack.approach?", "MONOLITH")

        stale_values.add("MONOLITH")
        clock.set(t(7))
        service.set_long_term(
            "stack.approach", "MICROSERVICES",
            recorded_at=t(7), source="user", reason="approach changed")
        ask(
            fact_client, "approach-2",
            "What is stack.approach?", "MICROSERVICES")

        # 5. Cumulative decisions
        clock.set(t(8))
        service.append_decision(
            "Use PostgreSQL",
            recorded_at=t(8), source="assistant", reason="storage choice")
        clock.set(t(9))
        service.append_decision(
            "Use Redis",
            recorded_at=t(9), source="assistant", reason="cache choice")
        cumulative_step = ask(
            recording_client, "cumulative-decisions",
            "What are the active decisions?", "ok")
        cumulative_prompt = cumulative_step.prompt

        # 6. Temporary memory
        clock.set(t(10))
        service.set_temporary(
            "task-1", "task.step", "TEMP-1",
            recorded_at=t(10), source="user", reason="task step")
        service.set_temporary(
            "task-1", "task.note", "TEMP-NOTE-1",
            recorded_at=t(10), source="user", reason="task note")
        clock.set(t(11))
        service.set_temporary(
            "task-2", "task.step", "TEMP-2",
            recorded_at=t(11), source="user", reason="other task")

        task_1_step = ask(
            fact_client, "temporary-task-1",
            "What is task.step for task-1?", "TEMP-1",
            task_id="task-1")
        task_2_step = ask(
            fact_client, "temporary-task-2",
            "What is task.step for task-2?", "TEMP-2",
            task_id="task-2")

        stale_values.add("TEMP-1")
        stale_values.add("TEMP-NOTE-1")
        clock.set(t(12))
        service.complete_task(
            "task-1", completed_at=t(12), source="system",
            reason="task completed")
        after_complete = ask(
            fact_client, "temporary-task-1-cleared",
            "What is task.step for task-1?", "UNKNOWN",
            task_id="task-1")
        ask(
            fact_client, "temporary-task-2-after-clear",
            "What is task.step for task-2?", "TEMP-2",
            task_id="task-2")

        # 7. Финальные значения
        clock.set(t(13))
        ask(
            fact_client, "final-deadline",
            "What is project.deadline?", "DEADLINE-2")
        ask(
            fact_client, "final-status",
            "What is project.status?", "STATUS-2")
        ask(
            fact_client, "final-approach",
            "What is stack.approach?", "MICROSERVICES")
        ask(
            fact_client, "final-requirement",
            "What is project.requirement?", "UNKNOWN")

        context = reader.context()
        task_1_context = reader.context("task-1")
        task_2_context = reader.context("task-2")
        long_term = {
            item.key: item.value for item in context.long_term
        }
        decisions = tuple(item.value for item in context.decisions)
        remaining_temporary = {
            "task-1": {
                item.key: item.value for item in task_1_context.temporary
            },
            "task-2": {
                item.key: item.value for item in task_2_context.temporary
            },
        }

        cumulative_ok = (
            "decisions:" in cumulative_prompt
            and "Use PostgreSQL" in cumulative_prompt
            and "Use Redis" in cumulative_prompt
        )
        temporary_scope_ok = (
            "TEMP-1" in task_1_step.prompt
            and "TEMP-NOTE-1" in task_1_step.prompt
            and "TEMP-2" not in task_1_step.prompt
            and "TEMP-2" in task_2_step.prompt
            and "TEMP-1" not in task_2_step.prompt
        )
        temporary_cleared_ok = (
            "TEMP-1" not in after_complete.prompt
            and "TEMP-NOTE-1" not in after_complete.prompt
            and "DEADLINE-2" in after_complete.prompt
            and "Use PostgreSQL" in after_complete.prompt
        )
        final_ok = (
            long_term.get("project.deadline") == "DEADLINE-2"
            and long_term.get("project.status") == "STATUS-2"
            and long_term.get("stack.approach") == "MICROSERVICES"
            and "project.requirement" not in long_term
            and decisions == ("Use PostgreSQL", "Use Redis")
            and remaining_temporary["task-1"] == {}
            and remaining_temporary["task-2"] == {"task.step": "TEMP-2"}
        )
        clean_prompt = all(
            "journal" not in step.prompt.lower()
            and "tombstone" not in step.prompt.lower()
            for step in steps
        )
        passed = (
            all(step.passed for step in steps)
            and stale_prompt == 0
            and stale_response == 0
            and cumulative_ok
            and temporary_scope_ok
            and temporary_cleared_ok
            and final_ok
            and clean_prompt
        )

        return DemoReport(
            steps=tuple(steps),
            stale_prompt=stale_prompt,
            stale_response=stale_response,
            stale_values=tuple(sorted(stale_values)),
            long_term=long_term,
            decisions=decisions,
            remaining_temporary=remaining_temporary,
            cumulative_prompt=cumulative_prompt,
            temporary_prompt_task_1=task_1_step.prompt,
            temporary_prompt_task_2=task_2_step.prompt,
            temporary_prompt_after_complete=after_complete.prompt,
            passed=passed,
        )


def _print_final_state(report: DemoReport) -> None:
    print("\n=== FINAL MEMORY ===")
    print("Long-term facts:")
    if not report.long_term:
        print("  (none)")
    for key in sorted(report.long_term):
        print(f"  {key} = {json.dumps(report.long_term[key], ensure_ascii=False)}")
    print("Cumulative decisions:")
    if not report.decisions:
        print("  (none)")
    for index, value in enumerate(report.decisions, 1):
        print(f"  {index}. {json.dumps(value, ensure_ascii=False)}")
    print("Remaining temporary:")
    for task_id in sorted(report.remaining_temporary):
        facts = report.remaining_temporary[task_id]
        if not facts:
            continue
        for key in sorted(facts):
            print(
                f"  {task_id}.{key} = "
                f"{json.dumps(facts[key], ensure_ascii=False)}")
    if all(not facts for facts in report.remaining_temporary.values()):
        print("  (none)")
    print(f"Stale in prompt: {report.stale_prompt}")
    print(f"Stale in responses: {report.stale_response}")
    print("RESULT:", "PASS" if report.passed else "FAIL")


def main() -> int:
    report = run_demo(verbose=True)
    _print_final_state(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
