"""Сохранение и восстановление истории диалога между перезапусками (офлайн, без API)."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from google.genai import types

import config
import main
from agent import Agent
from history_store import FORMAT_VERSION, HistoryLoadError, HistoryStore
from tests.fakes import FakeClient, call_response, text_response


def fresh_agent(store: HistoryStore, script) -> Agent:
    """«Перезапуск программы»: новый агент с историей, прочитанной из файла."""
    agent = Agent(FakeClient(script))
    agent.history = store.load()
    return agent


class HistoryStoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "results" / "chat_history.json"
        self.store = HistoryStore(self.path)


class SaveLoadTest(HistoryStoreTestCase):
    def test_plain_dialog_roundtrip(self):
        agent = Agent(FakeClient([text_response("Привет!", signature=b"sig-text"),
                                  text_response("Тебя зовут Аня.")]))
        agent.run("Меня зовут Аня")
        self.store.save(agent.history)

        restored = self.store.load()
        self.assertEqual(restored, agent.history)
        self.assertTrue(all(isinstance(c, types.Content) for c in restored))
        self.assertEqual([c.role for c in restored], ["user", "model"])
        self.assertEqual(restored[0].parts[0].text, "Меня зовут Аня")
        self.assertEqual(restored[1].parts[0].thought_signature, b"sig-text")

        # После «перезапуска» восстановленная история уходит модели целиком.
        client = FakeClient([text_response("Тебя зовут Аня.")])
        agent2 = Agent(client)
        agent2.history = restored
        result = agent2.run("Как меня зовут?")
        self.assertTrue(result.ok)
        sent = client.requests[0]["contents"]
        self.assertEqual(sent[:2], agent.history)
        self.assertEqual(sent[2].parts[0].text, "Как меня зовут?")

    def test_file_format_is_sdk_json_not_plain_text(self):
        self.store.save([types.Content(role="user", parts=[types.Part.from_text(text="Привет")])])
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(payload["format_version"], FORMAT_VERSION)
        self.assertIn("sdk_version", payload)
        self.assertEqual(payload["history"], [{"role": "user", "parts": [{"text": "Привет"}]}])
        self.assertIn("Привет", self.path.read_text(encoding="utf-8"))  # ensure_ascii=False

    def test_function_call_and_response_roundtrip(self):
        client = FakeClient([
            call_response(("convert_currency", {"amount": 150, "from_currency": "USD", "to_currency": "VND"},
                           "fc-42"), signature=b"\x00\xffsig-fc"),
            text_response("150 USD = 3 750 000 VND (фиксированный курс)."),
        ])
        agent = Agent(client)
        self.assertTrue(agent.run("Сколько будет 150 долларов в донгах?").ok)
        self.store.save(agent.history)

        restored = self.store.load()
        self.assertEqual(restored, agent.history)
        user, model_call, tool_resp, model_answer = restored
        fc_part = model_call.parts[0]
        self.assertEqual(fc_part.function_call.id, "fc-42")
        self.assertEqual(fc_part.function_call.name, "convert_currency")
        self.assertEqual(fc_part.function_call.args["amount"], 150)
        self.assertEqual(fc_part.thought_signature, b"\x00\xffsig-fc")
        fr = tool_resp.parts[0].function_response
        self.assertEqual(tool_resp.role, "user")
        self.assertEqual((fr.id, fr.name), ("fc-42", "convert_currency"))
        self.assertEqual(fr.response["result"]["converted_amount"], 3_750_000)

    def test_thought_signature_and_thought_parts_restored(self):
        signature = bytes(range(256))  # произвольные бинарные данные, не UTF-8
        history = [
            types.Content(role="user", parts=[types.Part.from_text(text="Вопрос")]),
            types.Content(role="model", parts=[
                types.Part(text="размышления", thought=True),
                types.Part(text="Ответ", thought_signature=signature),
            ]),
        ]
        self.store.save(history)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIsInstance(raw["history"][1]["parts"][1]["thought_signature"], str)  # base64 в JSON

        restored = self.store.load()
        self.assertEqual(restored[1].parts[0].thought, True)
        self.assertEqual(restored[1].parts[1].thought_signature, signature)
        self.assertEqual(restored, history)

    def test_parallel_calls_keep_ids_and_order(self):
        agent = Agent(FakeClient([
            call_response(("calculate", {"expression": "1+1"}, "a"), ("calculate", {"expression": "2+2"}, "b")),
            text_response("2 и 4"),
        ]))
        agent.run("Посчитай 1+1 и 2+2")
        self.store.save(agent.history)
        restored = self.store.load()
        self.assertEqual([p.function_call.id for p in restored[1].parts], ["a", "b"])
        self.assertEqual([p.function_response.id for p in restored[2].parts], ["a", "b"])

    def test_restart_continues_function_calling(self):
        """После перезапуска агент продолжает цикл Function Calling поверх восстановленной истории."""
        agent = Agent(FakeClient([call_response(("get_current_time", {}, "t1")), text_response("Сейчас 12:00")]))
        agent.run("Который час?")
        self.store.save(agent.history)

        agent2 = fresh_agent(self.store, [call_response(("calculate", {"expression": "2*3"}, "c1")),
                                          text_response("6")])
        result = agent2.run("А сколько 2*3?")
        self.assertTrue(result.ok)
        self.assertEqual(len(agent2.history), 8)
        self.assertEqual(agent2.history[:4], agent.history)

    def test_save_is_atomic_and_leaves_no_temp_files(self):
        self.store.save([types.Content(role="user", parts=[types.Part.from_text(text="1")])])
        self.store.save([types.Content(role="user", parts=[types.Part.from_text(text="2")])])
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["chat_history.json"])

    def test_api_key_is_not_saved(self):
        fake_key = "FAKE-KEY-FOR-HISTORY-TEST-0123456789"
        with mock.patch.dict(os.environ, {config.API_KEY_ENV: fake_key}):
            self.store.save([types.Content(role="user", parts=[types.Part.from_text(text=f"мой ключ {fake_key}")])])
        text = self.path.read_text(encoding="utf-8")
        self.assertNotIn(fake_key, text)
        self.assertIn("***", text)


class MissingAndResetTest(HistoryStoreTestCase):
    def test_missing_file_gives_empty_history(self):
        self.assertFalse(self.store.exists())
        self.assertEqual(self.store.load(), [])
        self.assertFalse(self.path.exists())  # загрузка ничего не создаёт

    def test_reset_clears_file(self):
        agent = Agent(FakeClient([text_response("Привет")]))
        agent.run("Привет")
        self.store.save(agent.history)
        agent.reset()
        self.store.clear()
        self.assertEqual(agent.history, [])
        self.assertEqual(self.store.load(), [])
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["history"], [])


class CorruptedFileTest(HistoryStoreTestCase):
    BAD = {
        "invalid_json": '{"format_version": 1, "history": [',
        "not_object": "[1, 2, 3]",
        "wrong_version": json.dumps({"format_version": 999, "history": []}),
        "history_not_list": json.dumps({"format_version": 1, "history": {"role": "user"}}),
        "unknown_field": json.dumps({"format_version": 1, "history": [
            {"role": "user", "parts": [{"text": "x", "field_from_future_sdk": 1}]}]}),
        "bad_role": json.dumps({"format_version": 1, "history": [{"role": "system", "parts": [{"text": "x"}]}]}),
        "bad_signature": json.dumps({"format_version": 1, "history": [
            {"role": "model", "parts": [{"text": "x", "thought_signature": "не base64!"}]}]}),
    }

    def write(self, text: str | bytes):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            self.path.write_bytes(text)
        else:
            self.path.write_text(text, encoding="utf-8")

    def test_corrupted_variants_raise_load_error(self):
        for name, text in {**self.BAD, "not_utf8": b"\xff\xfe\x00"}.items():
            with self.subTest(name):
                self.write(text)
                with self.assertRaises(HistoryLoadError):
                    self.store.load()

    def run_chat(self, answers: list[str], *, assume_yes=False):
        out, err = io.StringIO(), io.StringIO()
        client = FakeClient([])
        with mock.patch("builtins.input", side_effect=answers), redirect_stdout(out), redirect_stderr(err):
            code = main.cmd_chat(self.path, assume_yes=assume_yes, client=client)
        return code, out.getvalue(), err.getvalue()

    def test_chat_declined_keeps_corrupted_file(self):
        self.write(self.BAD["invalid_json"])
        code, out, err = self.run_chat(["n"])
        self.assertEqual(code, 1)
        self.assertIn("не удалось загрузить историю", err)
        self.assertIn("Файл истории не изменён", out)
        self.assertEqual(self.path.read_text(encoding="utf-8"), self.BAD["invalid_json"])

    def test_chat_eof_on_prompt_keeps_corrupted_file(self):
        self.write(self.BAD["invalid_json"])
        code, _, _ = self.run_chat([EOFError()])
        self.assertEqual(code, 1)
        self.assertEqual(self.path.read_text(encoding="utf-8"), self.BAD["invalid_json"])

    def test_chat_accepted_backs_up_and_starts_empty(self):
        self.write(self.BAD["invalid_json"])
        code, out, _ = self.run_chat(["y", "exit"])
        self.assertEqual(code, 0)
        backups = list(self.path.parent.glob("chat_history.corrupted_*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), self.BAD["invalid_json"])
        self.assertFalse(self.path.exists())  # новый файл появится после первого успешного запроса
        self.assertIn("Начинаем с пустой истории", out)


class ChatPersistenceTest(HistoryStoreTestCase):
    """Сквозная проверка CLI-чата: запуск → запрос → выход → перезапуск → reset → перезапуск."""

    def chat(self, inputs, script):
        client = FakeClient(script)
        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=inputs), redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = main.cmd_chat(self.path, client=client)
        return code, out.getvalue(), client

    def test_start_without_history_file(self):
        code, out, client = self.chat(["Привет", "exit"], [text_response("Здравствуйте!")])
        self.assertEqual(code, 0)
        self.assertNotIn("Восстановлена история", out)
        self.assertEqual(len(client.requests[0]["contents"]), 1)
        self.assertEqual(len(self.store.load()), 2)

    def test_history_survives_restart_and_reset(self):
        self.chat(["Меня зовут Аня", "exit"], [text_response("Приятно познакомиться")])

        code, out, client = self.chat(["Как меня зовут?", "exit"], [text_response("Аня")])
        self.assertIn("Восстановлена история", out)
        self.assertEqual(client.requests[0]["contents"][0].parts[0].text, "Меня зовут Аня")
        self.assertEqual(len(self.store.load()), 4)

        self.chat(["reset", "exit"], [])
        self.assertEqual(self.store.load(), [])

        _, out, client = self.chat(["Как меня зовут?", "exit"], [text_response("Не знаю")])
        self.assertNotIn("Восстановлена история", out)
        self.assertEqual(len(client.requests[0]["contents"]), 1)

    def test_failed_request_is_not_saved(self):
        from tests.fakes import api_error
        self.chat(["Привет", "exit"], [text_response("Здравствуйте")])
        before = self.path.read_text(encoding="utf-8")
        self.chat(["Сломайся", "exit"], [api_error()])
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["history"],
                         json.loads(before)["history"])

    def test_interrupted_request_is_rolled_back(self):
        def interrupt():
            raise KeyboardInterrupt
        code, _, _ = self.chat(["Привет", "Долгий запрос", "Ещё раз", "exit"],
                               [text_response("Здравствуйте"), interrupt, text_response("Ок")])
        self.assertEqual(code, 0)
        texts = [c.parts[0].text for c in self.store.load()]
        self.assertEqual(texts, ["Привет", "Здравствуйте", "Ещё раз", "Ок"])


if __name__ == "__main__":
    unittest.main()
