"""Офлайн-проверка через НАСТОЯЩИЙ google-genai Client с подменённым HTTP-транспортом.

В сеть ничего не уходит: httpx.MockTransport перехватывает запросы и отдаёт заготовленные
JSON-ответы в формате REST API Gemini. Так проверяется, что SDK действительно:
- отправляет схемы инструментов и не выполняет функции сам (AFC выключен);
- кладёт в следующий запрос functionCall вместе с thoughtSignature и functionResponse с тем же id.
"""

import base64
import json
import unittest

import httpx
from google import genai
from google.genai import types

from agent import Agent

SIG = base64.b64encode(b"opaque-thought-signature").decode()


def _reply(parts):
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 10, "totalTokenCount": 60},
    }


class SdkWireFormatTest(unittest.TestCase):
    def test_full_two_tool_loop_through_real_sdk(self):
        replies = [
            _reply([{"functionCall": {"id": "fc-1", "name": "convert_currency",
                                      "args": {"amount": 150, "from_currency": "USD", "to_currency": "VND"}},
                     "thoughtSignature": SIG}]),
            _reply([{"functionCall": {"id": "fc-2", "name": "calculate", "args": {"expression": "3750000 * 1.1"}},
                     "thoughtSignature": SIG}]),
            _reply([{"text": "Итого 4 125 000 VND (фиксированный демонстрационный курс)."}]),
        ]
        sent = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertIn(":generateContent", request.url.path)
            self.assertNotIn("key=", str(request.url))          # ключ не в URL, а в заголовке
            sent.append(json.loads(request.content))
            return httpx.Response(200, json=replies[len(sent) - 1])

        client = genai.Client(
            api_key="test-key-not-real",
            http_options=types.HttpOptions(httpx_client=httpx.Client(transport=httpx.MockTransport(handler))),
        )
        r = Agent(client).run("Переведи 150 долларов в донги и добавь 10 процентов")

        self.assertEqual(r.status, "ok", r.text)
        self.assertEqual(len(sent), 3)
        self.assertEqual([c.name for c in r.tool_calls], ["convert_currency", "calculate"])
        self.assertEqual(r.tool_calls[1].response["result"]["result"], 4_125_000)

        first = sent[0]
        decl_names = [d["name"] for d in first["tools"][0]["functionDeclarations"]]
        self.assertEqual(sorted(decl_names), ["calculate", "convert_currency", "get_current_time"])
        self.assertEqual(first["toolConfig"]["functionCallingConfig"]["mode"], "AUTO")

        third = sent[2]["contents"]
        self.assertEqual([c["role"] for c in third], ["user", "model", "user", "model", "user"])
        self.assertEqual(third[1]["parts"][0]["thoughtSignature"], SIG)
        self.assertEqual(third[3]["parts"][0]["thoughtSignature"], SIG)
        fr1 = third[2]["parts"][0]["functionResponse"]
        fr2 = third[4]["parts"][0]["functionResponse"]
        self.assertEqual((fr1["id"], fr1["name"]), ("fc-1", "convert_currency"))
        self.assertEqual(fr1["response"]["result"]["converted_amount"], 3_750_000)
        self.assertEqual((fr2["id"], fr2["name"], fr2["response"]["result"]["result"]), ("fc-2", "calculate", 4_125_000))

    def test_http_error_becomes_api_error_status(self):
        def handler(request):
            return httpx.Response(400, json={"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})

        client = genai.Client(
            api_key="test-key-not-real",
            http_options=types.HttpOptions(httpx_client=httpx.Client(transport=httpx.MockTransport(handler))),
        )
        r = Agent(client).run("x")
        self.assertEqual(r.status, "api_error")
        self.assertIn("400", r.text)


if __name__ == "__main__":
    unittest.main()
