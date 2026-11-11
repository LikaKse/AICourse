"""Офлайн-проверка через НАСТОЯЩИЙ google-genai Client с подменённым HTTP-транспортом.

В сеть ничего не уходит: httpx.MockTransport перехватывает запросы и отдаёт заготовленные
JSON-ответы в формате REST API Gemini. Так проверяется, что SDK действительно:
- отправляет схемы инструментов и не выполняет функции сам (AFC выключен);
- кладёт в следующий запрос functionCall вместе с thoughtSignature и functionResponse с тем же id;
- (День 8) отправляет в :countTokens всю историю, включая functionCall / functionResponse,
  а usageMetadata из ответа превращается в статистику токенов.
"""

import base64
import json
import unittest

import httpx
from google import genai
from google.genai import types

from agent import Agent

SIG = base64.b64encode(b"opaque-thought-signature").decode()


def _reply(parts, prompt=50, candidates=10, thoughts=0):
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": prompt, "candidatesTokenCount": candidates,
                          "thoughtsTokenCount": thoughts, "totalTokenCount": prompt + candidates + thoughts},
    }


def _client(handler):
    return genai.Client(
        api_key="test-key-not-real",
        http_options=types.HttpOptions(httpx_client=httpx.Client(transport=httpx.MockTransport(handler))),
    )


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
        sent, counted = [], []

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertNotIn("key=", str(request.url))          # ключ не в URL, а в заголовке
            body = json.loads(request.content)
            if request.url.path.endswith(":countTokens"):
                counted.append(body)
                return httpx.Response(200, json={"totalTokens": 10 * len(counted)})
            self.assertIn(":generateContent", request.url.path)
            sent.append(body)
            return httpx.Response(200, json=replies[len(sent) - 1])

        r = Agent(_client(handler)).run("Переведи 150 долларов в донги и добавь 10 процентов")

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

        # --- День 8: count_tokens ---
        # запрос пользователя + история перед каждым из 3 обращений + обмен с инструментами = 5
        self.assertEqual(len(counted), 5)
        self.assertEqual(counted[0]["contents"], [sent[0]["contents"][0]])     # только текущее сообщение
        # Перед 3-м обращением считается РОВНО то, что затем ушло в generateContent,
        # включая functionCall с thoughtSignature и functionResponse с тем же id.
        self.assertEqual(counted[3]["contents"], sent[2]["contents"])
        self.assertEqual(counted[3]["contents"][1]["parts"][0]["thoughtSignature"], SIG)
        self.assertEqual(counted[3]["contents"][2]["parts"][0]["functionResponse"]["id"], "fc-1")
        # Developer API: в countTokens нет systemInstruction/tools (SDK их здесь не принимает).
        self.assertEqual(set(counted[3]), {"contents"})
        # последний подсчёт — только вызовы инструментов и их результаты (без вопроса и финального ответа)
        self.assertEqual([c["role"] for c in counted[4]["contents"]], ["model", "user", "model", "user"])

        st = r.stats
        self.assertEqual(st.request_tokens, 10)
        self.assertEqual([c.history_tokens for c in st.calls], [20, 30, 40])
        self.assertEqual(st.tool_exchange_tokens, 50)
        self.assertEqual((st.input_tokens, st.output_tokens), (150, 30))  # 3 × usageMetadata
        self.assertEqual([c.shape.function_response_parts for c in st.calls], [0, 1, 2])

    def test_usage_metadata_with_thoughts_and_cache_parsed(self):
        def handler(request):
            if request.url.path.endswith(":countTokens"):
                return httpx.Response(200, json={"totalTokens": 7})
            reply = _reply([{"text": "ok"}], prompt=900, candidates=12, thoughts=30)
            reply["usageMetadata"]["cachedContentTokenCount"] = 512
            return httpx.Response(200, json=reply)

        r = Agent(_client(handler)).run("x")
        u = r.stats.calls[0].usage
        self.assertEqual((u.prompt_tokens, u.candidates_tokens, u.thoughts_tokens, u.cached_tokens),
                         (900, 12, 30, 512))
        self.assertEqual((r.input_tokens, r.output_tokens), (900, 42))     # thinking считается как output
        self.assertEqual(r.stats.calls[0].overhead_tokens, 900 - 7)

    def test_sdk_rejects_system_instruction_in_count_tokens_for_developer_api(self):
        """Почему count_tokens считает только историю: SDK 2.34.0 не отправит system_instruction/tools."""
        client = _client(lambda request: httpx.Response(200, json={"totalTokens": 1}))
        for cfg in (types.CountTokensConfig(system_instruction="x"),
                    types.CountTokensConfig(tools=[types.Tool(function_declarations=[])])):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                client.models.count_tokens(model="gemini-3.5-flash-lite", contents="hi", config=cfg)
        self.assertEqual(client.models.count_tokens(model="gemini-3.5-flash-lite", contents="hi").total_tokens, 1)

    def test_http_error_becomes_api_error_status(self):
        def handler(request):
            if request.url.path.endswith(":countTokens"):
                return httpx.Response(200, json={"totalTokens": 1})
            return httpx.Response(400, json={"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})

        r = Agent(_client(handler)).run("x")
        self.assertEqual(r.status, "api_error")
        self.assertIn("400", r.text)

    def test_context_overflow_error_through_real_sdk(self):
        msg = "The input token count (1100000) exceeds the maximum number of tokens allowed (1048576)."

        def handler(request):
            if request.url.path.endswith(":countTokens"):
                return httpx.Response(200, json={"totalTokens": 1_100_000})
            return httpx.Response(400, json={"error": {"code": 400, "message": msg, "status": "INVALID_ARGUMENT"}})

        agent = Agent(_client(handler), input_token_limit=1_048_576, enforce_context_limit=False)
        r = agent.run("огромный текст")
        self.assertEqual(r.status, "context_overflow")
        self.assertIn("exceeds the maximum number of tokens", r.text)
        self.assertEqual(agent.history, [])
        self.assertEqual(r.stats.calls[0].history_tokens, 1_100_000)
        self.assertIn("400", r.stats.calls[0].error)

    def test_count_tokens_http_error_does_not_break_turn(self):
        def handler(request):
            if request.url.path.endswith(":countTokens"):
                return httpx.Response(429, json={"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
            return httpx.Response(200, json=_reply([{"text": "ok"}]))

        client = genai.Client(
            api_key="test-key-not-real",
            http_options=types.HttpOptions(
                httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        r = Agent(client).run("x")
        self.assertEqual(r.status, "ok")
        self.assertIsNone(r.stats.request_tokens)
        self.assertTrue(any("429" in e for e in r.stats.count_errors))
        self.assertEqual(r.input_tokens, 50)        # usage_metadata всё равно есть



if __name__ == "__main__":
    unittest.main()
