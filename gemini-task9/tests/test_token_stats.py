"""Корректность подсчёта токенов."""

from fakes import FakeClient, api_error, fake_token_count, text_response

from context_manager import text_content
from token_stats import (
    TokenCounter, Usage, compare_count_with_usage, estimate_cost, savings_percent, sum_known, sum_usage,
    usage_from_response,
)


def test_counter_uses_count_tokens_api():
    client = FakeClient()
    counter = TokenCounter(client)
    contents = [text_content("user", "один два три"), text_content("model", "четыре")]
    assert counter.count(contents) == (fake_token_count(contents), None)
    assert counter.count(contents)[0] == (3 + 3) + (3 + 1)
    assert len(client.models.count_requests) == 2


def test_counter_empty_contents_is_zero_without_api_call():
    client = FakeClient()
    assert TokenCounter(client).count([]) == (0, None)
    assert client.models.count_requests == []


def test_counter_error_returns_none_and_message():
    client = FakeClient(count_error=api_error(429, "Resource exhausted", "RESOURCE_EXHAUSTED"))
    value, err = TokenCounter(client).count([text_content("user", "x")])
    assert value is None and "429" in err


def test_counter_disabled():
    value, err = TokenCounter(FakeClient(), enabled=False).count([text_content("user", "x")])
    assert value is None and "отключ" in err


def test_usage_from_response_and_missing_metadata():
    u = usage_from_response(text_response("a b c", prompt=50, thoughts=4))
    assert (u.input_tokens, u.candidates_tokens, u.thoughts_tokens, u.output_tokens, u.present) == (50, 3, 4, 7, True)
    assert usage_from_response(text_response("a", usage=False)).present is False


def test_sum_usage_marks_incomplete_metadata():
    total = sum_usage([Usage(prompt_tokens=10, candidates_tokens=2, present=True), Usage()])
    assert total["input_tokens"] == 10 and total["total_tokens"] == 12
    assert total["complete"] is False and total["calls"] == 2


def test_savings_and_helpers():
    assert savings_percent(1000, 250) == 75.0
    assert savings_percent(100, 120) == -20.0
    assert savings_percent(None, 1) is None and savings_percent(0, 1) is None
    assert sum_known([1, 2, 3]) == 6 and sum_known([1, None]) is None
    cmp = compare_count_with_usage(90, Usage(prompt_tokens=97, present=True))
    assert cmp["difference"] == 7
    assert compare_count_with_usage(90, Usage())["difference"] is None
    assert estimate_cost(1_000_000, 1_000_000) == 2.8
    assert estimate_cost(1, 1, model="unknown-model") is None
