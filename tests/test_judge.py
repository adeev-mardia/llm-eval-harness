"""Tests for the LLM-judge backend's HTTP request construction and
response parsing. Since this environment has no outbound access to
hosted LLM APIs, these tests stub ``requests.post`` -- a standard
transport-level testing technique -- so the real request-building and
response-parsing code paths are still exercised.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from evalharness.judge import JudgeTransportError, OpenAICompatibleJudge


def _fake_response(status_code=200, body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = json.dumps(body) if body is not None else ""
    resp.json.return_value = body
    return resp


def test_score_relevance_builds_correct_request_and_parses_response():
    judge = OpenAICompatibleJudge(model="gpt-4o-mini", base_url="https://api.openai.com/v1", api_key="sk-test")
    body = {"choices": [{"message": {"content": '{"relevance": 0.82}'}}]}

    with patch("requests.post", return_value=_fake_response(200, body)) as mock_post:
        result = judge.score_relevance("What is Mars?", "Mars is the fourth planet from the Sun.")

    assert result.score == 0.82
    assert "judged by gpt-4o-mini" in result.detail

    args, kwargs = mock_post.call_args
    assert args[0] == "https://api.openai.com/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
    payload = kwargs["json"]
    assert payload["model"] == "gpt-4o-mini"
    assert payload["messages"][1]["content"].startswith("QUESTION:")
    assert "What is Mars?" in payload["messages"][1]["content"]


def test_score_relevance_clamps_out_of_range_values():
    judge = OpenAICompatibleJudge(model="gpt-4o-mini")
    body = {"choices": [{"message": {"content": '{"relevance": 1.7}'}}]}
    with patch("requests.post", return_value=_fake_response(200, body)):
        result = judge.score_relevance("Q", "A")
    assert result.score == 1.0


def test_score_faithfulness_sends_one_request_per_claim_and_aggregates():
    judge = OpenAICompatibleJudge(model="local-model", base_url="http://localhost:11434/v1")
    answer = "Mercury is small. Mercury has rings."
    context = ["Mercury is the smallest planet in the Solar System."]

    responses = [
        _fake_response(200, {"choices": [{"message": {"content": '{"supported": true, "best_context_index": 0}'}}]}),
        _fake_response(200, {"choices": [{"message": {"content": '{"supported": false, "best_context_index": null}'}}]}),
    ]
    with patch("requests.post", side_effect=responses) as mock_post:
        result = judge.score_faithfulness(answer, context)

    assert mock_post.call_count == 2
    assert result.score == 0.5
    assert len(result.unsupported_claims) == 1
    assert result.unsupported_claims[0].text == "Mercury has rings."


def test_score_faithfulness_short_circuits_with_no_context():
    judge = OpenAICompatibleJudge(model="local-model")
    with patch("requests.post") as mock_post:
        result = judge.score_faithfulness("Mercury is small.", [])
    mock_post.assert_not_called()
    assert result.score == 0.0


def test_score_faithfulness_short_circuits_with_empty_answer():
    judge = OpenAICompatibleJudge(model="local-model")
    with patch("requests.post") as mock_post:
        result = judge.score_faithfulness("", ["some context"])
    mock_post.assert_not_called()
    assert result.score == 1.0


def test_non_200_response_raises_transport_error():
    judge = OpenAICompatibleJudge(model="gpt-4o-mini")
    with patch("requests.post", return_value=_fake_response(401, {"error": "unauthorized"})):
        with pytest.raises(JudgeTransportError):
            judge.score_relevance("Q", "A")


def test_malformed_json_content_raises_transport_error():
    judge = OpenAICompatibleJudge(model="gpt-4o-mini")
    body = {"choices": [{"message": {"content": "not json at all"}}]}
    with patch("requests.post", return_value=_fake_response(200, body)):
        with pytest.raises(JudgeTransportError):
            judge.score_relevance("Q", "A")


def test_fenced_json_code_block_is_tolerated():
    judge = OpenAICompatibleJudge(model="gpt-4o-mini")
    body = {"choices": [{"message": {"content": '```json\n{"relevance": 0.5}\n```'}}]}
    with patch("requests.post", return_value=_fake_response(200, body)):
        result = judge.score_relevance("Q", "A")
    assert result.score == 0.5


def test_unexpected_response_shape_raises_transport_error():
    judge = OpenAICompatibleJudge(model="gpt-4o-mini")
    with patch("requests.post", return_value=_fake_response(200, {"unexpected": "shape"})):
        with pytest.raises(JudgeTransportError):
            judge.score_relevance("Q", "A")


def test_name_defaults_to_model_qualified_name():
    judge = OpenAICompatibleJudge(model="gpt-4o-mini")
    assert judge.name == "llm-judge:gpt-4o-mini"


def test_explicit_name_is_preserved():
    judge = OpenAICompatibleJudge(model="gpt-4o-mini", name="my-custom-judge")
    assert judge.name == "my-custom-judge"
