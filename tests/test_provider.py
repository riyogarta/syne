"""Tests for syne/llm/provider.py — data classes and exception hierarchy."""

import pytest
from syne.llm.provider import (
    ChatMessage,
    ChatResponse,
    UsageAccumulator,
    EmbeddingResponse,
    LLMError,
    LLMRateLimitError,
    LLMAuthError,
    LLMBadRequestError,
    LLMContextWindowError,
    LLMEmptyResponseError,
)


class TestChatMessage:
    """Tests for the ChatMessage dataclass."""

    def test_creation(self):
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_default_metadata(self):
        msg = ChatMessage(role="assistant", content="hi")
        assert msg.metadata is None

    def test_with_metadata(self):
        meta = {"source": "test", "timestamp": 12345}
        msg = ChatMessage(role="system", content="prompt", metadata=meta)
        assert msg.metadata == meta
        assert msg.metadata["source"] == "test"

    def test_all_roles(self):
        for role in ("system", "user", "assistant", "tool"):
            msg = ChatMessage(role=role, content="x")
            assert msg.role == role


class TestChatResponse:
    """Tests for the ChatResponse dataclass."""

    def test_creation_minimal(self):
        resp = ChatResponse(content="hello", model="gpt-4")
        assert resp.content == "hello"
        assert resp.model == "gpt-4"

    def test_defaults(self):
        resp = ChatResponse(content="", model="test")
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.tool_calls is None
        assert resp.thinking is None

    def test_all_fields(self):
        tools = [{"name": "search", "args": {"q": "test"}}]
        resp = ChatResponse(
            content="result",
            model="claude-3",
            input_tokens=100,
            output_tokens=50,
            tool_calls=tools,
            thinking="Let me think about this...",
        )
        assert resp.content == "result"
        assert resp.model == "claude-3"
        assert resp.input_tokens == 100
        assert resp.output_tokens == 50
        assert resp.tool_calls == tools
        assert resp.thinking == "Let me think about this..."


class TestUsageAccumulator:
    """Tests for the UsageAccumulator class."""

    def test_initial_values(self):
        acc = UsageAccumulator()
        assert acc.total_input == 0
        assert acc.total_output == 0
        assert acc.last_input == 0
        assert acc.rounds == 0

    def test_add_single(self):
        acc = UsageAccumulator()
        resp = ChatResponse(content="x", model="m", input_tokens=100, output_tokens=50)
        acc.add(resp)
        assert acc.total_input == 100
        assert acc.total_output == 50
        assert acc.last_input == 100
        assert acc.rounds == 1

    def test_add_accumulation(self):
        acc = UsageAccumulator()
        r1 = ChatResponse(content="a", model="m", input_tokens=100, output_tokens=20)
        r2 = ChatResponse(content="b", model="m", input_tokens=150, output_tokens=30)
        r3 = ChatResponse(content="c", model="m", input_tokens=200, output_tokens=40)
        acc.add(r1)
        acc.add(r2)
        acc.add(r3)

        assert acc.total_input == 100 + 150 + 200
        assert acc.total_output == 20 + 30 + 40
        assert acc.last_input == 200  # last call's input
        assert acc.rounds == 3

    def test_apply_to(self):
        acc = UsageAccumulator()
        r1 = ChatResponse(content="a", model="m", input_tokens=100, output_tokens=20)
        r2 = ChatResponse(
            content="final",
            model="m",
            input_tokens=200,
            output_tokens=30,
            tool_calls=[{"name": "t"}],
            thinking="thought",
        )
        acc.add(r1)
        acc.add(r2)

        result = acc.apply_to(r2)
        assert result.content == "final"
        assert result.model == "m"
        assert result.input_tokens == 200  # last_input
        assert result.output_tokens == 50  # total_output (20+30)
        assert result.tool_calls == [{"name": "t"}]
        assert result.thinking == "thought"

    def test_apply_to_returns_new_instance(self):
        acc = UsageAccumulator()
        resp = ChatResponse(content="x", model="m", input_tokens=10, output_tokens=5)
        acc.add(resp)
        result = acc.apply_to(resp)
        assert result is not resp


class TestExceptionClasses:
    """Tests for the LLM exception hierarchy."""

    def test_llm_error_base(self):
        err = LLMError("something failed")
        assert str(err) == "something failed"
        assert isinstance(err, Exception)

    def test_rate_limit_error(self):
        err = LLMRateLimitError("429 too many requests")
        assert isinstance(err, LLMError)
        assert isinstance(err, Exception)
        assert str(err) == "429 too many requests"

    def test_auth_error(self):
        err = LLMAuthError("invalid api key")
        assert isinstance(err, LLMError)

    def test_bad_request_error(self):
        err = LLMBadRequestError("malformed")
        assert isinstance(err, LLMError)

    def test_context_window_error(self):
        err = LLMContextWindowError("too long")
        assert isinstance(err, LLMError)

    def test_empty_response_error(self):
        err = LLMEmptyResponseError("no content")
        assert isinstance(err, LLMError)

    def test_all_inherit_from_llm_error(self):
        subclasses = [
            LLMRateLimitError,
            LLMAuthError,
            LLMBadRequestError,
            LLMContextWindowError,
            LLMEmptyResponseError,
        ]
        for cls in subclasses:
            err = cls("test")
            assert isinstance(err, LLMError), f"{cls.__name__} should inherit from LLMError"

    def test_catchable_as_llm_error(self):
        with pytest.raises(LLMError):
            raise LLMRateLimitError("rate limited")


class TestEmbeddingResponse:
    """Tests for the EmbeddingResponse dataclass."""

    def test_creation(self):
        resp = EmbeddingResponse(
            vector=[0.1, 0.2, 0.3],
            model="text-embedding-3-small",
            dimensions=3,
        )
        assert resp.vector == [0.1, 0.2, 0.3]
        assert resp.model == "text-embedding-3-small"
        assert resp.dimensions == 3

    def test_default_input_tokens(self):
        resp = EmbeddingResponse(vector=[], model="m", dimensions=0)
        assert resp.input_tokens == 0

    def test_with_input_tokens(self):
        resp = EmbeddingResponse(
            vector=[0.5] * 1536,
            model="embed-v2",
            dimensions=1536,
            input_tokens=42,
        )
        assert resp.input_tokens == 42
        assert len(resp.vector) == 1536
