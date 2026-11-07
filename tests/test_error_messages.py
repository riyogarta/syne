"""User-facing error messages: owner/family see the raw error, others a generic one."""

from syne.communication.errors import GENERIC_ERROR_MESSAGE, classify_error
from syne.llm.provider import LLMBadRequestError


def test_owner_sees_raw_error_with_model():
    e = LLMBadRequestError("OpenAI 400: {'error': 'bad param'}")
    msg = classify_error(e, model="gpt-6-luna", access_level="owner")
    assert msg == "⚠️ [gpt-6-luna] LLMBadRequestError: OpenAI 400: {'error': 'bad param'}"


def test_family_sees_raw_error_without_model():
    msg = classify_error(RuntimeError("boom"), access_level="family")
    assert msg == "⚠️ RuntimeError: boom"


def test_public_gets_generic_message():
    for level in ("public", "", None, "blocked"):
        assert classify_error(RuntimeError("secret detail"), model="x", access_level=level) == GENERIC_ERROR_MESSAGE


def test_default_is_generic():
    assert classify_error(ValueError("x")) == GENERIC_ERROR_MESSAGE


def test_generic_text():
    assert GENERIC_ERROR_MESSAGE == (
        "⚠️ There is a problem with the model currently in use. Please try again later."
    )


def test_long_raw_error_is_truncated():
    msg = classify_error(RuntimeError("a" * 10000), access_level="owner")
    assert len(msg) < 4096
    assert msg.endswith("… (truncated, full error in log)")
