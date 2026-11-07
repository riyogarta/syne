"""SubAgentIncomplete must carry its real cause, not always 'max rounds'."""

from syne.subagent import SubAgentIncomplete


def test_default_reason_is_max_rounds():
    e = SubAgentIncomplete("partial")
    assert e.partial_result == "partial"
    assert e.reason == "reached max rounds"
    assert "reached max rounds" in str(e)


def test_consent_block_reason_is_preserved():
    e = SubAgentIncomplete("msg", reason="Command blocked pending owner consent: `python3 x.py`")
    assert e.reason.startswith("Command blocked pending owner consent")
    assert "max rounds" not in str(e)
    assert "Command blocked pending owner consent" in str(e)
