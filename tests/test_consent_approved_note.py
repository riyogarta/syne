from syne.conversation import consent_approved_result, CONSENT_APPROVED_PREFIX


def test_note_keeps_result_and_marks_approval():
    out = consent_approved_result("abc123", "Memory id 5 updated.")
    assert out.startswith(CONSENT_APPROVED_PREFIX)
    assert "abc123" in out
    assert out.endswith("Memory id 5 updated.")


def test_note_without_hash():
    out = consent_approved_result("", "ok")
    assert "(consent" not in out
    assert out.endswith("ok")


def test_note_has_no_consent_marker():
    # must not re-trigger Yes/No buttons or the held-row matcher
    out = consent_approved_result("abc123", "done")
    assert "[[CONSENT_BUTTONS:hash=" not in out
