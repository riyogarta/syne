"""Tests for the phantom-action regex (anti-hallucination layer 3).

_PHANTOM_ACTION_RE flags assistant text that CLAIMS an action was performed
(save/write/run/send/delete/create/search/store) when NO tool was actually
called. The engine appends a warning to such responses. These tests lock the
regex behaviour so future edits don't silently break the guard (positive
matches) or make it fire on innocent prose (false positives).
"""

import pytest

from syne.conversation import _PHANTOM_ACTION_RE


def _matches(text: str) -> bool:
    return _PHANTOM_ACTION_RE.search(text) is not None


# ---------------------------------------------------------------------------
# POSITIVE — must be flagged as phantom action claims
# ---------------------------------------------------------------------------

class TestPhantomPositive:

    @pytest.mark.parametrize("text", [
        "Sudah saya simpan ke memori.",
        "Sudah kusimpan catatannya.",
        "Berhasil menjalankan perintahnya.",
        "Sudah saya jalankan scriptnya.",
        "Sudah saya kirim pesannya ke grup.",
        "Sudah saya hapus file lama itu.",
        "Sudah saya buat PDF-nya.",
        "I have saved the file.",
        "I've already stored that in memory.",
        "Successfully executed the command.",
        "I already sent the message.",
        "Done — deleted the old record.",
    ])
    def test_flags_action_claims(self, text):
        assert _matches(text), f"should flag phantom claim: {text!r}"


# ---------------------------------------------------------------------------
# NEGATIVE — must NOT be flagged (innocent prose / no action claim)
# ---------------------------------------------------------------------------

class TestPhantomNegative:

    @pytest.mark.parametrize("text", [
        "Bagaimana kabarmu hari ini?",
        "Baik, aku akan bantu carikan informasinya.",
        "Menurutku decay v2 sudah sesuai cetak biru.",
        "Mau kubuatkan test-nya sekarang?",
        "The weather looks nice today.",
        "Let me know if you want me to proceed.",
        "Ini penjelasan tentang cara kerja fitur tersebut.",
        "Cap-nya sekarang 1000, bukan 50.",
        "",  # empty response
    ])
    def test_does_not_flag_innocent(self, text):
        assert not _matches(text), f"false positive on: {text!r}"


# ---------------------------------------------------------------------------
# Regex object sanity
# ---------------------------------------------------------------------------

class TestPhantomRegexObject:

    def test_is_case_insensitive(self):
        assert _matches("SUDAH SAYA SIMPAN datanya")
        assert _matches("i HAVE SAVED the file")

    def test_span_window_bounded(self):
        # A verb far (>40 chars) after the claim word should NOT match —
        # the regex only bridges a short gap, preventing sentence-spanning
        # false positives.
        far = "sudah " + ("x" * 60) + " simpan"
        assert not _matches(far), "should not bridge >40 char gap"
