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

    # Dev/ops verb list expansion (2026-07-13). Catches the claim patterns a
    # dev-flavored assistant most often phantoms.
    @pytest.mark.parametrize("text", [
        "Sudah saya push commit-nya ke main.",
        "Sudah saya pull latest changes.",
        "Berhasil install dependensinya.",
        "Sudah saya deploy ke staging.",
        "Sudah saya restart service-nya.",
        "I have deployed the app.",
        "Successfully committed the change.",
        "I've already installed the package.",
    ])
    def test_flags_dev_ops_claims(self, text):
        assert _matches(text), f"should flag dev/ops phantom claim: {text!r}"

    # Indonesian passive (ter- prefix). No marker needed — the prefix itself
    # carries claim of completion.
    @pytest.mark.parametrize("text", [
        "Tersimpan di database.",
        "Terkirim ke grup keluarga.",
        "Terhapus dari daftar.",
        "Terinstall semua dependensinya.",
        "Terupdate ke versi terbaru.",
        "Tercatat di log.",
        "Terupload ke drive.",
        "Terselesaikan semua tugasnya.",
    ])
    def test_flags_indonesian_passive(self, text):
        assert _matches(text), f"should flag ID passive claim: {text!r}"

    # English standalone participle at end of clause. Terse status reports
    # that carry the same 'action was performed' claim without any 'I have'
    # preamble.
    @pytest.mark.parametrize("text", [
        "Message sent.",
        "File saved.",
        "Task completed.",
        "Deployed!",
        "Committed.",
        "All records deleted.",
    ])
    def test_flags_english_standalone_participle(self, text):
        assert _matches(text), f"should flag EN participle claim: {text!r}"


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

    # Indonesian words that START with 'ter-' but are NOT action-completion
    # claims. The ter- passive branch must never trip on these — its verb
    # vocabulary is deliberately narrow.
    @pytest.mark.parametrize("text", [
        "Terima kasih atas bantuannya.",
        "Terlihat sudah sesuai.",
        "Termasuk juga permintaan itu.",
        "Terjadi kesalahan di server.",
        "Terlalu banyak untuk ditangani sekarang.",
        "Terserah kamu saja.",
        "Terbaru versinya sudah keluar.",
        "Terkadang begitulah cara kerjanya.",
    ])
    def test_does_not_flag_benign_ter_words(self, text):
        assert not _matches(text), f"false positive on benign ter- word: {text!r}"

    # English participles / verb-lookalikes that must NOT fire — either
    # they're not at end of clause, or they're a different tense (present
    # continuous / imperative), or the trailing punctuation is missing.
    @pytest.mark.parametrize("text", [
        "I'm committed to helping you.",         # 'committed' not at end of clause
        "The user sent me this",                  # 'sent' not at end (followed by ' me')
        "Sending you a summary now.",             # 'sending' ≠ participle 'sent'
        "Save your work before restarting.",      # 'save' as imperative, no marker
        "Please push the button.",                # 'push' as imperative, no marker
        "Baik, aku akan install dependensinya.",  # future tense, no marker
        "Mau kupush sekarang?",                   # question / future, no marker
    ])
    def test_does_not_flag_non_completion_forms(self, text):
        assert not _matches(text), f"false positive on non-completion form: {text!r}"


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


# ---------------------------------------------------------------------------
# Guard gating — the WARNING should only be appended when:
#   (regex matches) AND (final response has no tool_calls)
#   AND (no tool ran this turn)
#
# This mirrors the runtime condition in ConversationManager:
#     if not response.tool_calls and not tools_ran_this_turn:
#         if _PHANTOM_ACTION_RE.search(content): <append warning>
#
# The bug fixed on 11 Jul 2026: after a tool ran, _handle_tool_calls returns a
# FINAL response with empty .tool_calls, so the old condition (which only
# checked response.tool_calls) fired a FALSE POSITIVE when the model summarized
# a real tool_result using words like "Berhasil". `tools_ran_this_turn` closes
# that gap.
# ---------------------------------------------------------------------------

def _should_warn(content: str, final_has_tool_calls: bool, tools_ran_this_turn: bool) -> bool:
    """Pure replica of the runtime gating decision (see ConversationManager)."""
    if final_has_tool_calls or tools_ran_this_turn:
        return False
    return _PHANTOM_ACTION_RE.search(content) is not None


class TestPhantomGuardGating:

    def test_real_phantom_warns(self):
        # Model claims an action, NO tool ran, final has no tool_calls → WARN
        assert _should_warn("Sudah saya simpan ke memori.",
                            final_has_tool_calls=False,
                            tools_ran_this_turn=False)

    def test_tool_ran_then_summary_no_warn(self):
        # THE FIX: tool actually ran this turn, model summarizes with an
        # action-word → must NOT warn (this was the speedtest false positive).
        assert not _should_warn("Berhasil menjalankan speedtest lewat WiFi.",
                                final_has_tool_calls=False,
                                tools_ran_this_turn=True)

    def test_final_still_calling_tool_no_warn(self):
        # Final response is itself still calling a tool → not a phantom claim.
        assert not _should_warn("Sudah saya jalankan.",
                                final_has_tool_calls=True,
                                tools_ran_this_turn=False)

    def test_innocent_prose_no_warn_even_without_tools(self):
        # No action claim + no tool → still no warning (regex doesn't match).
        assert not _should_warn("Ini penjelasan biasa saja.",
                                final_has_tool_calls=False,
                                tools_ran_this_turn=False)

    def test_tool_ran_but_innocent_summary_no_warn(self):
        assert not _should_warn("Hasilnya 111 Mbps download.",
                                final_has_tool_calls=False,
                                tools_ran_this_turn=True)
