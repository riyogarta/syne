"""1.22.97: recall query for media turns = caption text + tail of the
appended analysis (embedding.recall_media_tail_chars, default 1000); the
general embed cap keeps the END of a long text, where the question
usually is."""
import inspect

from syne.conversation import Conversation


def _ctx_src():
    return inspect.getsource(Conversation.build_context)


def _chat_src():
    return inspect.getsource(Conversation._chat_inner)


def test_embed_cap_keeps_tail():
    src = _ctx_src()
    assert "_qtext = _qtext[-_qmax:]" in src
    assert "_qtext = _qtext[:_qmax]" not in src


def test_media_recall_query_caption_plus_tail():
    src = _chat_src()
    assert "_pre_media_text = user_message" in src
    # snapshot is taken BEFORE the ability pre-process appends results
    assert src.index("_pre_media_text = user_message") < src.index(
        "await self._ability_first_preprocess("
    )
    assert '"embedding.recall_media_tail_chars", 1000' in src
    assert "_added[-_tail_n:]" in src
    assert "user_message.startswith(_pre_media_text)" in src


def test_original_text_still_wins():
    src = _chat_src()
    assert '.get("original_text")' in src
    i = src.index('.get("original_text")')
    assert "if recall_query is None:" in src[i:i + 200]


def _tail_logic(pre, full, tail_n):
    # mirror of the _chat_inner logic, for behaviour checks
    if full != pre and full.startswith(pre):
        added = full[len(pre):]
        if tail_n > 0 and added.strip():
            return f"{pre}\n\n{added[-tail_n:]}"
        return pre
    return full


def test_tail_logic_behaviour():
    pre = "[User sent 2 photos]\n\nRevisi gambar ini"
    analysis = "\n\n[Image analysis result: " + "x" * 5000 + "END]"
    q = _tail_logic(pre, pre + analysis, 1000)
    assert q.startswith(pre)
    assert q.endswith("END]")
    assert len(q) == len(pre) + 2 + 1000
    assert _tail_logic("halo", "halo", 1000) == "halo"
    assert _tail_logic(pre, pre + analysis, 0) == pre
