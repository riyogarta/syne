"""1.22.96: the text embedded for memory recall is capped
(embedding.recall_query_max_chars, default 1500), so a photo/album turn
whose message carries every image analysis no longer misses
embedding.recall_timeout on its own."""
import inspect

from syne.conversation import Conversation


def _src():
    return inspect.getsource(Conversation.build_context)


def test_recall_embed_is_capped():
    src = _src()
    assert "embedding.recall_query_max_chars" in src
    assert "_qtext[:_qmax]" in src
    i = src.index("_qtext = _qtext[:_qmax]")
    assert "quick_embed(self.memory.provider, _qtext)" in src[i:i + 200]


def test_cap_default_and_off_switch():
    src = _src()
    assert '"embedding.recall_query_max_chars", 1500' in src
    assert "_qmax > 0 and len(_qtext) > _qmax" in src


def test_cap_only_on_embed_not_recall_query():
    # recall() and keyword routing still get the full query text.
    src = _src()
    assert "query=recall_query or user_message" in src
