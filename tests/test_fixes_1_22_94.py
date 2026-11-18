"""1.22.94: an image turn was refused because the user row's history embed
(long: image analysis appended) queued on the single-request Ollama ahead of
the 10s recall embed; the refusal then deleted the user message."""
import asyncio
import inspect

from syne.conversation import Conversation


def _chat_inner_src() -> str:
    return inspect.getsource(Conversation._chat_inner)


def test_user_row_saved_with_deferred_embed():
    src = _chat_inner_src()
    assert src.count('save_message("user"') == 2
    assert src.count("defer_embed=True") == 2


def test_history_embed_starts_after_recall():
    src = _chat_inner_src()
    i = src.index("self.build_context(")
    j = src.index("self._flush_deferred_embed()")
    assert i < j
    # flushed in a finally: a refused recall still embeds the kept row
    assert "finally:" in src[i:j]


def test_save_message_defers_only_on_request():
    src = inspect.getsource(Conversation.save_message)
    assert "defer_embed: bool = False" in src
    assert "self._deferred_embed = (inserted_id, content)" in src


async def test_flush_starts_embed_once_and_clears():
    conv = object.__new__(Conversation)
    seen = []

    async def fake_embed(mid, content):
        seen.append((mid, content))

    conv._embed_message_row = fake_embed
    conv._deferred_embed = (7, "isi pesan")
    conv._flush_deferred_embed()
    assert conv._deferred_embed is None
    conv._flush_deferred_embed()  # nothing pending: no second task
    await asyncio.sleep(0)
    assert seen == [(7, "isi pesan")]


def test_embed_refusal_keeps_user_message():
    src = inspect.getsource(Conversation.chat)
    i = src.index("except EmbeddingUnavailableError")
    j = src.index("except LLMBadRequestError", i)
    assert "_delete_last_user_message()" not in src[i:j]
    # a provider content refusal still drops it
    assert "_delete_last_user_message()" in src[j:j + 900]
