"""recall_graph (LLM-guessed relations) is not injected per turn (1.22.93)."""
import inspect

from syne import conversation as C


def test_turn_does_not_call_recall_graph():
    src = inspect.getsource(C)
    assert "recall_graph" not in src
    assert "_do_graph_recall" not in src


def test_chunk_links_still_wired():
    src = inspect.getsource(C)
    assert "recall_chunk_links" in src
    assert "kg.chunk_link_limit" in src
