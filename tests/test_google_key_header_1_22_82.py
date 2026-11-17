"""1.22.82: Vertex provider, image_gen and image_analysis send the Google API
key in the x-goog-api-key header, never as ?key= (which leaks into httpx
error messages and logs)."""

import inspect


def _src(mod):
    return inspect.getsource(mod)


def test_vertex_key_in_header():
    from syne.llm import vertex as V
    src = _src(V)
    assert '"key": api_key' not in src
    assert '"key": self.api_key' not in src
    assert src.count('"x-goog-api-key": api_key') == 1          # region probe
    assert src.count('"x-goog-api-key": self.api_key') == 3     # chat, embed, batch


def test_image_gen_key_in_header():
    from syne.abilities import image_gen as G
    src = _src(G)
    assert 'params={"key"' not in src
    assert src.count('"x-goog-api-key": api_key') == 2          # imagen + gemini


def test_image_analysis_key_in_header():
    from syne.abilities import image_analysis as A
    src = _src(A)
    assert 'params={"key"' not in src
    assert src.count('"x-goog-api-key": api_key') == 1
