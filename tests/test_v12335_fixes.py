"""v1.23.35 (review C): suspect chunks cannot fill the KG batch."""
import inspect
from datetime import datetime, timezone

from syne.memory import kg_extract as K


def _it(i, t="message"):
    return {"type": t, "id": i}


def test_suspects_go_last_before_the_cut():
    items = [_it(i) for i in range(1, 9)]
    sus = {("message", 1): 0, ("message", 2): 0, ("message", 3): 0}
    got = K.order_batch(items, sus, batch=5)
    assert [x["id"] for x in got] == [4, 5, 6, 7, 8]


def test_only_suspects_left_are_still_tried():
    items = [_it(1), _it(2)]
    sus = {("message", 1): 0, ("message", 2): 0}
    assert [x["id"] for x in K.order_batch(items, sus, batch=5)] == [1, 2]


class _Conn:
    def __init__(self):
        self.limits = []

    async def fetch(self, q, *a):
        self.limits.append(a[3])
        return []


async def test_pending_asks_for_batch_plus_extra():
    c = _Conn()
    s = {"kg.extract_batch": 5, "kg.extract_max_attempts": 3,
         "kg.extract_skip_categories": [], "kg.extract_skip_scheduled": True}
    era = datetime(2026, 1, 1, tzinfo=timezone.utc)
    await K.pending(c, {"memory": 0, "messages": 0}, era, s, extra=3)
    assert c.limits == [8, 8]


def test_worker_wiring():
    src = inspect.getsource(K)
    assert "extra=min(len(suspects), MAX_SUSPECT_EXTRA)" in src
    assert "items = order_batch(items, suspects" in src
    assert "items = items[: limit - used]" not in src
    assert "elif round_ok:" in src and "round_ok = True" in src
