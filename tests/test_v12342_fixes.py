"""v1.23.42: a location/venue in a replied-to Telegram message reaches the LLM.

Telegram cannot caption a location, so in groups the only way to ask about one
is to reply to it with a mention. Before this fix the replied location was
dropped from the reply context, so Molt could not read it.
"""
from types import SimpleNamespace as N

from syne.communication.telegram import TelegramChannel

LOC = N(latitude=-6.269419, longitude=106.625033)


def _update(**kw):
    base = dict(
        text=None, caption=None, photo=None, document=None, voice=None,
        audio=None, video=None, video_note=None, venue=None, location=None,
        from_user=N(is_bot=False, first_name="Riyo", username="riyo", id=1),
    )
    base.update(kw)
    return N(message=N(reply_to_message=N(**base), quote=None))


def test_replied_location_in_context():
    r = TelegramChannel._extract_reply_context_raw(_update(location=LOC))
    assert r is not None
    assert r["body"] == "[Location shared: lat: -6.269419, lng: 106.625033]"


def test_replied_venue_in_context():
    venue = N(title="Ruko Hudson", address="Gading Serpong", location=LOC)
    r = TelegramChannel._extract_reply_context_raw(_update(location=LOC, venue=venue))
    assert r["body"] == (
        "[Location shared: Ruko Hudson \u2014 Gading Serpong "
        "(lat: -6.269419, lng: 106.625033)]"
    )


def test_plain_text_reply_unchanged():
    r = TelegramChannel._extract_reply_context_raw(_update(text="halo"))
    assert r["body"] == "halo"
