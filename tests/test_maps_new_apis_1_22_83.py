"""1.22.83: maps ability and Telegram location geocode use Places API (New),
Routes API and Geocoding API v4, with the key in the X-Goog-Api-Key header
(never ?key= in the URL)."""

import asyncio
import inspect

import httpx

from syne.abilities import maps as M


def test_no_key_query_param_anywhere():
    src = inspect.getsource(M)
    assert '"key": api_key' not in src
    assert "maps.googleapis.com" not in src
    assert src.count('"X-Goog-Api-Key": api_key') == 4


def test_telegram_reverse_geocode_uses_v4_header():
    from syne.communication import telegram as T
    src = inspect.getsource(T.TelegramChannel._reverse_geocode) if hasattr(T, "TelegramChannel") else inspect.getsource(T)
    assert "maps.googleapis.com/maps/api/geocode" not in src
    assert "geocode.googleapis.com/v4/geocode/location/" in src
    assert '"X-Goog-Api-Key": api_key' in src


def test_waypoint_latlng_and_address():
    assert M._waypoint("-6.2, 106.8") == {"location": {"latLng": {"latitude": -6.2, "longitude": 106.8}}}
    assert M._waypoint("Monas, Jakarta") == {"address": "Monas, Jakarta"}


def test_travel_modes_map():
    assert M._TRAVEL_MODES == {"driving": "DRIVE", "walking": "WALK", "bicycling": "BICYCLE", "transit": "TRANSIT"}


class _FakeClient:
    """Captures the request and returns a canned response."""
    last = {}

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None, params=None):
        _FakeClient.last = {"url": url, "json": json, "headers": headers, "params": params}
        return httpx.Response(200, json=_FakeClient.reply, request=httpx.Request("POST", url))

    async def get(self, url, headers=None, params=None):
        _FakeClient.last = {"url": url, "headers": headers, "params": params}
        return httpx.Response(200, json=_FakeClient.reply, request=httpx.Request("GET", url))


def _run(monkeypatch, reply, params):
    _FakeClient.reply = reply
    monkeypatch.setattr(M.httpx, "AsyncClient", _FakeClient)
    return asyncio.run(M.MapsAbility().execute(params, {"config": {"api_key": "TESTKEY"}}))


def test_search_places_new(monkeypatch):
    reply = {"places": [{
        "displayName": {"text": "Warung A"}, "formattedAddress": "Jl. X",
        "location": {"latitude": -6.27, "longitude": 106.72},
        "rating": 4.7, "userRatingCount": 10, "businessStatus": "OPERATIONAL",
    }]}
    r = _run(monkeypatch, reply, {"action": "search", "query": "padang", "location": "-6.27,106.72"})
    assert r["success"] and "Warung A" in r["result"] and "-6.27,106.72" in r["result"]
    req = _FakeClient.last
    assert req["url"] == M._PLACES_SEARCH_URL
    assert req["headers"]["X-Goog-Api-Key"] == "TESTKEY"
    assert "places.displayName" in req["headers"]["X-Goog-FieldMask"]
    assert req["json"]["locationBias"]["circle"]["center"] == {"latitude": -6.27, "longitude": 106.72}
    assert "TESTKEY" not in req["url"]


def test_directions_routes(monkeypatch):
    reply = {"routes": [{
        "localizedValues": {"distance": {"text": "30 km"}, "duration": {"text": "1 jam"}},
        "legs": [{"steps": [{
            "navigationInstruction": {"instructions": "Belok kanan"},
            "localizedValues": {"distance": {"text": "0,2 km"}},
        }]}],
    }]}
    r = _run(monkeypatch, reply, {"action": "directions", "origin": "Bintaro", "destination": "-6.17,106.82", "mode": "walking"})
    assert r["success"] and "30 km" in r["result"] and "1 jam" in r["result"] and "Belok kanan (0,2 km)" in r["result"]
    req = _FakeClient.last
    assert req["url"] == M._ROUTES_URL
    assert req["json"]["travelMode"] == "WALK"
    assert req["json"]["origin"] == {"address": "Bintaro"}
    assert req["json"]["destination"]["location"]["latLng"]["latitude"] == -6.17
    assert req["headers"]["X-Goog-Api-Key"] == "TESTKEY"


def test_geocode_v4(monkeypatch):
    reply = {"results": [{"formattedAddress": "Monas", "location": {"latitude": -6.17, "longitude": 106.82}}]}
    r = _run(monkeypatch, reply, {"action": "geocode", "address": "Monas, Jakarta"})
    assert r["success"] and "-6.17,106.82" in r["result"]
    assert _FakeClient.last["url"].endswith("/address/Monas%2C%20Jakarta")
    assert _FakeClient.last["headers"] == {"X-Goog-Api-Key": "TESTKEY"}


def test_reverse_geocode_v4(monkeypatch):
    reply = {"results": [{"formattedAddress": "Jl. Tugu Monas"}]}
    r = _run(monkeypatch, reply, {"action": "reverse_geocode", "lat": -6.175, "lng": 106.827})
    assert r["success"] and "Jl. Tugu Monas" in r["result"]
    assert _FakeClient.last["url"].endswith("/location/-6.175,106.827")


def test_reverse_geocode_rejects_bad_coords(monkeypatch):
    r = _run(monkeypatch, {}, {"action": "reverse_geocode", "latlng": "abc"})
    assert not r["success"]
