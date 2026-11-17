"""1.22.84: maps exposes the extra Routes / Places (New) / Geocoding v4 features."""

import asyncio

import httpx

from syne.abilities import maps as M


class _FakeClient:
    last = {}
    reply = {}

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None, params=None):
        _FakeClient.last = {"url": url, "json": json, "headers": headers}
        return httpx.Response(200, json=_FakeClient.reply, request=httpx.Request("POST", url))

    async def get(self, url, headers=None, params=None):
        _FakeClient.last = {"url": url, "headers": headers, "params": params}
        return httpx.Response(200, json=_FakeClient.reply, request=httpx.Request("GET", url))


def _run(monkeypatch, reply, params):
    _FakeClient.reply = reply
    _FakeClient.last = {}
    monkeypatch.setattr(M.httpx, "AsyncClient", _FakeClient)
    return asyncio.run(M.MapsAbility().execute(params, {"config": {"api_key": "TESTKEY"}}))


_ROUTE_REPLY = {"routes": [
    {"description": "Tol JORR", "warnings": ["Rute ini melewati tol."],
     "localizedValues": {"distance": {"text": "28,6 km"}, "duration": {"text": "1 jam 7 menit"},
                         "staticDuration": {"text": "49 menit"}},
     "legs": [{"steps": [{"navigationInstruction": {"instructions": "Belok kanan"},
                          "localizedValues": {"distance": {"text": "0,2 km"}}}]}]},
    {"description": "Jl. Sudirman",
     "localizedValues": {"distance": {"text": "24,9 km"}, "duration": {"text": "1 jam 11 menit"}},
     "legs": [{"steps": [{"navigationInstruction": {"instructions": "JANGAN TAMPIL"}}]}]},
]}


# ── route body ────────────────────────────────────────────────────────────

def test_new_modes_mapped():
    assert M._TRAVEL_MODES["two_wheeler"] == "TWO_WHEELER"
    assert M._TRAVEL_MODES["truck"] == "TRUCK"


def test_body_avoid_alternatives_traffic_departure():
    b = M.MapsAbility()._build_route_body({
        "origin": "A", "destination": "B", "mode": "two_wheeler",
        "avoid": ["tolls", "highways"], "alternatives": True,
        "departure_time": "2026-11-18T07:00:00+07:00",
    })
    assert b["travelMode"] == "TWO_WHEELER"
    assert b["routeModifiers"] == {"avoidTolls": True, "avoidHighways": True}
    assert b["computeAlternativeRoutes"] is True
    assert b["routingPreference"] == "TRAFFIC_AWARE"
    assert b["departureTime"] == "2026-11-18T07:00:00+07:00"


def test_body_waypoints_optimize_disable_alternatives():
    b = M.MapsAbility()._build_route_body({
        "origin": "A", "destination": "B", "waypoints": ["X", "-6.2,106.8"],
        "optimize_waypoints": True, "alternatives": True,
    })
    assert b["intermediates"][0] == {"address": "X"}
    assert b["intermediates"][1]["location"]["latLng"]["latitude"] == -6.2
    assert b["optimizeWaypointOrder"] is True
    assert "computeAlternativeRoutes" not in b


def test_body_traffic_off_and_walking_has_no_traffic():
    assert "routingPreference" not in M.MapsAbility()._build_route_body(
        {"origin": "A", "destination": "B", "traffic": False})
    assert "routingPreference" not in M.MapsAbility()._build_route_body(
        {"origin": "A", "destination": "B", "mode": "walking"})


def test_body_transit_options():
    b = M.MapsAbility()._build_route_body({
        "origin": "A", "destination": "B", "mode": "transit",
        "arrival_time": "2026-11-18T09:00:00+07:00",
        "transit_preference": "less_walking", "transit_modes": ["bus", "train"],
    })
    assert b["arrivalTime"] == "2026-11-18T09:00:00+07:00"
    assert b["transitPreferences"] == {"routingPreference": "LESS_WALKING", "allowedTravelModes": ["BUS", "TRAIN"]}
    assert "routingPreference" not in b


def test_body_validation_errors():
    ab = M.MapsAbility()
    assert isinstance(ab._build_route_body({"origin": "A", "destination": "B", "avoid": ["kerikil"]}), str)
    assert isinstance(ab._build_route_body({"origin": "A", "destination": "B", "mode": "walking", "avoid": ["tolls"]}), str)
    assert isinstance(ab._build_route_body({"origin": "A", "destination": "B", "arrival_time": "2026-11-18T09:00:00+07:00"}), str)
    assert isinstance(ab._build_route_body({"origin": "A", "destination": "B",
                                            "departure_time": "2026-11-18T07:00:00+07:00",
                                            "arrival_time": "2026-11-18T09:00:00+07:00", "mode": "transit"}), str)
    assert isinstance(ab._build_route_body({"origin": "A", "destination": "B", "departure_time": "besok pagi"}), str)
    assert isinstance(ab._build_route_body({"origin": "A", "destination": "B", "mode": "transit", "waypoints": ["X"]}), str)
    assert isinstance(ab._build_route_body({"origin": "A", "destination": "B", "mode": "helikopter"}), str)


def test_rfc3339_naive_gets_offset():
    out = M._rfc3339("2026-11-18T07:00:00")
    assert out.startswith("2026-11-18T07:00:00") and ("+" in out[19:] or "-" in out[19:])


# ── directions output ─────────────────────────────────────────────────────

def test_directions_alternatives_output(monkeypatch):
    r = _run(monkeypatch, _ROUTE_REPLY, {"action": "directions", "origin": "Bintaro", "destination": "Monas",
                                         "alternatives": True, "avoid": ["ferries"]})
    out = r["result"]
    assert r["success"]
    assert "Rute utama** via Tol JORR" in out and "Alternatif 1** via Jl. Sudirman" in out
    assert "(tanpa macet 49 menit)" in out
    assert "⚠️ Rute ini melewati tol." in out
    assert "1. Belok kanan (0,2 km)" in out
    assert "JANGAN TAMPIL" not in out
    assert "Hindari: ferries" in out
    assert "routes.warnings" in _FakeClient.last["headers"]["X-Goog-FieldMask"]


def test_directions_waypoint_order_and_transit(monkeypatch):
    reply = {"routes": [{
        "optimizedIntermediateWaypointIndex": [1, 0],
        "localizedValues": {"distance": {"text": "4,6 km"}, "duration": {"text": "21 menit"},
                            "transitFare": {"text": "Rp 18.000"}},
        "legs": [{"steps": [{"transitDetails": {
            "headsign": "Monas", "stopCount": 3,
            "transitLine": {"nameShort": "1A", "vehicle": {"name": {"text": "Bus"}}},
            "stopDetails": {"departureStop": {"name": "Tosari"}, "arrivalStop": {"name": "Monas"}},
        }}]}],
    }]}
    r = _run(monkeypatch, reply, {"action": "directions", "origin": "A", "destination": "B",
                                  "waypoints": ["P", "Q"], "optimize_waypoints": True})
    out = r["result"]
    assert "Urutan singgah optimal: Q → P" in out
    assert "💵 Tarif: Rp 18.000" in out
    assert "🚌 Bus 1A arah Monas: Tosari → Monas (3 halte)" in out


# ── search ────────────────────────────────────────────────────────────────

def test_search_basic_mask_has_no_detail_fields(monkeypatch):
    _run(monkeypatch, {"places": []}, {"action": "search", "query": "kopi"})
    mask = _FakeClient.last["headers"]["X-Goog-FieldMask"]
    assert "places.googleMapsUri" in mask
    assert "places.nationalPhoneNumber" not in mask and "regularOpeningHours" not in mask


def test_search_details_filters_and_output(monkeypatch):
    reply = {"places": [{
        "displayName": {"text": "Gadang Barubah"}, "formattedAddress": "Jl. Menteng Raya",
        "rating": 4.7, "userRatingCount": 2544, "priceLevel": "PRICE_LEVEL_MODERATE",
        "primaryTypeDisplayName": {"text": "Restoran"},
        "currentOpeningHours": {"openNow": True},
        "regularOpeningHours": {"weekdayDescriptions": ["Senin: 10.00–22.00"]},
        "nationalPhoneNumber": "0859-2073-1919", "websiteUri": "https://x.id/",
        "googleMapsUri": "https://maps.google.com/?cid=1",
    }]}
    r = _run(monkeypatch, reply, {"action": "search", "query": "padang", "details": True,
                                  "open_now": True, "min_rating": 4.2, "max_results": 50})
    out = r["result"]
    body = _FakeClient.last["json"]
    assert body["openNow"] is True and body["minRating"] == 4.0 and body["pageSize"] == 20
    assert "places.nationalPhoneNumber" in _FakeClient.last["headers"]["X-Goog-FieldMask"]
    for s in ("Restoran · harga sedang", "Buka sekarang", "Senin: 10.00–22.00", "0859-2073-1919",
              "https://x.id/", "https://maps.google.com/?cid=1"):
        assert s in out


# ── destination ───────────────────────────────────────────────────────────

def test_destination_by_location(monkeypatch):
    reply = {"destinations": [{
        "primary": {"displayName": {"text": "Gandaria City"}, "formattedAddress": "Jl. Sultan Iskandar Muda"},
        "entrances": [{"location": {"latitude": -6.1, "longitude": 106.1}, "tags": ["PREFERRED"]}],
        "navigationPoints": [{"displayName": {"text": "Lobi Utara"},
                              "location": {"latitude": -6.2, "longitude": 106.2},
                              "travelModes": ["DRIVE", "WALK"], "usages": ["DROPOFF", "PICKUP"]}],
        "landmarks": [{"relationalDescription": {"text": "Di samping Gandaria City XXI"},
                       "straightLineDistanceMeters": 57.2}],
    }]}
    r = _run(monkeypatch, reply, {"action": "destination", "lat": -6.244, "lng": 106.7834})
    out = r["result"]
    assert _FakeClient.last["url"] == M._DESTINATIONS_URL
    assert _FakeClient.last["json"]["locationQuery"]["location"] == {"latitude": -6.244, "longitude": 106.7834}
    assert _FakeClient.last["headers"]["X-Goog-Api-Key"] == "TESTKEY"
    assert "Gandaria City" in out and "(utama)" in out
    assert "Lobi Utara — -6.2,106.2 (mobil/jalan kaki, turun/jemput)" in out
    assert "Di samping Gandaria City XXI (~57 m)" in out


def test_destination_by_address_and_empty(monkeypatch):
    r = _run(monkeypatch, {}, {"action": "destination", "address": "Gandaria City"})
    assert r["success"] and "No destination details" in r["result"]
    assert _FakeClient.last["json"]["addressQuery"] == {"addressQuery": "Gandaria City"}


def test_destination_requires_input(monkeypatch):
    r = _run(monkeypatch, {}, {"action": "destination"})
    assert not r["success"]


def test_schema_lists_new_action_and_modes():
    fn = M.MapsAbility().get_schema()["function"]
    props = fn["parameters"]["properties"]
    assert "destination" in props["action"]["enum"]
    assert "two_wheeler" in props["mode"]["enum"] and "truck" in props["mode"]["enum"]
    for k in ("avoid", "alternatives", "waypoints", "optimize_waypoints", "departure_time",
              "arrival_time", "transit_preference", "transit_modes", "open_now", "min_rating", "details"):
        assert k in props


def test_multiline_instruction_and_min_normalised(monkeypatch):
    reply = {"routes": [{
        "localizedValues": {"distance": {"text": "23,6 km"}, "duration": {"text": "57 min"},
                            "staticDuration": {"text": "1 jam 0 min"}},
        "legs": [{"steps": [{"navigationInstruction": {"instructions": "Ambil jalan ke Tomang\nJalan tol"},
                             "localizedValues": {"distance": {"text": "0,9 km"}}}]}],
    }]}
    r = _run(monkeypatch, reply, {"action": "directions", "origin": "A", "destination": "B", "mode": "two_wheeler"})
    out = r["result"]
    assert "57 menit (tanpa macet 1 jam 0 menit)" in out
    assert "1. Ambil jalan ke Tomang · Jalan tol (0,9 km)" in out
    assert out.startswith("🏍️")
