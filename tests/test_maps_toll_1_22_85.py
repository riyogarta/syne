"""1.22.85: maps shows toll prices (Indonesia needs the ID_E_TOLL pass)."""

from syne.abilities import maps as M
from tests.test_maps_features_1_22_84 import _FakeClient, _run  # noqa: F401


def test_drive_requests_tolls_with_e_toll():
    b = M.MapsAbility()._build_route_body({"origin": "A", "destination": "B"})
    assert b["extraComputations"] == ["TOLLS"]
    assert b["routeModifiers"] == {"tollPasses": ["ID_E_TOLL"]}


def test_toll_merged_with_avoid_highways():
    b = M.MapsAbility()._build_route_body({"origin": "A", "destination": "B", "avoid": ["highways"]})
    assert b["routeModifiers"] == {"avoidHighways": True, "tollPasses": ["ID_E_TOLL"]}


def test_no_toll_request_when_avoiding_tolls_or_not_driving():
    ab = M.MapsAbility()
    b = ab._build_route_body({"origin": "A", "destination": "B", "avoid": ["tolls"]})
    assert "extraComputations" not in b and "tollPasses" not in b["routeModifiers"]
    for mode in ("two_wheeler", "truck", "walking", "transit", "bicycling"):
        b = ab._build_route_body({"origin": "A", "destination": "B", "mode": mode})
        assert "extraComputations" not in b
        assert "tollPasses" not in (b.get("routeModifiers") or {})


def test_money_text():
    assert M._money_text({"currencyCode": "IDR", "units": "21500"}) == "Rp 21.500"
    assert M._money_text({"currencyCode": "IDR", "units": "1250000"}) == "Rp 1.250.000"
    assert M._money_text({"currencyCode": "USD", "units": "4", "nanos": 400000000}) == "4.40 USD"


def test_toll_text_states():
    assert M._toll_text({}) is None
    assert M._toll_text({"travelAdvisory": {}}) is None
    assert M._toll_text({"travelAdvisory": {"tollInfo": {}}}) == "ada tol, tarif tidak diketahui"
    assert M._toll_text({"travelAdvisory": {"tollInfo": {"estimatedPrice": [
        {"currencyCode": "IDR", "units": "17000"}]}}}) == "Rp 17.000"


def test_directions_output_shows_toll_and_header_unaffected(monkeypatch):
    reply = {"routes": [
        {"description": "Tol Jakarta-Tangerang",
         "localizedValues": {"distance": {"text": "30,5 km"}, "duration": {"text": "45 menit"}},
         "travelAdvisory": {"tollInfo": {"estimatedPrice": [{"currencyCode": "IDR", "units": "21500"}]}}},
        {"description": "Jl. Raden Patah",
         "localizedValues": {"distance": {"text": "25,2 km"}, "duration": {"text": "1 jam"}},
         "travelAdvisory": {}},
    ]}
    r = _run(monkeypatch, reply, {"action": "directions", "origin": "Bintaro", "destination": "Monas",
                                  "alternatives": True})
    out = r["result"]
    assert "💳 Tol (e-toll): Rp 21.500" in out
    assert out.count("💳") == 1
    assert "Hindari" not in out
    assert "routes.travelAdvisory.tollInfo" in _FakeClient.last["headers"]["X-Goog-FieldMask"]
    assert _FakeClient.last["json"]["routeModifiers"]["tollPasses"] == ["ID_E_TOLL"]
