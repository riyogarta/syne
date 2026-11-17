"""Maps & Places Ability — Google Maps Platform (Places API (New), Routes API, Geocoding API v4).

The API key is sent in the X-Goog-Api-Key header, never as ?key= in the URL,
so it cannot leak into httpx error messages or logs.
"""

import os
import re
from datetime import datetime
from urllib.parse import quote

import httpx

from .base import Ability

_PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
_GEOCODE_V4 = "https://geocode.googleapis.com/v4/geocode"
_DESTINATIONS_URL = f"{_GEOCODE_V4}/destinations"

_PLACES_BASIC_FIELDS = [
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.rating",
    "places.userRatingCount",
    "places.businessStatus",
    "places.googleMapsUri",
]
# Extra fields are billed at a higher Places SKU, so they are opt-in (details=true).
_PLACES_DETAIL_FIELDS = [
    "places.primaryTypeDisplayName",
    "places.priceLevel",
    "places.currentOpeningHours.openNow",
    "places.regularOpeningHours.weekdayDescriptions",
    "places.nationalPhoneNumber",
    "places.websiteUri",
]
_ROUTES_FIELDS = ",".join([
    "routes.description",
    "routes.routeLabels",
    "routes.warnings",
    "routes.localizedValues",
    "routes.optimizedIntermediateWaypointIndex",
    "routes.legs.stepsOverview",
    "routes.legs.steps.travelMode",
    "routes.legs.steps.navigationInstruction",
    "routes.legs.steps.localizedValues",
    "routes.legs.steps.transitDetails",
    "routes.travelAdvisory.tollInfo",
])
_DESTINATION_FIELDS = ",".join([
    "destinations.primary.displayName",
    "destinations.primary.formattedAddress",
    "destinations.primary.location",
    "destinations.entrances",
    "destinations.navigationPoints",
    "destinations.landmarks.place.displayName",
    "destinations.landmarks.relationalDescription",
    "destinations.landmarks.straightLineDistanceMeters",
])

# Schema values -> Routes API travelMode
_TRAVEL_MODES = {
    "driving": "DRIVE",
    "walking": "WALK",
    "bicycling": "BICYCLE",
    "transit": "TRANSIT",
    "two_wheeler": "TWO_WHEELER",
    "truck": "TRUCK",
}
# Modes where traffic-aware routing is allowed
_TRAFFIC_MODES = {"DRIVE", "TWO_WHEELER"}
_AVOID = {"tolls": "avoidTolls", "highways": "avoidHighways", "ferries": "avoidFerries"}
_TRANSIT_PREFS = {"less_walking": "LESS_WALKING", "fewer_transfers": "FEWER_TRANSFERS"}
_TRANSIT_MODES = {"bus": "BUS", "subway": "SUBWAY", "train": "TRAIN", "light_rail": "LIGHT_RAIL", "rail": "RAIL"}
_PRICE_LEVELS = {
    "PRICE_LEVEL_FREE": "gratis",
    "PRICE_LEVEL_INEXPENSIVE": "murah",
    "PRICE_LEVEL_MODERATE": "sedang",
    "PRICE_LEVEL_EXPENSIVE": "mahal",
    "PRICE_LEVEL_VERY_EXPENSIVE": "sangat mahal",
}
_NAV_USAGES = {"DROPOFF": "turun", "PICKUP": "jemput", "PARKING": "parkir"}
_NAV_MODES = {"DRIVE": "mobil", "WALK": "jalan kaki"}

_LATLNG_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")


def _parse_latlng(value: str) -> tuple[float, float] | None:
    m = _LATLNG_RE.match(value or "")
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _waypoint(value: str) -> dict:
    """Routes API waypoint: coordinates when given 'lat,lng', else an address."""
    ll = _parse_latlng(value)
    if ll:
        return {"location": {"latLng": {"latitude": ll[0], "longitude": ll[1]}}}
    return {"address": value}


def _rfc3339(value: str) -> str:
    """ISO datetime -> RFC 3339 with offset. Naive times use the server's local zone."""
    dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.isoformat()


def _as_list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


def _text(obj) -> str:
    return (obj or {}).get("text", "")


# Indonesian toll roads require an e-toll card; without a toll pass the Routes
# API returns no price. ID_E_TOLL covers all bank e-money cards (same tariff).
_TOLL_PASSES = ["ID_E_TOLL"]
_TOLL_MODES = {"DRIVE"}


def _money_text(m: dict) -> str:
    cur = m.get("currencyCode", "")
    try:
        amount = int(m.get("units", 0) or 0) + int(m.get("nanos", 0) or 0) / 1e9
    except (TypeError, ValueError):
        return ""
    if cur == "IDR":
        return "Rp " + f"{round(amount):,}".replace(",", ".")
    return f"{amount:,.2f} {cur}".strip()


def _toll_text(route: dict) -> str | None:
    """None = route has no toll; '' never returned."""
    advisory = route.get("travelAdvisory") or {}
    if "tollInfo" not in advisory:
        return None
    prices = [t for t in (_money_text(m) for m in advisory["tollInfo"].get("estimatedPrice") or []) if t]
    return " / ".join(prices) if prices else "ada tol, tarif tidak diketahui"


_MIN_RE = re.compile(r"\bmin\b")


def _duration_text(obj, language: str) -> str:
    """Google sometimes answers 'min' instead of 'menit' for languageCode=id."""
    text = _text(obj)
    return _MIN_RE.sub("menit", text) if language.startswith("id") else text


def _api_error(resp: httpx.Response, label: str) -> dict:
    try:
        err = resp.json().get("error", {})
        detail = f"{err.get('status', '')} - {err.get('message', '')}".strip(" -")
    except Exception:
        detail = ""
    return {"success": False, "error": f"{label} error ({resp.status_code}){': ' + detail if detail else ''}"}


def _format_transit_step(step: dict) -> str | None:
    td = step.get("transitDetails") or {}
    if not td:
        return None
    line = td.get("transitLine") or {}
    vehicle = _text((line.get("vehicle") or {}).get("name"))
    name = line.get("nameShort") or line.get("name") or ""
    stops = td.get("stopDetails") or {}
    dep = (stops.get("departureStop") or {}).get("name", "")
    arr = (stops.get("arrivalStop") or {}).get("name", "")
    head = td.get("headsign", "")
    count = td.get("stopCount")
    lv = td.get("localizedValues") or {}
    dep_t = _text((lv.get("departureTime") or {}).get("time"))
    arr_t = _text((lv.get("arrivalTime") or {}).get("time"))
    parts = [p for p in (vehicle, name) if p]
    out = f"🚌 {' '.join(parts) or 'Transit'}"
    if head:
        out += f" arah {head}"
    if dep or arr:
        out += f": {dep or '?'}{' ' + dep_t if dep_t else ''} → {arr or '?'}{' ' + arr_t if arr_t else ''}"
    if count:
        out += f" ({count} halte)"
    return out


class MapsAbility(Ability):
    """Search for places, get directions, and geocode addresses using Google Maps."""

    name = "maps"
    description = "Search for nearby places (restaurants, cafes, etc.), get driving/motorbike/walking/transit directions between locations, convert addresses to coordinates or vice versa, and find a destination's drop-off points and nearby landmarks"
    version = "3.0"
    # 0o444 — everyone may invoke; r-only (geocode/search). Was 0o555 which
    # had spurious x bit under the new gate rule.
    permission = 0o444

    def _get_api_key(self, context: dict) -> str | None:
        config = context.get("config", {})
        return (
            config.get("GOOGLE_MAPS_API_KEY")
            or config.get("GOOGLE_PLACES_API_KEY")
            or config.get("google_maps_api_key")
            or config.get("api_key")
            or os.environ.get("GOOGLE_PLACES_API_KEY")
            or os.environ.get("GOOGLE_MAPS_API_KEY")
        )

    async def execute(self, params: dict, context: dict) -> dict:
        """Route to the appropriate sub-function based on 'action' parameter."""
        action = params.get("action", "search")
        api_key = self._get_api_key(context)

        if not api_key:
            return {"success": False, "error": "Google Maps API key not configured. Set it via: update_ability(action='config', name='maps', config='{\"api_key\": \"YOUR_KEY\"}')"}

        if action == "search":
            return await self._search_places(params, api_key)
        elif action == "directions":
            return await self._get_directions(params, api_key)
        elif action == "geocode":
            return await self._geocode(params, api_key)
        elif action == "reverse_geocode":
            return await self._reverse_geocode(params, api_key)
        elif action == "destination":
            return await self._destination(params, api_key)
        else:
            return {"success": False, "error": f"Unknown action: {action}. Use: search, directions, geocode, reverse_geocode, destination"}

    # ── search ────────────────────────────────────────────────────────────

    async def _search_places(self, params: dict, api_key: str) -> dict:
        """Search places with Places API (New) Text Search."""
        query = params.get("query", "")
        if not query:
            return {"success": False, "error": "Query is required for place search"}

        location = params.get("location", "")  # "lat,lng"
        radius = params.get("radius", 5000)  # meters
        language = params.get("language", "id")
        details = bool(params.get("details", False))
        try:
            max_results = max(1, min(int(params.get("max_results", 8)), 20))
        except (TypeError, ValueError):
            max_results = 8

        body: dict = {"textQuery": query, "languageCode": language, "pageSize": max_results}
        ll = _parse_latlng(location)
        if ll:
            try:
                radius = float(radius)
            except (TypeError, ValueError):
                radius = 5000.0
            body["locationBias"] = {"circle": {
                "center": {"latitude": ll[0], "longitude": ll[1]},
                "radius": max(1.0, min(radius, 50000.0)),
            }}
        if params.get("open_now"):
            body["openNow"] = True
        if params.get("min_rating") not in (None, ""):
            try:
                # Places accepts 0.0-5.0 in 0.5 steps
                body["minRating"] = max(0.0, min(round(float(params["min_rating"]) * 2) / 2, 5.0))
            except (TypeError, ValueError):
                return {"success": False, "error": "min_rating must be a number 0-5"}

        fields = _PLACES_BASIC_FIELDS + (_PLACES_DETAIL_FIELDS if details else [])

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    _PLACES_SEARCH_URL,
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Goog-Api-Key": api_key,
                        "X-Goog-FieldMask": ",".join(fields),
                    },
                )

            if resp.status_code != 200:
                return _api_error(resp, "Places API")

            places = resp.json().get("places", [])
            if not places:
                return {"success": True, "result": f"No places found for: {query}"}

            formatted = []
            for i, place in enumerate(places[:max_results], 1):
                name = _text(place.get("displayName")) or "Unknown"
                address = place.get("formattedAddress", "")
                rating = place.get("rating", "N/A")
                total_ratings = place.get("userRatingCount", 0)
                status = place.get("businessStatus", "")
                loc = place.get("location") or {}
                lat = loc.get("latitude", "")
                lng = loc.get("longitude", "")

                line = f"{i}. **{name}** ⭐ {rating} ({total_ratings} reviews)"
                kind = _text(place.get("primaryTypeDisplayName"))
                price = _PRICE_LEVELS.get(place.get("priceLevel", ""), "")
                if kind or price:
                    line += "\n   🏷️ " + " · ".join(p for p in (kind, f"harga {price}" if price else "") if p)
                line += f"\n   📍 {address}"
                if lat != "" and lng != "":
                    line += f"\n   🗺️ {lat},{lng}"
                open_now = (place.get("currentOpeningHours") or {}).get("openNow")
                if open_now is not None:
                    line += f"\n   🕒 {'Buka sekarang' if open_now else 'Tutup sekarang'}"
                hours = (place.get("regularOpeningHours") or {}).get("weekdayDescriptions") or []
                if hours:
                    line += "\n   📅 " + "; ".join(hours)
                if place.get("nationalPhoneNumber"):
                    line += f"\n   ☎️ {place['nationalPhoneNumber']}"
                if place.get("websiteUri"):
                    line += f"\n   🌐 {place['websiteUri']}"
                if place.get("googleMapsUri"):
                    line += f"\n   🔗 {place['googleMapsUri']}"
                if status and status != "OPERATIONAL":
                    line += f"\n   ⚠️ {status}"
                formatted.append(line)

            return {
                "success": True,
                "result": f"Places for \"{query}\":\n\n" + "\n\n".join(formatted),
            }

        except httpx.TimeoutException:
            return {"success": False, "error": "Request timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── directions ────────────────────────────────────────────────────────

    def _build_route_body(self, params: dict) -> dict | str:
        """Build the computeRoutes body, or return an error string."""
        origin = params.get("origin", "")
        destination = params.get("destination", "")
        if not origin or not destination:
            return "Both 'origin' and 'destination' are required"

        mode = params.get("mode", "driving")
        travel_mode = _TRAVEL_MODES.get(mode)
        if not travel_mode:
            return f"Unknown mode: {mode}. Use: {', '.join(_TRAVEL_MODES)}"

        body: dict = {
            "origin": _waypoint(origin),
            "destination": _waypoint(destination),
            "travelMode": travel_mode,
            "languageCode": params.get("language", "id"),
            "regionCode": "ID",
        }

        waypoints = _as_list(params.get("waypoints"))
        if waypoints:
            if travel_mode == "TRANSIT":
                return "Waypoints are not supported for transit"
            if len(waypoints) > 25:
                return "At most 25 waypoints"
            body["intermediates"] = [_waypoint(w) for w in waypoints]
            if params.get("optimize_waypoints"):
                body["optimizeWaypointOrder"] = True

        avoid = _as_list(params.get("avoid"))
        bad = [a for a in avoid if a not in _AVOID]
        if bad:
            return f"Unknown avoid value(s): {', '.join(bad)}. Use: {', '.join(_AVOID)}"
        if avoid:
            if travel_mode not in ("DRIVE", "TWO_WHEELER", "TRUCK"):
                return "'avoid' only applies to driving, two_wheeler and truck"
            body["routeModifiers"] = {_AVOID[a]: True for a in avoid}

        # Alternatives are not available together with intermediate waypoints.
        if params.get("alternatives") and not waypoints:
            body["computeAlternativeRoutes"] = True

        dep = params.get("departure_time")
        arr = params.get("arrival_time")
        if dep and arr:
            return "Use either departure_time or arrival_time, not both"
        try:
            if dep:
                body["departureTime"] = _rfc3339(dep)
            if arr:
                if travel_mode != "TRANSIT":
                    return "arrival_time only works with mode=transit"
                body["arrivalTime"] = _rfc3339(arr)
        except ValueError:
            return "departure_time/arrival_time must be ISO datetime, e.g. 2026-11-18T07:00:00+07:00"

        if travel_mode in _TRAFFIC_MODES and params.get("traffic", True):
            body["routingPreference"] = "TRAFFIC_AWARE"

        if travel_mode in _TOLL_MODES and "tolls" not in avoid:
            body["extraComputations"] = ["TOLLS"]
            body.setdefault("routeModifiers", {})["tollPasses"] = list(_TOLL_PASSES)

        if travel_mode == "TRANSIT":
            prefs: dict = {}
            tp = params.get("transit_preference")
            if tp:
                if tp not in _TRANSIT_PREFS:
                    return f"Unknown transit_preference: {tp}. Use: {', '.join(_TRANSIT_PREFS)}"
                prefs["routingPreference"] = _TRANSIT_PREFS[tp]
            tm = _as_list(params.get("transit_modes"))
            bad = [m for m in tm if m not in _TRANSIT_MODES]
            if bad:
                return f"Unknown transit_modes: {', '.join(bad)}. Use: {', '.join(_TRANSIT_MODES)}"
            if tm:
                prefs["allowedTravelModes"] = [_TRANSIT_MODES[m] for m in tm]
            if prefs:
                body["transitPreferences"] = prefs

        return body

    async def _get_directions(self, params: dict, api_key: str) -> dict:
        """Get directions between two points with Routes API."""
        body = self._build_route_body(params)
        if isinstance(body, str):
            return {"success": False, "error": body}

        origin = params.get("origin", "")
        destination = params.get("destination", "")
        mode = params.get("mode", "driving")
        waypoints = _as_list(params.get("waypoints"))

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    _ROUTES_URL,
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Goog-Api-Key": api_key,
                        "X-Goog-FieldMask": _ROUTES_FIELDS,
                    },
                )

            if resp.status_code != 200:
                return _api_error(resp, "Routes API")

            routes = resp.json().get("routes", [])
            if not routes:
                return {"success": True, "result": "No routes found."}

            icon = {"walking": "🚶", "bicycling": "🚲", "transit": "🚆", "two_wheeler": "🏍️", "truck": "🚚"}.get(mode, "🚗")
            header = f"{icon} **{origin}** → **{destination}**\n🚦 Mode: {mode}"
            if body.get("departureTime"):
                header += f"\n🕐 Berangkat: {body['departureTime']}"
            if body.get("arrivalTime"):
                header += f"\n🕐 Tiba: {body['arrivalTime']}"
            if _as_list(params.get("avoid")):
                header += "\n⛔ Hindari: " + ", ".join(_as_list(params.get("avoid")))

            blocks = [header]
            for idx, route in enumerate(routes):
                blocks.append(self._format_route(route, idx, len(routes), waypoints, body.get("languageCode", "id")))

            return {"success": True, "result": "\n\n".join(blocks)}

        except httpx.TimeoutException:
            return {"success": False, "error": "Request timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _format_route(self, route: dict, idx: int, total: int, waypoints: list, language: str = "id") -> str:
        values = route.get("localizedValues") or {}
        distance = _text(values.get("distance")) or "?"
        duration = _duration_text(values.get("duration"), language) or "?"
        static = _duration_text(values.get("staticDuration"), language)
        fare = _text(values.get("transitFare"))

        title = "Rute utama" if idx == 0 else f"Alternatif {idx}"
        lines = [f"**{title}**" + (f" via {route['description']}" if route.get("description") else "") if total > 1
                 else (f"Via {route['description']}" if route.get("description") else "")]
        lines.append(f"📏 Distance: {distance}")
        dur = f"⏱️ Duration: {duration}"
        if static and static != duration:
            dur += f" (tanpa macet {static})"
        lines.append(dur)
        if fare:
            lines.append(f"💵 Tarif: {fare}")
        toll = _toll_text(route)
        if toll:
            lines.append(f"💳 Tol (e-toll): {toll}")
        order = route.get("optimizedIntermediateWaypointIndex")
        if order and waypoints and order != list(range(len(waypoints))):
            lines.append("🔀 Urutan singgah optimal: " + " → ".join(waypoints[i] for i in order if 0 <= i < len(waypoints)))
        for w in route.get("warnings") or []:
            lines.append(f"⚠️ {w}")

        # Only the first route gets turn-by-turn steps.
        if idx == 0:
            steps_out = []
            n = 0
            for leg in route.get("legs") or []:
                for step in leg.get("steps") or []:
                    if n >= 25:
                        break
                    transit = _format_transit_step(step)
                    if transit:
                        n += 1
                        steps_out.append(f"{n}. {transit}")
                        continue
                    instruction = " · ".join(
                        part.strip()
                        for part in ((step.get("navigationInstruction") or {}).get("instructions") or "").splitlines()
                        if part.strip()
                    )
                    if not instruction:
                        continue
                    n += 1
                    step_dist = _text((step.get("localizedValues") or {}).get("distance"))
                    steps_out.append(f"{n}. {instruction} ({step_dist})" if step_dist else f"{n}. {instruction}")
            if steps_out:
                lines.append("\n**Steps:**\n" + "\n".join(steps_out))

        return "\n".join(l for l in lines if l)

    # ── geocode ───────────────────────────────────────────────────────────

    async def _geocode(self, params: dict, api_key: str) -> dict:
        """Convert address to coordinates with Geocoding API v4."""
        address = params.get("address", "")
        if not address:
            return {"success": False, "error": "Address is required"}

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{_GEOCODE_V4}/address/{quote(address, safe='')}",
                    params={"languageCode": params.get("language", "id")},
                    headers={"X-Goog-Api-Key": api_key},
                )

            if resp.status_code != 200:
                return _api_error(resp, "Geocode")

            results = resp.json().get("results", [])
            if not results:
                return {"success": False, "error": f"Geocode: no results for {address}"}

            result = results[0]
            loc = result.get("location") or {}
            formatted = result.get("formattedAddress", address)

            return {
                "success": True,
                "result": f"📍 **{formatted}**\n🗺️ {loc.get('latitude')},{loc.get('longitude')}",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _reverse_geocode(self, params: dict, api_key: str) -> dict:
        """Convert coordinates to address with Geocoding API v4."""
        lat = params.get("lat")
        lng = params.get("lng")
        latlng = params.get("latlng", "")  # Accept "lat,lng" string too

        if latlng:
            pass
        elif lat is not None and lng is not None:
            latlng = f"{lat},{lng}"
        else:
            return {"success": False, "error": "Provide 'lat'+'lng' or 'latlng' (e.g. '-6.2,106.8')"}

        ll = _parse_latlng(latlng)
        if not ll:
            return {"success": False, "error": f"Invalid coordinates: {latlng}"}

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{_GEOCODE_V4}/location/{ll[0]},{ll[1]}",
                    params={"languageCode": params.get("language", "id")},
                    headers={"X-Goog-Api-Key": api_key},
                )

            if resp.status_code != 200:
                return _api_error(resp, "Reverse geocode")

            results = resp.json().get("results", [])
            if not results:
                return {"success": False, "error": f"Reverse geocode: no results for {latlng}"}

            formatted = results[0].get("formattedAddress", latlng)

            return {
                "success": True,
                "result": f"📍 {latlng} → **{formatted}**",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── destination (SearchDestinations) ─────────────────────────────────

    async def _destination(self, params: dict, api_key: str) -> dict:
        """Entrances, drop-off/pick-up points and nearby landmarks for a destination."""
        address = params.get("address", "")
        latlng = params.get("latlng", "")
        if not latlng and params.get("lat") is not None and params.get("lng") is not None:
            latlng = f"{params['lat']},{params['lng']}"

        body: dict = {"languageCode": params.get("language", "id"), "regionCode": "ID"}
        if latlng:
            ll = _parse_latlng(latlng)
            if not ll:
                return {"success": False, "error": f"Invalid coordinates: {latlng}"}
            body["locationQuery"] = {"location": {"latitude": ll[0], "longitude": ll[1]}}
            label = latlng
        elif address:
            body["addressQuery"] = {"addressQuery": address}
            label = address
        else:
            return {"success": False, "error": "Provide 'address' (full street address works best) or 'lat'+'lng' / 'latlng'"}

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    _DESTINATIONS_URL,
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Goog-Api-Key": api_key,
                        "X-Goog-FieldMask": _DESTINATION_FIELDS,
                    },
                )

            if resp.status_code != 200:
                return _api_error(resp, "Destinations")

            dests = resp.json().get("destinations", [])
            if not dests:
                return {"success": True, "result": (
                    f"No destination details for: {label}. Try a full street address "
                    "or coordinates; place names alone are often not resolved."
                )}

            d = dests[0]
            primary = d.get("primary") or {}
            name = _text(primary.get("displayName"))
            addr = primary.get("formattedAddress", "")
            out = [f"📍 **{name or label}**" + (f"\n   {addr}" if addr else "")]

            entrances = d.get("entrances") or []
            if entrances:
                out.append("🚪 **Pintu masuk:**")
                for e in entrances[:5]:
                    loc = e.get("location") or {}
                    tag = " (utama)" if "PREFERRED" in (e.get("tags") or []) else ""
                    out.append(f"- {loc.get('latitude')},{loc.get('longitude')}{tag}")

            navs = d.get("navigationPoints") or []
            if navs:
                out.append("🚗 **Titik turun/jemput:**")
                for n in navs[:6]:
                    loc = n.get("location") or {}
                    nm = _text(n.get("displayName"))
                    modes = "/".join(_NAV_MODES.get(m, m.lower()) for m in n.get("travelModes") or [])
                    uses = "/".join(_NAV_USAGES.get(u, u.lower()) for u in n.get("usages") or [])
                    desc = ", ".join(p for p in (modes, uses) if p)
                    out.append(f"- {nm + ' — ' if nm else ''}{loc.get('latitude')},{loc.get('longitude')}" + (f" ({desc})" if desc else ""))

            landmarks = d.get("landmarks") or []
            if landmarks:
                out.append("🏛️ **Patokan:**")
                for lm in landmarks[:5]:
                    desc = _text(lm.get("relationalDescription")) or _text((lm.get("place") or {}).get("displayName"))
                    dist = lm.get("straightLineDistanceMeters")
                    out.append(f"- {desc}" + (f" (~{round(dist)} m)" if isinstance(dist, (int, float)) else ""))

            if len(out) == 1:
                out.append("(Google has no entrance, drop-off or landmark data for this destination.)")

            return {"success": True, "result": "\n".join(out)}

        except httpx.TimeoutException:
            return {"success": False, "error": "Request timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── schema ────────────────────────────────────────────────────────────

    def get_schema(self) -> dict:
        """Return OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "maps",
                "description": (
                    "Google Maps: search places (optionally with opening hours, phone, website, price), "
                    "directions (car, motorbike, truck, walking, bicycle, public transit; traffic-aware, "
                    "alternatives, avoid tolls/highways/ferries, waypoints, departure/arrival time, transit fare), "
                    "geocode, reverse geocode, and destination details (drop-off/pick-up points, entrances, nearby landmarks)"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["search", "directions", "geocode", "reverse_geocode", "destination"],
                            "description": "search=find places, directions=get route, geocode=address to coordinates, reverse_geocode=coordinates to address, destination=drop-off/pick-up points, entrances and landmarks of a destination",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query for places (action=search). E.g. 'restoran padang dekat Bintaro'",
                        },
                        "location": {
                            "type": "string",
                            "description": "Center point as 'lat,lng' for nearby search (action=search)",
                        },
                        "radius": {
                            "type": "integer",
                            "description": "Search radius in meters (default 5000, max 50000, action=search)",
                        },
                        "open_now": {
                            "type": "boolean",
                            "description": "Only places open right now (action=search)",
                        },
                        "min_rating": {
                            "type": "number",
                            "description": "Minimum rating 0-5 (action=search)",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Number of places, 1-20 (default 8, action=search)",
                        },
                        "details": {
                            "type": "boolean",
                            "description": "Include opening hours, open-now status, phone, website, category and price level (action=search). Costs more per request; use when the user asks for these.",
                        },
                        "origin": {
                            "type": "string",
                            "description": "Starting point — address or lat,lng (action=directions)",
                        },
                        "destination": {
                            "type": "string",
                            "description": "End point — address or lat,lng (action=directions)",
                        },
                        "mode": {
                            "type": "string",
                            "enum": list(_TRAVEL_MODES),
                            "description": "Travel mode (action=directions, default=driving). two_wheeler = motorbike",
                        },
                        "waypoints": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Stops between origin and destination, address or lat,lng (action=directions, not for transit)",
                        },
                        "optimize_waypoints": {
                            "type": "boolean",
                            "description": "Let Google reorder waypoints for the shortest trip (action=directions)",
                        },
                        "avoid": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(_AVOID)},
                            "description": "Avoid tolls/highways/ferries (action=directions; driving, two_wheeler, truck)",
                        },
                        "alternatives": {
                            "type": "boolean",
                            "description": "Also return alternative routes (action=directions; ignored with waypoints)",
                        },
                        "departure_time": {
                            "type": "string",
                            "description": "ISO datetime to leave at, e.g. 2026-11-18T07:00:00+07:00 (action=directions). Must be in the future for traffic estimates",
                        },
                        "arrival_time": {
                            "type": "string",
                            "description": "ISO datetime to arrive by (action=directions, mode=transit only)",
                        },
                        "traffic": {
                            "type": "boolean",
                            "description": "Traffic-aware duration for driving/two_wheeler (default true)",
                        },
                        "transit_preference": {
                            "type": "string",
                            "enum": list(_TRANSIT_PREFS),
                            "description": "Transit routing preference (action=directions, mode=transit)",
                        },
                        "transit_modes": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(_TRANSIT_MODES)},
                            "description": "Allowed transit vehicles (action=directions, mode=transit)",
                        },
                        "address": {
                            "type": "string",
                            "description": "Address (action=geocode or destination; for destination a full street address works best)",
                        },
                        "lat": {
                            "type": "number",
                            "description": "Latitude (action=reverse_geocode or destination)",
                        },
                        "lng": {
                            "type": "number",
                            "description": "Longitude (action=reverse_geocode or destination)",
                        },
                        "latlng": {
                            "type": "string",
                            "description": "Coordinates as 'lat,lng' string (action=reverse_geocode or destination)",
                        },
                        "language": {
                            "type": "string",
                            "description": "Response language (default 'id' for Indonesian)",
                        },
                    },
                    "required": ["action"],
                },
            },
        }

    def get_guide(self, enabled: bool, config: dict) -> str:
        has_key = bool(
            config.get("GOOGLE_MAPS_API_KEY")
            or config.get("GOOGLE_PLACES_API_KEY")
            or config.get("google_maps_api_key")
            or config.get("api_key")
        )
        if enabled and has_key:
            return (
                "- Status: **ready** (Google Maps Platform)\n"
                "- Actions: search (places; `details=true` for hours/phone/website), "
                "directions (modes incl. two_wheeler=motorbike, transit with fare; avoid, alternatives, "
                "waypoints, departure_time/arrival_time), geocode, reverse_geocode, "
                "destination (drop-off/pick-up points, entrances, landmarks)"
            )
        reasons = []
        if not enabled:
            reasons.append("ability disabled")
        if not has_key:
            reasons.append("no Google Maps API key")
        return (
            f"- Status: **not ready** ({', '.join(reasons)})\n"
            "- Setup: `update_ability(action='config', name='maps', "
            "config='{\"api_key\": \"GOOGLE_MAPS_KEY\"}')`\n"
            "- The key's Google Cloud project needs Places API (New), Routes API "
            "and Geocoding API enabled"
        )

    def get_required_config(self) -> list[str]:
        """API key can come from config or environment."""
        return []
