"""Maps & Places Ability — Google Maps Platform (Places API (New), Routes API, Geocoding API v4).

The API key is sent in the X-Goog-Api-Key header, never as ?key= in the URL,
so it cannot leak into httpx error messages or logs.
"""

import os
import re
from urllib.parse import quote

import httpx

from .base import Ability

_PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
_GEOCODE_V4 = "https://geocode.googleapis.com/v4/geocode"

_PLACES_FIELDS = ",".join([
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.rating",
    "places.userRatingCount",
    "places.businessStatus",
])
_ROUTES_FIELDS = ",".join([
    "routes.localizedValues",
    "routes.legs.steps.navigationInstruction",
    "routes.legs.steps.localizedValues",
])

# Old schema values (kept for the tool schema) -> Routes API travelMode
_TRAVEL_MODES = {
    "driving": "DRIVE",
    "walking": "WALK",
    "bicycling": "BICYCLE",
    "transit": "TRANSIT",
}

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


def _api_error(resp: httpx.Response, label: str) -> dict:
    try:
        err = resp.json().get("error", {})
        detail = f"{err.get('status', '')} - {err.get('message', '')}".strip(" -")
    except Exception:
        detail = ""
    return {"success": False, "error": f"{label} error ({resp.status_code}){': ' + detail if detail else ''}"}


class MapsAbility(Ability):
    """Search for places, get directions, and geocode addresses using Google Maps."""

    name = "maps"
    description = "Search for nearby places (restaurants, cafes, etc.), get driving/walking directions between locations, and convert addresses to coordinates or vice versa"
    version = "2.0"
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
        else:
            return {"success": False, "error": f"Unknown action: {action}. Use: search, directions, geocode, reverse_geocode"}

    async def _search_places(self, params: dict, api_key: str) -> dict:
        """Search places with Places API (New) Text Search."""
        query = params.get("query", "")
        if not query:
            return {"success": False, "error": "Query is required for place search"}

        location = params.get("location", "")  # "lat,lng"
        radius = params.get("radius", 5000)  # meters
        language = params.get("language", "id")

        body: dict = {"textQuery": query, "languageCode": language, "pageSize": 8}
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

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    _PLACES_SEARCH_URL,
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Goog-Api-Key": api_key,
                        "X-Goog-FieldMask": _PLACES_FIELDS,
                    },
                )

            if resp.status_code != 200:
                return _api_error(resp, "Places API")

            places = resp.json().get("places", [])
            if not places:
                return {"success": True, "result": f"No places found for: {query}"}

            formatted = []
            for i, place in enumerate(places[:8], 1):
                name = (place.get("displayName") or {}).get("text", "Unknown")
                address = place.get("formattedAddress", "")
                rating = place.get("rating", "N/A")
                total_ratings = place.get("userRatingCount", 0)
                status = place.get("businessStatus", "")
                loc = place.get("location") or {}
                lat = loc.get("latitude", "")
                lng = loc.get("longitude", "")

                line = f"{i}. **{name}** ⭐ {rating} ({total_ratings} reviews)\n   📍 {address}"
                if lat != "" and lng != "":
                    line += f"\n   🗺️ {lat},{lng}"
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

    async def _get_directions(self, params: dict, api_key: str) -> dict:
        """Get directions between two points with Routes API."""
        origin = params.get("origin", "")
        destination = params.get("destination", "")

        if not origin or not destination:
            return {"success": False, "error": "Both 'origin' and 'destination' are required"}

        mode = params.get("mode", "driving")  # driving, walking, bicycling, transit
        travel_mode = _TRAVEL_MODES.get(mode)
        if not travel_mode:
            return {"success": False, "error": f"Unknown mode: {mode}. Use: {', '.join(_TRAVEL_MODES)}"}
        language = params.get("language", "id")

        body = {
            "origin": _waypoint(origin),
            "destination": _waypoint(destination),
            "travelMode": travel_mode,
            "languageCode": language,
        }

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

            route = routes[0]
            values = route.get("localizedValues") or {}
            distance = (values.get("distance") or {}).get("text", "?")
            duration = (values.get("duration") or {}).get("text", "?")

            steps = []
            legs = route.get("legs") or [{}]
            for i, step in enumerate((legs[0].get("steps") or [])[:15], 1):
                instruction = (step.get("navigationInstruction") or {}).get("instructions", "").strip()
                step_dist = ((step.get("localizedValues") or {}).get("distance") or {}).get("text", "")
                if instruction:
                    steps.append(f"{i}. {instruction} ({step_dist})" if step_dist else f"{i}. {instruction}")

            result = (
                f"🚗 **{origin}** → **{destination}**\n"
                f"📏 Distance: {distance}\n"
                f"⏱️ Duration: {duration}\n"
                f"🚦 Mode: {mode}\n\n"
                f"**Steps:**\n" + "\n".join(steps)
            )

            return {"success": True, "result": result}

        except httpx.TimeoutException:
            return {"success": False, "error": "Request timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _geocode(self, params: dict, api_key: str) -> dict:
        """Convert address to coordinates with Geocoding API v4."""
        address = params.get("address", "")
        if not address:
            return {"success": False, "error": "Address is required"}

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{_GEOCODE_V4}/address/{quote(address, safe='')}",
                    params={"languageCode": "id"},
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
                    params={"languageCode": "id"},
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

    def get_schema(self) -> dict:
        """Return OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "maps",
                "description": "Search for places, get directions, geocode addresses, or reverse-geocode coordinates using Google Maps",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["search", "directions", "geocode", "reverse_geocode"],
                            "description": "What to do: search=find places, directions=get route, geocode=address to coordinates, reverse_geocode=coordinates to address",
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
                            "enum": ["driving", "walking", "bicycling", "transit"],
                            "description": "Travel mode (action=directions, default=driving)",
                        },
                        "address": {
                            "type": "string",
                            "description": "Address to geocode (action=geocode)",
                        },
                        "lat": {
                            "type": "number",
                            "description": "Latitude (action=reverse_geocode)",
                        },
                        "lng": {
                            "type": "number",
                            "description": "Longitude (action=reverse_geocode)",
                        },
                        "latlng": {
                            "type": "string",
                            "description": "Coordinates as 'lat,lng' string (action=reverse_geocode)",
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
                "- Actions: search (places), directions, geocode, reverse_geocode"
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
