"""Maps & Places Ability — Google Maps Platform (Places, Directions, Geocoding)."""

import os
import httpx

from .base import Ability

_MAPS_BASE = "https://maps.googleapis.com/maps/api"


class MapsAbility(Ability):
    """Search for places, get directions, and geocode addresses using Google Maps."""

    name = "maps"
    description = "Search for nearby places (restaurants, cafes, etc.), get driving/walking directions between locations, and convert addresses to coordinates or vice versa"
    version = "1.0"

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
        """Search for nearby places using Google Places Text Search."""
        query = params.get("query", "")
        if not query:
            return {"success": False, "error": "Query is required for place search"}

        location = params.get("location", "")  # "lat,lng"
        radius = params.get("radius", 5000)  # meters
        language = params.get("language", "id")

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                request_params = {
                    "query": query,
                    "key": api_key,
                    "language": language,
                }
                if location:
                    request_params["location"] = location
                    request_params["radius"] = radius

                resp = await client.get(
                    f"{_MAPS_BASE}/place/textsearch/json",
                    params=request_params,
                )

                if resp.status_code != 200:
                    return {"success": False, "error": f"API error ({resp.status_code})"}

                data = resp.json()
                if data.get("status") not in ("OK", "ZERO_RESULTS"):
                    return {"success": False, "error": f"Places API error: {data.get('status')} - {data.get('error_message', '')}"}

                results = data.get("results", [])
                if not results:
                    return {"success": True, "result": f"No places found for: {query}"}

                formatted = []
                for i, place in enumerate(results[:8], 1):
                    name = place.get("name", "Unknown")
                    address = place.get("formatted_address", "")
                    rating = place.get("rating", "N/A")
                    total_ratings = place.get("user_ratings_total", 0)
                    status = place.get("business_status", "")
                    location = place.get("geometry", {}).get("location", {})
                    lat = location.get("lat", "")
                    lng = location.get("lng", "")

                    line = f"{i}. **{name}** ⭐ {rating} ({total_ratings} reviews)\n   📍 {address}"
                    if lat and lng:
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
        """Get directions between two points."""
        origin = params.get("origin", "")
        destination = params.get("destination", "")

        if not origin or not destination:
            return {"success": False, "error": "Both 'origin' and 'destination' are required"}

        mode = params.get("mode", "driving")  # driving, walking, bicycling, transit
        language = params.get("language", "id")

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{_MAPS_BASE}/directions/json",
                    params={
                        "origin": origin,
                        "destination": destination,
                        "mode": mode,
                        "key": api_key,
                        "language": language,
                    },
                )

                if resp.status_code != 200:
                    return {"success": False, "error": f"API error ({resp.status_code})"}

                data = resp.json()
                if data.get("status") != "OK":
                    return {"success": False, "error": f"Directions API: {data.get('status')} - {data.get('error_message', '')}"}

                routes = data.get("routes", [])
                if not routes:
                    return {"success": True, "result": "No routes found."}

                route = routes[0]
                leg = route.get("legs", [{}])[0]

                start = leg.get("start_address", origin)
                end = leg.get("end_address", destination)
                distance = leg.get("distance", {}).get("text", "?")
                duration = leg.get("duration", {}).get("text", "?")

                steps = []
                for i, step in enumerate(leg.get("steps", [])[:15], 1):
                    instruction = step.get("html_instructions", "")
                    # Strip HTML tags
                    import re
                    instruction = re.sub(r"<[^>]+>", " ", instruction).strip()
                    step_dist = step.get("distance", {}).get("text", "")
                    steps.append(f"{i}. {instruction} ({step_dist})")

                result = (
                    f"🚗 **{start}** → **{end}**\n"
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
        """Convert address to coordinates."""
        address = params.get("address", "")
        if not address:
            return {"success": False, "error": "Address is required"}

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{_MAPS_BASE}/geocode/json",
                    params={
                        "address": address,
                        "key": api_key,
                        "language": "id",
                    },
                )

                data = resp.json()
                if data.get("status") != "OK":
                    return {"success": False, "error": f"Geocode: {data.get('status')}"}

                result = data["results"][0]
                loc = result["geometry"]["location"]
                formatted = result.get("formatted_address", address)

                return {
                    "success": True,
                    "result": f"📍 **{formatted}**\n🗺️ {loc['lat']},{loc['lng']}",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _reverse_geocode(self, params: dict, api_key: str) -> dict:
        """Convert coordinates to address."""
        lat = params.get("lat")
        lng = params.get("lng")
        latlng = params.get("latlng", "")  # Accept "lat,lng" string too

        if latlng:
            pass
        elif lat and lng:
            latlng = f"{lat},{lng}"
        else:
            return {"success": False, "error": "Provide 'lat'+'lng' or 'latlng' (e.g. '-6.2,106.8')"}

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{_MAPS_BASE}/geocode/json",
                    params={
                        "latlng": latlng,
                        "key": api_key,
                        "language": "id",
                    },
                )

                data = resp.json()
                if data.get("status") != "OK":
                    return {"success": False, "error": f"Reverse geocode: {data.get('status')}"}

                result = data["results"][0]
                formatted = result.get("formatted_address", latlng)

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
                            "description": "Search radius in meters (default 5000, action=search)",
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

    def get_required_config(self) -> list[str]:
        """API key can come from config or environment."""
        return []
