"""Locations and travel-time estimates.

Geocoding (address -> latitude/longitude) uses OpenStreetMap's free Nominatim
service. Its usage policy allows at most 1 request per second and asks for an
identifying User-Agent, so we rate-limit, cache every answer, and fall back to
a built-in table of NYC neighbourhood centres (good enough for volunteers, who
tell us a neighbourhood rather than an address).

Travel time: we don't pay for a routing API. Straight-line (haversine)
distance times a detour factor, divided by a typical city speed, is a decent
estimate for ranking candidates, which is all matching needs. The interface
(`travel_minutes`) can be swapped for Google's Distance Matrix later.
"""

from __future__ import annotations

import asyncio
import difflib
import math
import re

import httpx

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "HungerCue/2.0 (food rescue dispatch; +https://github.com/aarushM923/hungercue-voice-ai)"
NYC_VIEWBOX = "-74.26,40.92,-73.70,40.49"  # left,top,right,bottom: prefer NYC results

DETOUR_FACTOR = 1.35  # streets aren't straight lines
CAR_KMH = 18.0  # average NYC driving speed including traffic
TRANSIT_KMH = 14.0  # subway/bus/bike average door to door
TRANSIT_OVERHEAD_MIN = 8.0  # walking to the station and waiting
UNKNOWN_LOCATION_MIN = 35.0  # pessimistic guess when we can't place someone

Point = tuple[float, float]

# Approximate centres of common NYC neighbourhoods (lat, lon).
NEIGHBORHOODS: dict[str, Point] = {
    "harlem": (40.8116, -73.9465), "east harlem": (40.7957, -73.9389), "morningside heights": (40.8100, -73.9625),
    "upper west side": (40.7870, -73.9754), "upper east side": (40.7736, -73.9566), "midtown": (40.7549, -73.9840),
    "chelsea": (40.7465, -74.0014), "greenwich village": (40.7336, -74.0027), "east village": (40.7265, -73.9815),
    "lower east side": (40.7150, -73.9843), "soho": (40.7233, -74.0030), "tribeca": (40.7163, -74.0086),
    "financial district": (40.7075, -74.0113), "chinatown": (40.7158, -73.9970), "washington heights": (40.8417, -73.9394),
    "inwood": (40.8677, -73.9212), "hells kitchen": (40.7638, -73.9918), "murray hill": (40.7479, -73.9757),
    "astoria": (40.7644, -73.9235), "long island city": (40.7447, -73.9485), "jackson heights": (40.7557, -73.8831),
    "flushing": (40.7675, -73.8331), "jamaica": (40.7027, -73.7890), "williamsburg": (40.7081, -73.9571),
    "greenpoint": (40.7304, -73.9515), "bushwick": (40.6944, -73.9213), "bed stuy": (40.6872, -73.9418),
    "bedford stuyvesant": (40.6872, -73.9418), "crown heights": (40.6694, -73.9422), "park slope": (40.6710, -73.9814),
    "downtown brooklyn": (40.6928, -73.9903), "sunset park": (40.6455, -74.0124), "flatbush": (40.6415, -73.9594),
    "south bronx": (40.8160, -73.9200), "mott haven": (40.8091, -73.9229), "fordham": (40.8615, -73.8904),
    "riverdale": (40.8999, -73.9123), "st george": (40.6437, -74.0736), "staten island": (40.5795, -74.1502),
    "manhattan": (40.7831, -73.9712), "brooklyn": (40.6782, -73.9442), "queens": (40.7282, -73.7949),
    "bronx": (40.8448, -73.8648), "the bronx": (40.8448, -73.8648),
}


def haversine_km(a: Point, b: Point) -> float:
    """Great-circle distance between two lat/lon points, in kilometres."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def travel_minutes(a: Point | None, b: Point | None, has_car: bool = True) -> float:
    if a is None or b is None:
        return UNKNOWN_LOCATION_MIN
    km = haversine_km(a, b) * DETOUR_FACTOR
    if has_car:
        return km / CAR_KMH * 60
    return km / TRANSIT_KMH * 60 + TRANSIT_OVERHEAD_MIN


def canonical_neighborhood(name: str) -> str:
    """Snap a heard neighbourhood name to a known one: speech recognition writes
    "Park Slope" as "Parksloe" sometimes, and a misspelt name can't be located,
    so that volunteer would never be matched. Unknown names are kept as said."""
    key = re.sub(r"[^a-z ]", "", name.lower().replace("-", " ")).strip()
    if key in NEIGHBORHOODS:
        return name.strip()
    squashed = {k.replace(" ", ""): k for k in NEIGHBORHOODS}
    match = difflib.get_close_matches(key.replace(" ", ""), list(squashed), n=1, cutoff=0.8)
    return squashed[match[0]].title() if match else name.strip()


def neighborhood_point(name: str) -> Point | None:
    key = re.sub(r"[^a-z ]", "", name.lower().replace("-", " ")).strip()
    key = re.sub(r",?\s*(new york|ny|nyc)$", "", key).strip()
    return NEIGHBORHOODS.get(key)


class Geocoder:
    """Address -> (lat, lon) with caching and Nominatim's 1 request/second limit."""

    def __init__(self, client: httpx.AsyncClient | None = None, min_interval_s: float = 1.0) -> None:
        self.client = client or httpx.AsyncClient(timeout=5.0, headers={"User-Agent": USER_AGENT})
        self.cache: dict[str, Point | None] = {}
        self.lock = asyncio.Lock()
        self.min_interval_s = min_interval_s
        self._last_request = 0.0

    async def __call__(self, text: str) -> Point | None:
        key = " ".join(text.lower().split())
        if not key:
            return None
        if key in self.cache:
            return self.cache[key]
        # Strip a trailing ", New York, NY" so "Harlem, New York, NY" hits the table.
        point = neighborhood_point(key.split(",")[0]) if "," in key else neighborhood_point(key)
        if point is None:
            point = await self._nominatim(text)
        self.cache[key] = point
        return point

    async def _nominatim(self, text: str) -> Point | None:
        query = text if re.search(r"new york|nyc|brooklyn|queens|bronx|staten", text, re.I) else f"{text}, New York, NY"
        async with self.lock:  # one request at a time, spaced >= 1 s apart
            loop = asyncio.get_running_loop()
            wait = self._last_request + self.min_interval_s - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = loop.time()
            r = await self.client.get(
                NOMINATIM_URL,
                params={"q": query, "format": "json", "limit": 1, "viewbox": NYC_VIEWBOX, "countrycodes": "us"},
            )
        r.raise_for_status()
        results = r.json()
        if not results:
            return None
        return float(results[0]["lat"]), float(results[0]["lon"])

    async def aclose(self) -> None:
        await self.client.aclose()
