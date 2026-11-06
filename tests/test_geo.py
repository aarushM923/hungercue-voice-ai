import httpx
import respx

from hungercue.matching.geo import NOMINATIM_URL, Geocoder, haversine_km, neighborhood_point, travel_minutes


def test_haversine_known_distance() -> None:
    # Times Square to Grand Central is about 0.8 km.
    assert 0.6 < haversine_km((40.7580, -73.9855), (40.7527, -73.9772)) < 1.0


def test_travel_minutes() -> None:
    a, b = (40.81, -73.95), (40.68, -73.94)
    assert travel_minutes(a, b, has_car=False) > travel_minutes(a, b, has_car=True) > 0
    assert travel_minutes(None, b) == 35.0


def test_neighborhood_lookup() -> None:
    assert neighborhood_point("Harlem") == (40.8116, -73.9465)
    assert neighborhood_point("Bed-Stuy") is not None
    assert neighborhood_point("harlem, new york, ny") is None  # callers pass the part before the comma
    assert neighborhood_point("Atlantis") is None


@respx.mock
async def test_geocoder_uses_table_then_nominatim_and_caches() -> None:
    route = respx.get(NOMINATIM_URL).mock(return_value=httpx.Response(200, json=[{"lat": "40.80", "lon": "-73.96"}]))
    g = Geocoder(min_interval_s=0)
    assert await g("Harlem, New York, NY") == (40.8116, -73.9465)  # table, no HTTP
    assert route.call_count == 0
    assert await g("200 W 112th St") == (40.80, -73.96)
    assert await g("200  w 112th st") == (40.80, -73.96)  # cached (normalised key)
    assert route.call_count == 1
    req = route.calls.last.request
    assert req.url.params["q"] == "200 W 112th St, New York, NY" and "HungerCue" in req.headers["User-Agent"]
    assert await g("") is None
    await g.aclose()


@respx.mock
async def test_geocoder_no_result() -> None:
    respx.get(NOMINATIM_URL).mock(return_value=httpx.Response(200, json=[]))
    assert await Geocoder(min_interval_s=0)("nowhere street brooklyn") is None


def test_canonical_neighborhood_fixes_misheard_names() -> None:
    from hungercue.matching.geo import canonical_neighborhood
    assert canonical_neighborhood("Parksloe") == "Park Slope"
    assert canonical_neighborhood("Williamsburgh") == "Williamsburg"
    assert canonical_neighborhood("Harlem") == "Harlem"  # already known: kept as said
    assert canonical_neighborhood("Atlantis") == "Atlantis"  # unknown and not close: kept
