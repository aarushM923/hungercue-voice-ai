"""Scoring a (shelter, volunteer) pair for one donation.

    score(s, v) = w1*fit(d, s) + w2*overlap(d, s) - w3*dist(d, v) - w4*dist(v, s)

- fit(d, s)     in [0, 1]: does the shelter want this food and have room for it?
- overlap(d, s) hours shared by the pickup window and the shelter's receiving hours.
- dist(d, v)    minutes for the volunteer to reach the restaurant.
- dist(v, s)    minutes for the volunteer to carry the food to the shelter
                (the restaurant-to-shelter leg, at the volunteer's speed: car or transit).

Some things are not trade-offs but hard rules, and make a pair infeasible
(score None) instead of merely lowering it:
- dietary conflict (pork to a halal shelter, nuts to a nut-free one...)
- the shelter isn't receiving at any time during the pickup window
- the volunteer isn't available during the pickup window, or is already busy
- the shelter has room for less than half the donation today

The functions here are pure (plain data in, numbers out) so every rule can
be unit-tested and explained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from hungercue.matching.geo import Point, travel_minutes
from hungercue.timeutil import parse_hhmm

DAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


@dataclass
class DonationInfo:
    id: int
    start: datetime
    end: datetime
    servings: int = 0
    tags: list[str] = field(default_factory=list)  # lowercase: "bread", "contains pork", "vegetarian"...
    location: Point | None = None


@dataclass
class ShelterInfo:
    id: int
    receiving_start: str = "09:00"
    receiving_end: str = "17:00"
    capacity: int = 0  # 0 = not stated (treated as "has room")
    committed: int = 0  # servings already assigned to it for the same day
    needs: list[str] = field(default_factory=list)
    restrictions: list[str] = field(default_factory=list)
    location: Point | None = None


@dataclass
class VolunteerInfo:
    id: int
    availability: list[dict[str, str]] = field(default_factory=list)  # [{"day","start","end"}]
    has_car: bool = False
    location: Point | None = None
    busy: list[tuple[datetime, datetime]] = field(default_factory=list)  # pickups already accepted


@dataclass
class Weights:
    fit: float = 3.0
    overlap: float = 1.0
    dist_dv: float = 0.05
    dist_vs: float = 0.05


@dataclass
class PairScore:
    shelter_id: int
    volunteer_id: int
    score: float
    fit: float
    overlap_h: float
    dist_dv_min: float
    dist_vs_min: float


# Words people use for the same category, so "baked goods" satisfies a need for "bread".
SYNONYMS: dict[str, set[str]] = {
    "bread": {"bread", "bagels", "baked goods", "pastries", "bakery"},
    "produce": {"produce", "fruit", "vegetables", "veggies", "salad"},
    "prepared meals": {"prepared meals", "hot food", "meals", "cooked food", "prepared food", "entrees"},
    "dairy": {"dairy", "milk", "cheese", "yogurt"},
    "protein": {"protein", "meat", "chicken", "fish", "beans", "eggs"},
}


def _canonical(term: str) -> str:
    t = term.lower().strip()
    for canon, words in SYNONYMS.items():
        if t in words:
            return canon
    return t


def dietary_conflict(tags: list[str], restrictions: list[str]) -> bool:
    """True if this food breaks one of the shelter's dietary rules."""
    t = {x.lower() for x in tags}
    has_pork = "contains pork" in t
    has_meat = has_pork or "contains meat" in t
    for r in (x.lower() for x in restrictions):
        if r in ("no pork", "pork free") and has_pork:
            return True
        if r == "halal" and (has_pork or (has_meat and "halal" not in t)):
            return True
        if r == "kosher" and has_meat and "kosher" not in t:
            return True
        if r in ("vegetarian", "vegan") and has_meat:
            return True
        if r in ("nut free", "no nuts") and "contains nuts" in t:
            return True
    return False


def fit(d: DonationInfo, s: ShelterInfo) -> float:
    """0..1: how well the food matches what the shelter needs and can hold."""
    if s.needs:
        wanted = {_canonical(n) for n in s.needs}
        offered = {_canonical(tag) for tag in d.tags}
        need_score = 1.0 if wanted & offered else 0.3  # unrequested food still helps a little
    else:
        need_score = 0.7  # shelter didn't say; neutral-positive
    if s.capacity > 0 and d.servings > 0:
        room = s.capacity - s.committed
        # Squared so a partial fit drops off fast: room for half the food scores
        # 0.25, not 0.5. Food the shelter can't take is food wasted.
        room_score = 1.0 if d.servings <= room else max(0.0, room / d.servings) ** 2
    else:
        room_score = 1.0
    return need_score * room_score


def _window_on(day: datetime, start: str, end: str) -> tuple[datetime, datetime]:
    sh, sm = parse_hhmm(start)
    eh, em = parse_hhmm(end)
    base = day.replace(hour=0, minute=0, second=0, microsecond=0)
    return base.replace(hour=sh, minute=sm), base.replace(hour=eh, minute=em)


def overlap_hours(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> float:
    shared = min(a_end, b_end) - max(a_start, b_start)
    return max(0.0, shared / timedelta(hours=1))


def receiving_overlap(d: DonationInfo, s: ShelterInfo) -> float:
    r_start, r_end = _window_on(d.start, s.receiving_start, s.receiving_end)
    return overlap_hours(d.start, d.end, r_start, r_end)


def volunteer_available(d: DonationInfo, v: VolunteerInfo) -> bool:
    for b_start, b_end in v.busy:
        if overlap_hours(d.start, d.end, b_start, b_end) > 0:
            return False
    if not v.availability:
        return True  # they didn't give hours; offer it and let them say yes or no
    day = DAY_NAMES[d.start.weekday()]
    for slot in v.availability:
        if slot.get("day") != day:
            continue
        s_start, s_end = _window_on(d.start, slot["start"], slot["end"])
        if overlap_hours(d.start, d.end, s_start, s_end) >= 0.25:  # at least 15 minutes to grab it
            return True
    return False


def score_pair(d: DonationInfo, s: ShelterInfo, v: VolunteerInfo, w: Weights) -> PairScore | None:
    """Score one pair, or None if a hard rule makes it impossible."""
    if dietary_conflict(d.tags, s.restrictions):
        return None
    if s.capacity > 0:
        room = s.capacity - s.committed
        # Hard rule: the shelter must have room for at least half the donation.
        if room <= 0 or (d.servings > 0 and room < d.servings / 2):
            return None
    hours = receiving_overlap(d, s)
    if hours <= 0:
        return None
    if not volunteer_available(d, v):
        return None
    f = fit(d, s)
    dv = travel_minutes(v.location, d.location, v.has_car)
    vs = travel_minutes(d.location, s.location, v.has_car)
    total = w.fit * f + w.overlap * hours - w.dist_dv * dv - w.dist_vs * vs
    return PairScore(s.id, v.id, round(total, 4), round(f, 3), round(hours, 2), round(dv, 1), round(vs, 1))
