"""Scoring one call against its ground truth.

Slot accuracy = share of fields captured exactly right. Each field has a
precise, deterministic rule (no LLM judge, so scores are reproducible and
explainable):

restaurant: items (every keyword group present), quantity (the number),
            pickup date, start time, end time, address (house number AND
            street name), business name (new callers only)
shelter:    capacity, receiving start, receiving end, needs, dietary rules,
            name, address
volunteer:  available days, start time, end time, neighborhoods, has car

Task completion = the call produced a valid record of the right kind
(for restaurants: a pickup with a date, time window and address).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from hungercue.db.models import Pickup, Restaurant, Shelter, Volunteer
from hungercue.matching.geo import neighborhood_point
from hungercue.matching.scoring import _canonical

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "diez": 10}
STREET_FILLER = {"west", "east", "north", "south", "w", "e", "street", "st", "avenue", "ave", "boulevard", "blvd"}


@dataclass
class FieldResult:
    field: str
    ok: bool
    expected: Any
    got: Any


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9áéíóúñü ]", " ", (text or "").lower())


def numbers_in(text: str) -> set[int]:
    words = _norm(text).split()
    found = {int(w) for w in words if w.isdigit()}
    found |= {NUMBER_WORDS[w] for w in words if w in NUMBER_WORDS}
    return found


def street_key(street: str) -> str:
    """The distinctive part of a street name: 'West 112th Street' -> '112'."""
    for word in _norm(street).split():
        if word not in STREET_FILLER:
            return re.sub(r"(st|nd|rd|th)$", "", word) if word[0].isdigit() else word
    return _norm(street).strip()


def address_ok(captured: str, number: str, street: str) -> bool:
    words = _norm(captured).split()
    key = street_key(street)
    has_number = number in words
    has_street = any(w == key or re.sub(r"(st|nd|rd|th)$", "", w) == key for w in words)
    return has_number and has_street


def keywords_ok(captured: str, groups: list[list[str]]) -> bool:
    text = _norm(captured)
    return all(any(alt in text for alt in group) for group in groups)


def name_ok(captured: str, expected: str) -> bool:
    """At least one significant word of the name was captured ("Sunrise Bagels" ~ "sunrise")."""
    want = {w for w in _norm(expected).split() if len(w) > 2 and w not in {"the", "st"}}
    return bool(want & set(_norm(captured).split()))


def resolve_day(day: str, call_date: datetime) -> str:
    return (call_date + timedelta(days=1 if day == "tomorrow" else 0)).strftime("%Y-%m-%d")


def score_restaurant(truth: dict[str, Any], pickup: Pickup | None, restaurant: Restaurant | None,
                     call_date: datetime, returning: bool) -> list[FieldResult]:
    p = pickup
    day = resolve_day(truth["pickup_day"], call_date)
    fields = [
        FieldResult("items", bool(p) and keywords_ok(p.items, truth["item_keywords"]), truth["item_keywords"], p and p.items),
        FieldResult("quantity", bool(p) and truth["quantity_number"] in numbers_in(p.quantity), truth["quantity_number"], p and p.quantity),
        FieldResult("pickup_date", bool(p) and f"{p.pickup_start:%Y-%m-%d}" == day, day, p and f"{p.pickup_start:%Y-%m-%d}"),
        FieldResult("pickup_start", bool(p) and f"{p.pickup_start:%H:%M}" == truth["pickup_start"], truth["pickup_start"], p and f"{p.pickup_start:%H:%M}"),
        FieldResult("pickup_end", bool(p) and f"{p.pickup_end:%H:%M}" == truth["pickup_end"], truth["pickup_end"], p and f"{p.pickup_end:%H:%M}"),
        FieldResult("address", bool(p) and address_ok(p.address, truth["address_number"], truth["address_street"]),
                    f"{truth['address_number']} {truth['address_street']}", p and p.address),
    ]
    if not returning:  # a returning caller's name was already on file, so it isn't a test of this call
        got = restaurant.name if restaurant else ""
        fields.append(FieldResult("business_name", name_ok(got, truth["business_name"]), truth["business_name"], got))
    return fields


def score_shelter(truth: dict[str, Any], s: Shelter | None) -> list[FieldResult]:
    want_needs = {_canonical(n) for n in truth["needs"]}
    got_needs = {_canonical(n) for n in (s.needs if s else [])}
    want_diet = {_norm(d).strip() for d in truth["dietary_restrictions"]}
    got_diet = {_norm(d).strip() for d in (s.dietary_restrictions if s else [])}
    return [
        FieldResult("capacity", bool(s) and s.capacity == truth["capacity"], truth["capacity"], s and s.capacity),
        FieldResult("receiving_start", bool(s) and s.receiving_start == truth["receiving_start"], truth["receiving_start"], s and s.receiving_start),
        FieldResult("receiving_end", bool(s) and s.receiving_end == truth["receiving_end"], truth["receiving_end"], s and s.receiving_end),
        FieldResult("needs", bool(s) and want_needs <= got_needs, sorted(want_needs), sorted(got_needs)),
        # Exact set: a missed restriction is unsafe, and an invented one wrongly blocks donations.
        FieldResult("dietary_restrictions", bool(s) and want_diet == got_diet, sorted(want_diet), sorted(got_diet)),
        FieldResult("name", bool(s) and name_ok(s.name, truth["shelter_name"]), truth["shelter_name"], s and s.name),
        FieldResult("address", bool(s) and address_ok(s.address, truth["address_number"], truth["address_street"]),
                    f"{truth['address_number']} {truth['address_street']}", s and s.address),
    ]


def _same_hood(a: str, b: str) -> bool:
    pa, pb = neighborhood_point(a), neighborhood_point(b)
    return (pa is not None and pa == pb) or _norm(a).strip() == _norm(b).strip()


def score_volunteer(truth: dict[str, Any], v: Volunteer | None) -> list[FieldResult]:
    slots = v.availability if v else []
    days = sorted({s["day"] for s in slots}, key=["mon", "tue", "wed", "thu", "fri", "sat", "sun"].index)
    starts = {s["start"] for s in slots}
    ends = {s["end"] for s in slots}
    hoods = v.neighborhoods if v else []
    return [
        FieldResult("availability_days", days == truth["availability_days"], truth["availability_days"], days),
        FieldResult("availability_start", bool(slots) and starts == {truth["availability_start"]}, truth["availability_start"], sorted(starts)),
        FieldResult("availability_end", bool(slots) and ends == {truth["availability_end"]}, truth["availability_end"], sorted(ends)),
        FieldResult("neighborhoods", bool(v) and all(any(_same_hood(t, h) for h in hoods) for t in truth["neighborhoods"]),
                    truth["neighborhoods"], hoods),
        FieldResult("has_car", bool(v) and v.has_car == truth["has_car"], truth["has_car"], v and v.has_car),
    ]


def completed(caller_type: str, pickup: Pickup | None, shelter: Shelter | None, volunteer: Volunteer | None) -> bool:
    if caller_type == "restaurant":
        return pickup is not None and bool(pickup.address) and pickup.pickup_end > pickup.pickup_start
    if caller_type == "shelter":
        return shelter is not None
    return volunteer is not None and bool(volunteer.availability)
