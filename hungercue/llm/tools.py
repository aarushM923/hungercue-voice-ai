"""Tools the LLM can call, and the code that runs them.

"Tool calling" means the model, instead of only writing text, can output a
structured request like log_donation(items="bagels", quantity="3 bags", ...).
We run the matching Python function and send the result back, and the model
then tells the caller what happened. Each schema below is what the model sees.

We use OpenAI "strict" schemas: the model's arguments are guaranteed to match
the JSON schema exactly. Strict mode requires every property to be listed as
required, so optional fields are typed as e.g. ["string", "null"].
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from hungercue.db.repo import Repo
from hungercue.phone import normalize_phone
from hungercue.state.session_store import CallState
from hungercue.timeutil import parse_hhmm

log = logging.getLogger(__name__)

Geocoder = Callable[[str], Awaitable[tuple[float, float] | None]]


def _fn(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    """Wrap a property dict into the OpenAI tool schema format (strict mode)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


S = {"type": "string"}
S_NULL = {"type": ["string", "null"]}
STR_LIST = {"type": "array", "items": {"type": "string"}}
HHMM = {"type": "string", "description": "24-hour local time HH:MM"}
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

TOOL_SCHEMAS: list[dict[str, Any]] = [
    _fn("set_caller_type", "Record what kind of caller this is as soon as it's clear.",
        {"caller_type": {"type": "string", "enum": ["restaurant", "shelter", "volunteer"]}}),
    _fn("log_donation", "Book a food pickup from a restaurant. Only call after reading details back and the caller confirmed.", {
        "business_name": S,
        "contact_name": S_NULL,
        "items": {"type": "string", "description": "What food, e.g. 'bagels and cream cheese'"},
        "quantity": {"type": "string", "description": "Amount as the caller said it, e.g. '3 large trays'"},
        "servings_estimate": {"type": "integer", "description": "Rough number of servings, best guess"},
        "food_tags": {**STR_LIST, "description": "Categories and dietary flags, e.g. 'bread', 'produce', 'prepared meals', 'contains pork', 'vegetarian', 'halal', 'contains nuts'"},
        "pickup_date": {"type": "string", "description": "YYYY-MM-DD"},
        "pickup_start": HHMM,
        "pickup_end": HHMM,
        "address": {"type": "string", "description": "Pickup street address"},
        "notes": S_NULL,
        "confirmed_by_caller": {"type": "boolean", "description": "True only if the caller said yes to the read-back"},
    }),
    _fn("update_shelter", "Create or update a shelter's intake details.", {
        "name": S_NULL,
        "contact_name": S_NULL,
        "address": S_NULL,
        "capacity": {"type": ["integer", "null"], "description": "Servings they can accept today"},
        "needs": {**STR_LIST, "description": "Food they need, e.g. 'produce', 'prepared meals'"},
        "dietary_restrictions": {**STR_LIST, "description": "e.g. 'halal', 'no pork', 'nut free', 'vegetarian'"},
        "receiving_start": {"type": ["string", "null"], "description": "HH:MM"},
        "receiving_end": {"type": ["string", "null"], "description": "HH:MM"},
    }),
    _fn("update_volunteer", "Create or update a volunteer's availability.", {
        "name": S_NULL,
        "availability": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"day": {"type": "string", "enum": DAYS}, "start": HHMM, "end": HHMM},
                "required": ["day", "start", "end"],
                "additionalProperties": False,
            },
        },
        "neighborhoods": STR_LIST,
        "has_car": {"type": ["boolean", "null"]},
    }),
    _fn("transfer_to_human", "Transfer the call to a human coordinator.", {"reason": S}),
    _fn("take_voicemail", "Let the caller leave a voicemail for the coordinators.", {"reason": S}),
    _fn("end_call", "Hang up after saying goodbye.", {"reason": S}),
]


@dataclass
class ToolContext:
    """Everything a tool needs, injected so tests can supply fakes."""

    repo: Repo
    state: CallState
    geocode: Geocoder | None = None
    # Hooks run after a successful write (matching + SMS, Sheets logging, etc.).
    on_donation: Callable[[int], Awaitable[None]] | None = None
    on_directory_update: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None
    # Set by transfer/voicemail/end tools; the call orchestrator acts on it after speaking.
    pending_action: str = ""
    calls: list[str] = field(default_factory=list)  # tool names called, for eval metrics


def _clean(args: dict[str, Any]) -> dict[str, Any]:
    """Turn empty strings into None so the repo's upsert leaves those fields alone."""
    return {k: (None if v == "" else v) for k, v in args.items()}


async def _geo(ctx: ToolContext, text: str | None) -> tuple[float | None, float | None]:
    if not text or ctx.geocode is None:
        return None, None
    try:
        point = await ctx.geocode(text)
    except Exception as e:  # geocoding is a nice-to-have; never fail a booking over it
        log.warning("geocode failed for %r: %s", text, e)
        return None, None
    return point if point else (None, None)


def _valid_hhmm(value: str | None) -> bool:
    if not value:
        return False
    try:
        parse_hhmm(value)
        return True
    except ValueError:
        return False


async def set_caller_type(ctx: ToolContext, caller_type: str) -> dict[str, Any]:
    ctx.state.caller_type = caller_type
    await ctx.repo.update_call(ctx.state.call_sid, caller_type=caller_type)
    return {"ok": True}


async def log_donation(ctx: ToolContext, **a: Any) -> dict[str, Any]:
    if not a.get("confirmed_by_caller"):
        return {"ok": False, "error": "Read the details back to the caller and get a clear yes first."}
    try:
        day = datetime.strptime(a["pickup_date"], "%Y-%m-%d")
        sh, sm = parse_hhmm(a["pickup_start"])
        eh, em = parse_hhmm(a["pickup_end"])
    except (ValueError, KeyError):
        return {"ok": False, "error": "Pickup date or time unclear. Ask for the date and a start and end time."}
    start, end = day.replace(hour=sh, minute=sm), day.replace(hour=eh, minute=em)
    if end <= start:
        return {"ok": False, "error": "Pickup end time must be after the start time. Ask the caller to clarify."}
    if not a.get("address"):
        return {"ok": False, "error": "Need the pickup address."}

    phone = normalize_phone(ctx.state.from_number)
    lat, lon = await _geo(ctx, a["address"])
    restaurant = await ctx.repo.upsert_restaurant(
        phone or f"unknown:{ctx.state.call_sid}",
        name=a.get("business_name") or None,
        contact_name=a.get("contact_name") or None,
        address=a["address"],
        lat=lat,
        lon=lon,
        language=ctx.state.language,
    )
    pickup = await ctx.repo.create_pickup(
        restaurant.id,
        items=a.get("items", ""),
        quantity=a.get("quantity", ""),
        servings=max(0, int(a.get("servings_estimate") or 0)),
        food_tags=[t.lower() for t in a.get("food_tags") or []],
        pickup_start=start,
        pickup_end=end,
        address=a["address"],
        lat=lat,
        lon=lon,
        notes=a.get("notes") or "",
        call_sid=ctx.state.call_sid,
    )
    ctx.state.pickup_id = pickup.id
    ctx.state.outcome = "pickup_booked"
    await ctx.repo.update_call(ctx.state.call_sid, outcome="pickup_booked")
    if ctx.on_donation:
        await ctx.on_donation(pickup.id)
    return {"ok": True, "pickup_id": pickup.id, "next": "Tell the caller they'll get a text with the volunteer's name."}


async def update_shelter(ctx: ToolContext, **a: Any) -> dict[str, Any]:
    a = _clean(a)
    for key in ("receiving_start", "receiving_end"):
        if a.get(key) is not None and not _valid_hhmm(a[key]):
            return {"ok": False, "error": f"{key} must be HH:MM."}
    lat, lon = await _geo(ctx, a.get("address"))
    fields = {**a, "needs": [n.lower() for n in a.get("needs") or []] or None,
              "dietary_restrictions": [d.lower() for d in a.get("dietary_restrictions") or []] or None,
              "lat": lat, "lon": lon, "language": ctx.state.language}
    shelter = await ctx.repo.upsert_shelter(normalize_phone(ctx.state.from_number) or f"unknown:{ctx.state.call_sid}", **fields)
    ctx.state.outcome = "shelter_updated"
    await ctx.repo.update_call(ctx.state.call_sid, outcome="shelter_updated")
    if ctx.on_directory_update:
        await ctx.on_directory_update("shelter", {**a, "id": shelter.id})
    return {"ok": True}


async def update_volunteer(ctx: ToolContext, **a: Any) -> dict[str, Any]:
    a = _clean(a)
    slots = [s for s in a.get("availability") or [] if _valid_hhmm(s.get("start")) and _valid_hhmm(s.get("end"))]
    hoods = a.get("neighborhoods") or []
    lat, lon = await _geo(ctx, f"{hoods[0]}, New York, NY" if hoods else None)
    vol = await ctx.repo.upsert_volunteer(
        normalize_phone(ctx.state.from_number) or f"unknown:{ctx.state.call_sid}",
        name=a.get("name"),
        availability=slots or None,
        neighborhoods=hoods or None,
        has_car=a.get("has_car"),
        lat=lat,
        lon=lon,
        language=ctx.state.language,
    )
    ctx.state.outcome = "volunteer_updated"
    await ctx.repo.update_call(ctx.state.call_sid, outcome="volunteer_updated")
    if ctx.on_directory_update:
        await ctx.on_directory_update("volunteer", {**a, "id": vol.id})
    return {"ok": True}


async def _set_action(ctx: ToolContext, action: str, reason: str) -> dict[str, Any]:
    ctx.pending_action = action
    if action != "end_call":
        ctx.state.outcome = action
        await ctx.repo.update_call(ctx.state.call_sid, outcome=action)
    log.info("%s requested: %s", action, reason)
    return {"ok": True, "next": "Say one short sentence; the system handles the rest."}


HANDLERS: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
    "set_caller_type": set_caller_type,
    "log_donation": log_donation,
    "update_shelter": update_shelter,
    "update_volunteer": update_volunteer,
    "transfer_to_human": lambda ctx, reason: _set_action(ctx, "transferred", reason),
    "take_voicemail": lambda ctx, reason: _set_action(ctx, "voicemail", reason),
    "end_call": lambda ctx, reason: _set_action(ctx, "end_call", reason),
}


async def run_tool(ctx: ToolContext, name: str, arguments_json: str) -> str:
    """Execute one tool call from the model; always returns a JSON string for the model."""
    ctx.calls.append(name)
    handler = HANDLERS.get(name)
    if handler is None:
        return json.dumps({"ok": False, "error": f"unknown tool {name}"})
    try:
        args = json.loads(arguments_json or "{}")
        result = await handler(ctx, **args)
    except Exception as e:  # report the failure to the model so it can recover gracefully
        log.exception("tool %s failed", name)
        result = {"ok": False, "error": f"internal error: {type(e).__name__}. Offer to transfer to a coordinator."}
    return json.dumps(result)
