"""Tool handler tests against a temporary database."""

import json

from hungercue.db.repo import Repo
from hungercue.llm.tools import TOOL_SCHEMAS, ToolContext, run_tool
from hungercue.state.session_store import CallState

BASE = {
    "business_name": "Deli", "contact_name": "Sam", "items": "soup", "quantity": "2 pots", "servings_estimate": 30,
    "food_tags": [], "pickup_date": "2026-11-04", "pickup_start": "15:00", "pickup_end": "16:00",
    "address": "1 Main St", "notes": "", "confirmed_by_caller": True,
}


def ctx_for(repo: Repo, phone: str = "+12125550100", **kw) -> ToolContext:
    return ToolContext(repo=repo, state=CallState(call_sid="CA1", from_number=phone), **kw)


async def call(ctx: ToolContext, name: str, args: dict) -> dict:
    return json.loads(await run_tool(ctx, name, json.dumps(args)))


def test_schemas_are_strict_and_complete() -> None:
    for t in TOOL_SCHEMAS:
        params = t["function"]["parameters"]
        assert t["function"]["strict"] and params["additionalProperties"] is False
        assert set(params["required"]) == set(params["properties"])


async def test_donation_requires_confirmation_and_valid_times(repo: Repo) -> None:
    ctx = ctx_for(repo)
    assert "Read the details back" in (await call(ctx, "log_donation", {**BASE, "confirmed_by_caller": False}))["error"]
    assert "after the start" in (await call(ctx, "log_donation", {**BASE, "pickup_end": "14:00"}))["error"]
    assert "unclear" in (await call(ctx, "log_donation", {**BASE, "pickup_date": "tomorrow"}))["error"]
    assert "address" in (await call(ctx, "log_donation", {**BASE, "address": ""}))["error"]
    assert ctx.state.pickup_id is None


async def test_donation_geocodes_and_hooks(repo: Repo) -> None:
    async def geocode(text: str):
        return (40.8, -73.96)

    ctx = ctx_for(repo, geocode=geocode)
    out = await call(ctx, "log_donation", BASE)
    p = await repo.get_pickup(out["pickup_id"])
    assert (p.lat, p.lon) == (40.8, -73.96) and p.restaurant.contact_name == "Sam"


async def test_geocode_failure_does_not_block_booking(repo: Repo) -> None:
    async def broken(text: str):
        raise RuntimeError("down")

    out = await call(ctx_for(repo, geocode=broken), "log_donation", BASE)
    assert out["ok"]


async def test_update_shelter_and_hook(repo: Repo) -> None:
    seen = []

    async def hook(kind, data):
        seen.append((kind, data["capacity"]))

    ctx = ctx_for(repo, "+12125550111", on_directory_update=hook)
    args = {"name": "Hope House", "contact_name": None, "address": "", "capacity": 60, "needs": ["Produce"],
            "dietary_restrictions": ["Halal"], "receiving_start": "10:00", "receiving_end": "18:00"}
    assert (await call(ctx, "update_shelter", args))["ok"]
    s = (await repo.active_shelters())[0]
    assert s.needs == ["produce"] and s.dietary_restrictions == ["halal"] and s.receiving_end == "18:00"
    assert seen == [("shelter", 60)] and ctx.state.outcome == "shelter_updated"
    assert not (await call(ctx, "update_shelter", {**args, "receiving_start": "10am"}))["ok"]


async def test_update_volunteer_drops_bad_slots(repo: Repo) -> None:
    ctx = ctx_for(repo, "+12125550122")
    args = {"name": "Ana", "availability": [{"day": "mon", "start": "09:00", "end": "12:00"},
                                            {"day": "tue", "start": "nine", "end": "12:00"}],
            "neighborhoods": ["Harlem"], "has_car": True}
    assert (await call(ctx, "update_volunteer", args))["ok"]
    v = (await repo.active_volunteers())[0]
    assert v.availability == [{"day": "mon", "start": "09:00", "end": "12:00"}] and v.has_car


async def test_actions_and_errors(repo: Repo) -> None:
    ctx = ctx_for(repo)
    await call(ctx, "set_caller_type", {"caller_type": "restaurant"})
    assert ctx.state.caller_type == "restaurant"
    await call(ctx, "transfer_to_human", {"reason": "asked"})
    assert ctx.pending_action == "transferred" and ctx.state.outcome == "transferred"
    await call(ctx, "end_call", {"reason": "done"})
    assert ctx.pending_action == "end_call" and ctx.state.outcome == "transferred"
    assert "unknown tool" in (await call(ctx, "nope", {}))["error"]
    assert "internal error" in (await call(ctx, "set_caller_type", {"wrong_arg": 1}))["error"]
    assert ctx.calls == ["set_caller_type", "transfer_to_human", "end_call", "nope", "set_caller_type"]
