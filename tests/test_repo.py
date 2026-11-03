"""Tests for the repository layer against a real (temporary) SQLite database."""

from datetime import datetime

from hungercue.db.repo import Repo


async def test_unknown_caller(repo: Repo) -> None:
    assert (await repo.lookup_caller("+12125550100")).kind == ""
    assert (await repo.lookup_caller("")).kind == ""


async def test_returning_restaurant_with_last_pickup(repo: Repo) -> None:
    r = await repo.upsert_restaurant("+12125550100", name="Bagel Shop", address="200 W 112th St")
    await repo.create_pickup(
        r.id, items="bagels", quantity="3 bags",
        pickup_start=datetime(2026, 11, 3, 15), pickup_end=datetime(2026, 11, 3, 17),
    )
    prof = await repo.lookup_caller("+12125550100")
    assert prof.kind == "restaurant" and prof.name == "Bagel Shop"
    assert "112th" in prof.summary and "3 bags bagels" in prof.summary


async def test_upsert_keeps_fields_not_given(repo: Repo) -> None:
    await repo.upsert_shelter("+12125550111", name="Hope House", address="1 Main St", capacity=50)
    s = await repo.upsert_shelter("+12125550111", capacity=80, address=None)
    assert s.capacity == 80 and s.address == "1 Main St"
    prof = await repo.lookup_caller("+12125550111")
    assert prof.kind == "shelter" and "Capacity 80" in prof.summary


async def test_volunteer_lookup_and_lists(repo: Repo) -> None:
    await repo.upsert_volunteer("+12125550122", name="Ana", neighborhoods=["Harlem"], has_car=True)
    await repo.upsert_volunteer("+12125550123", name="Old", active=False)
    assert (await repo.lookup_caller("+12125550122")).kind == "volunteer"
    assert [v.name for v in await repo.active_volunteers()] == ["Ana"]
    assert (await repo.volunteer_by_phone("+12125550122")).name == "Ana"


async def test_pickup_queries(repo: Repo) -> None:
    r = await repo.upsert_restaurant("+12125550100", name="Deli")
    v = await repo.upsert_volunteer("+12125550122", name="Ana")
    p = await repo.create_pickup(r.id, items="soup", pickup_start=datetime(2026, 11, 3, 15), pickup_end=datetime(2026, 11, 3, 16))
    await repo.update_pickup(p.id, status="offered", volunteer_id=v.id, offered_at=datetime(2026, 11, 3, 12))
    loaded = await repo.get_pickup(p.id)
    assert loaded.restaurant.name == "Deli" and loaded.volunteer.name == "Ana"
    assert len(await repo.pickups_between(datetime(2026, 11, 3), datetime(2026, 11, 4))) == 1
    assert len(await repo.pickups_with_status("offered")) == 1
    assert (await repo.offered_pickup_for_volunteer(v.id)).id == p.id
    assert await repo.update_pickup(999, status="x") is None


async def test_call_transcript_and_turns(repo: Repo) -> None:
    await repo.start_call("CA1", "+12125550100")
    await repo.start_call("CA1", "+12125550100")  # idempotent: Twilio can reconnect
    await repo.add_transcript("CA1", "caller", "hello")
    await repo.add_transcript("CA1", "agent", "hi there")
    await repo.update_call("CA1", outcome="pickup_booked")
    c = await repo.get_call("CA1")
    assert [t["text"] for t in c.transcript] == ["hello", "hi there"] and c.outcome == "pickup_booked"
    await repo.add_turn("CA1", 0, v2v_ms=700.0)
    assert (await repo.turns_since(datetime(2000, 1, 1)))[0].v2v_ms == 700.0
    assert len(await repo.recent_calls()) == 1


async def test_sms_log(repo: Repo) -> None:
    r = await repo.upsert_restaurant("+12125550100")
    p = await repo.create_pickup(r.id, pickup_start=datetime(2026, 1, 1), pickup_end=datetime(2026, 1, 1))
    await repo.log_sms("outbound", "+12125550100", "Confirmed", kind="confirmation", pickup_id=p.id)
    assert [m.kind for m in await repo.sms_for_pickup(p.id)] == ["confirmation"]
