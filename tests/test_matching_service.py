"""Matcher against the database: capacity holds, busy volunteers, fallbacks, batch rebalance."""

from datetime import datetime

from hungercue.config import Settings
from hungercue.db.repo import Repo
from hungercue.matching.service import Matcher

S = Settings(_env_file=None)


async def seed(repo: Repo) -> tuple[int, int]:
    await repo.upsert_shelter("+1000", name="Near", lat=40.787, lon=-73.975, capacity=50, needs=["bread"],
                              receiving_start="09:00", receiving_end="21:00")
    await repo.upsert_shelter("+1001", name="Far", lat=40.678, lon=-73.944, receiving_start="09:00", receiving_end="21:00")
    v1 = await repo.upsert_volunteer("+2000", name="Ana", lat=40.81, lon=-73.95, has_car=True)
    v2 = await repo.upsert_volunteer("+2001", name="Ben", lat=40.70, lon=-73.99, has_car=True)
    return v1.id, v2.id


async def new_pickup(repo: Repo, servings: int = 30, hour: int = 15) -> int:
    r = await repo.upsert_restaurant("+3000", name="Bagels", lat=40.80, lon=-73.96)
    p = await repo.create_pickup(r.id, items="bagels", servings=servings, food_tags=["bagels"],
                                 pickup_start=datetime(2026, 11, 3, hour), pickup_end=datetime(2026, 11, 3, hour + 2),
                                 lat=40.80, lon=-73.96)
    return p.id


async def test_match_picks_best_and_saves(repo: Repo) -> None:
    ana, _ = await seed(repo)
    pid = await new_pickup(repo)
    best = await Matcher(repo, S).match(pid)
    p = await repo.get_pickup(pid)
    assert best.volunteer_id == ana and p.volunteer_id == ana and p.shelter.name == "Near" and p.match_score == best.score


async def test_fallback_skips_offered_volunteers_and_busy_ones(repo: Repo) -> None:
    ana, ben = await seed(repo)
    first = await new_pickup(repo)
    m = Matcher(repo, S)
    await m.match(first)
    await repo.update_pickup(first, status="accepted")
    # A second pickup at the same time: Ana is busy now, so Ben gets it.
    second = await new_pickup(repo)
    assert (await m.match(second)).volunteer_id == ben
    # Ben doesn't answer: excluded, nobody else free -> unmatched.
    await repo.update_pickup(second, offered_volunteer_ids=[ben])
    assert await m.match(second) is None
    assert (await repo.get_pickup(second)).status == "unmatched"
    assert await m.match(999) is None


async def test_capacity_is_held_by_active_pickups(repo: Repo) -> None:
    await seed(repo)
    m = Matcher(repo, S)
    a = await new_pickup(repo, servings=40)
    await m.match(a)
    await repo.update_pickup(a, status="accepted")
    b = await new_pickup(repo, servings=40, hour=18)
    assert (await repo.get_pickup(b)).id and (await m.match(b)) is not None
    assert (await repo.get_pickup(b)).shelter.name == "Far"  # "Near" has only 10 servings of room left


async def test_rebalance_day_uses_hungarian_and_keeps_accepted(repo: Repo) -> None:
    ana, ben = await seed(repo)
    m = Matcher(repo, S)
    locked = await new_pickup(repo, hour=9)
    await repo.update_pickup(locked, status="accepted", volunteer_id=ana)
    p1, p2 = await new_pickup(repo, hour=15), await new_pickup(repo, hour=15)
    result, needs_offer = await m.rebalance_day(datetime(2026, 11, 3))
    assert set(result.assignments) == {p1, p2} and set(needs_offer) == {p1, p2}
    vols = {(await repo.get_pickup(p)).volunteer_id for p in (p1, p2)}
    assert vols == {ana, ben}  # two simultaneous pickups need two different people
    assert (await repo.get_pickup(locked)).status == "accepted"


async def test_rebalance_does_not_reoffer_to_same_volunteer(repo: Repo) -> None:
    ana, _ = await seed(repo)
    m = Matcher(repo, S)
    pid = await new_pickup(repo)
    best = await m.match(pid)
    await repo.update_pickup(pid, status="offered", offered_at=datetime(2026, 11, 3, 12),
                             offered_volunteer_ids=[best.volunteer_id])
    _, needs_offer = await m.rebalance_day(datetime(2026, 11, 3))
    p = await repo.get_pickup(pid)
    assert needs_offer == [] and p.status == "offered" and p.offered_at == datetime(2026, 11, 3, 12)
    assert p.offered_volunteer_ids == [best.volunteer_id]
