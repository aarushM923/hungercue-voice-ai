"""SMS follow-up flow: offers, YES/NO replies, 15-minute fallback, reminders."""

from datetime import datetime, timedelta

from hungercue.config import Settings
from hungercue.db.repo import Repo
from hungercue.matching.service import Matcher
from hungercue.notify.followup import FollowUp, confirmation_text, offer_text, window_text
from hungercue.notify.sms import Texter

START = datetime(2026, 11, 3, 15)
OFFERED_AT = datetime(2026, 11, 3, 12)


class FakeGateway:
    def __init__(self, fail: bool = False) -> None:
        self.sent: list[tuple[str, str]] = []
        self.fail = fail

    async def send(self, to: str, body: str) -> None:
        if self.fail:
            raise RuntimeError("twilio down")
        self.sent.append((to, body))


async def setup(repo: Repo, enabled: bool = True, volunteers: int = 2) -> tuple[FollowUp, FakeGateway, int]:
    await repo.upsert_shelter("+15550000001", name="Hope House", address="1 Shelter Rd", lat=40.787, lon=-73.975,
                              receiving_start="09:00", receiving_end="21:00")
    for i, (lat, lon) in enumerate([(40.81, -73.95), (40.70, -73.99)][:volunteers]):
        await repo.upsert_volunteer(f"+1555000010{i}", name=["Ana Ruiz", "Ben Cole"][i], lat=lat, lon=lon, has_car=True)
    r = await repo.upsert_restaurant("+15550000200", name="Bagel Shop", lat=40.80, lon=-73.96)
    p = await repo.create_pickup(r.id, items="bagels", quantity="3 bags", servings=30, address="200 W 112th St",
                                 pickup_start=START, pickup_end=START + timedelta(hours=2), lat=40.80, lon=-73.96)
    gw = FakeGateway()
    s = Settings(_env_file=None)
    return FollowUp(repo, Matcher(repo, s), Texter(repo, gw, enabled=enabled), s), gw, p.id


async def test_texter_statuses(repo: Repo) -> None:
    assert await Texter(repo, FakeGateway(), enabled=False).send("+1555", "hi", "x") == "pending_verification"
    assert await Texter(repo, None, enabled=True).send("+1555", "hi", "x") == "pending_verification"
    assert await Texter(repo, FakeGateway(fail=True), enabled=True).send("+1555", "hi", "x") == "failed"
    assert await Texter(repo, FakeGateway(), enabled=True).send("unknown:CA1", "hi", "x") == "failed"
    assert await Texter(repo, FakeGateway(), enabled=True).send("+1555", "hi", "x") == "sent"
    assert [m.status for m in reversed(await repo.recent_sms())] == [
        "pending_verification", "pending_verification", "failed", "failed", "sent"]


async def test_offer_then_yes_confirms_restaurant(repo: Repo) -> None:
    f, gw, pid = await setup(repo)
    assert await f.offer_next(pid, now=OFFERED_AT)
    to, body = gw.sent[0]
    assert to == "+15550000100" and "Reply YES" in body and "Hope House" in body and "3 bags bagels" in body
    p = await repo.get_pickup(pid)
    assert p.status == "offered" and p.offered_volunteer_ids == [p.volunteer_id]

    reply = await f.handle_reply("(555) 000-0100", "Yes!")  # formatting differences are normalised
    assert "confirmed" in reply
    assert (await repo.get_pickup(pid)).status == "accepted"
    to, body = gw.sent[-1]
    assert to == "+15550000200" and "Ana will pick it up" in body and "Tue Nov 3, 3:00 PM-5:00 PM" in body


async def test_no_reply_offers_next_volunteer(repo: Repo) -> None:
    f, gw, pid = await setup(repo)
    await f.offer_next(pid, now=OFFERED_AT)
    first = (await repo.get_pickup(pid)).volunteer_id
    assert "someone else" in await f.handle_reply("+15550000100", "no sorry")
    p = await repo.get_pickup(pid)
    assert p.volunteer_id != first and p.status == "offered" and gw.sent[-1][0] == "+15550000101"


async def test_timeout_after_15_minutes_then_unmatched(repo: Repo) -> None:
    f, gw, pid = await setup(repo)
    await f.offer_next(pid, now=OFFERED_AT)
    assert await f.sweep(now=OFFERED_AT + timedelta(minutes=14)) == {"reoffered": 0, "reminded": 0}
    assert (await f.sweep(now=OFFERED_AT + timedelta(minutes=15)))["reoffered"] == 1
    assert gw.sent[-1][0] == "+15550000101"  # Ben now
    later = OFFERED_AT + timedelta(minutes=31)
    assert (await f.sweep(now=later))["reoffered"] == 1  # Ben silent too: nobody left
    p = await repo.get_pickup(pid)
    assert p.status == "unmatched" and "coordinator will follow up" in gw.sent[-1][1]
    assert gw.sent[-1][0] == "+15550000200"
    assert (await f.sweep(now=later + timedelta(minutes=20)))["reoffered"] == 0  # stays put


async def test_reminder_30_minutes_before_once(repo: Repo) -> None:
    f, gw, pid = await setup(repo)
    await f.offer_next(pid, now=OFFERED_AT)
    await f.accept(pid)
    assert (await f.sweep(now=START - timedelta(minutes=31)))["reminded"] == 0
    assert (await f.sweep(now=START - timedelta(minutes=30)))["reminded"] == 1
    assert "Reminder" in gw.sent[-1][1] and gw.sent[-1][0] == "+15550000100"
    assert (await f.sweep(now=START - timedelta(minutes=10)))["reminded"] == 0  # not twice


async def test_replies_from_unknown_numbers_and_no_offers(repo: Repo) -> None:
    f, gw, pid = await setup(repo)
    assert await f.handle_reply("+19998887777", "YES") == ""
    assert "no open pickup" in await f.handle_reply("+15550000100", "YES")
    await f.offer_next(pid, now=OFFERED_AT)
    assert "reply YES" in await f.handle_reply("+15550000100", "what time?")
    inbound = [m for m in await repo.recent_sms() if m.direction == "inbound"]
    assert len(inbound) == 3 and inbound[0].status == "received"


async def test_background_task_runs(repo: Repo) -> None:
    import asyncio
    f, gw, pid = await setup(repo, enabled=False)
    f.donation_logged_background(pid)
    while f._tasks:
        await asyncio.sleep(0.01)
    assert (await repo.get_pickup(pid)).status == "offered"
    assert gw.sent == []  # SMS disabled: logged as pending, nothing actually sent
    assert {m.status for m in await repo.recent_sms()} == {"pending_verification"}


async def test_message_texts(repo: Repo) -> None:
    f, gw, pid = await setup(repo)
    await f.offer_next(pid, now=OFFERED_AT)
    p = await repo.get_pickup(pid)
    assert window_text(p) == "Tue Nov 3, 3:00 PM-5:00 PM"
    assert len(offer_text(p)) < 320 and len(confirmation_text(p)) < 320  # at most 2 SMS segments
