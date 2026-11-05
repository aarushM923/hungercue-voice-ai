"""Outbound follow-up after a donation is booked.

    call ends with log_donation
        -> match best shelter + volunteer
        -> text the volunteer: "Reply YES to accept"
             YES          -> pickup accepted; text the restaurant the time and volunteer's name
             NO / 15 min  -> offer it to the next-best volunteer
             nobody left  -> mark unmatched and tell the restaurant a coordinator will follow up
        -> 30 minutes before pickup: reminder text to the volunteer

Timing lives in the database (offered_at, pickup_start, reminder_sent) and a
"sweeper" loop checks it every 30 seconds. A restart therefore loses nothing:
the next sweep simply finds whatever is overdue.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

from hungercue.config import Settings
from hungercue.db.models import Pickup
from hungercue.db.repo import Repo
from hungercue.matching.service import Matcher
from hungercue.notify.sms import Texter
from hungercue.phone import normalize_phone
from hungercue.timeutil import now_local

log = logging.getLogger(__name__)

YES_WORDS = {"yes", "y", "yes!", "si", "sí", "ok", "okay", "accept", "yep", "sure"}
NO_WORDS = {"no", "n", "pass", "decline", "cant", "can't", "nope"}


def _time(dt: datetime) -> str:
    """6:00 PM style; %-I (no leading zero) isn't portable, so strip it by hand."""
    return dt.strftime("%I:%M %p").lstrip("0")


def window_text(p: Pickup) -> str:
    return f"{p.pickup_start:%a %b} {p.pickup_start.day}, {_time(p.pickup_start)}-{_time(p.pickup_end)}"


def offer_text(p: Pickup) -> str:
    shelter = f"{p.shelter.name} ({p.shelter.address})" if p.shelter else "the shelter we'll text you"
    restaurant = p.restaurant.name if p.restaurant else "a restaurant"
    return (
        f"HungerCue pickup: {p.quantity} {p.items} from {restaurant}, {p.address}, {window_text(p)}. "
        f"Deliver to {shelter}. Reply YES to accept or NO to pass."
    )


def confirmation_text(p: Pickup) -> str:
    name = p.volunteer.name.split()[0] if p.volunteer and p.volunteer.name else "A volunteer"
    return (
        f"HungerCue: your donation of {p.items} is confirmed. {name} will pick it up {window_text(p)}"
        + (f" and bring it to {p.shelter.name}." if p.shelter else ".")
        + " Thank you!"
    )


def reminder_text(p: Pickup) -> str:
    return f"Reminder from HungerCue: pickup at {p.address} ({p.restaurant.name if p.restaurant else 'restaurant'}) starts at {_time(p.pickup_start)}."


class FollowUp:
    def __init__(
        self,
        repo: Repo,
        matcher: Matcher,
        texter: Texter,
        settings: Settings,
        on_pickup_change: Callable[[Pickup], Awaitable[None]] | None = None,
    ) -> None:
        self.repo = repo
        self.on_pickup_change = on_pickup_change  # e.g. update the Pickups tab in Google Sheets
        self.matcher = matcher
        self.texter = texter
        self.timeout = timedelta(minutes=settings.volunteer_reply_timeout_min)
        self.reminder_lead = timedelta(minutes=settings.reminder_lead_min)
        self.tz = settings.timezone
        self._tasks: set[asyncio.Task[None]] = set()

    async def _changed(self, pickup_id: int) -> None:
        """Tell the observer (Sheets) the pickup's status changed. Never raises."""
        if self.on_pickup_change is None:
            return
        try:
            p = await self.repo.get_pickup(pickup_id)
            if p is not None:
                await self.on_pickup_change(p)
        except Exception:
            log.exception("pickup change hook failed for %s", pickup_id)

    # ---------- entry points ----------

    def donation_logged_background(self, pickup_id: int) -> None:
        """Called from the live call: start matching without making the caller wait.

        We keep a reference to the task because asyncio only holds weak
        references, and an unreferenced task can be garbage-collected mid-run.
        """
        task = asyncio.create_task(self.donation_logged(pickup_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def donation_logged(self, pickup_id: int) -> None:
        try:
            await self.offer_next(pickup_id)
        except Exception:
            log.exception("follow-up for pickup %s failed", pickup_id)

    async def offer_next(self, pickup_id: int, now: datetime | None = None) -> bool:
        """Match the pickup (skipping volunteers already asked) and text the offer."""
        match = await self.matcher.match(pickup_id)
        if match is None:
            p = await self.repo.get_pickup(pickup_id)
            if p is not None and p.restaurant is not None:
                await self.texter.send(
                    p.restaurant.phone,
                    "HungerCue: thanks for your donation! We're still finding a volunteer; a coordinator will follow up shortly.",
                    kind="unmatched",
                    pickup_id=pickup_id,
                )
            await self._changed(pickup_id)
            return False
        return await self.send_offer(pickup_id, now)

    async def send_offer(self, pickup_id: int, now: datetime | None = None) -> bool:
        """Text the offer to the volunteer ALREADY assigned to the pickup.

        Separate from offer_next so the dashboard's Hungarian rebalance can send
        offers for its own assignments without greedy matching overriding them.
        """
        p = await self.repo.get_pickup(pickup_id)
        if p is None or p.volunteer is None:
            return False
        await self.repo.update_pickup(
            pickup_id,
            status="offered",
            offered_at=now or now_local(self.tz),
            offered_volunteer_ids=[*p.offered_volunteer_ids, p.volunteer.id],
        )
        await self.texter.send(p.volunteer.phone, offer_text(p), kind="offer", pickup_id=pickup_id)
        await self._changed(pickup_id)
        return True

    async def handle_reply(self, from_number: str, body: str) -> str:
        """Process an inbound text. Returns a short reply to send back ("" for none)."""
        phone = normalize_phone(from_number)
        await self.repo.log_sms("inbound", phone, body, kind="reply", status="received")
        volunteer = await self.repo.volunteer_by_phone(phone)
        if volunteer is None:
            return ""
        pickup = await self.repo.offered_pickup_for_volunteer(volunteer.id)
        if pickup is None:
            return "HungerCue: you have no open pickup offers right now. Thanks for volunteering!"
        word = body.strip().lower().split()[0] if body.strip() else ""
        if word in YES_WORDS:
            await self.accept(pickup.id)
            return "HungerCue: thank you! You're confirmed. We'll text a reminder 30 minutes before."
        if word in NO_WORDS:
            await self.offer_next(pickup.id)
            return "HungerCue: no problem, we'll ask someone else."
        return "HungerCue: please reply YES to accept the pickup or NO to pass."

    async def accept(self, pickup_id: int) -> None:
        await self.repo.update_pickup(pickup_id, status="accepted")
        p = await self.repo.get_pickup(pickup_id)
        if p is not None and p.restaurant is not None:
            await self.texter.send(p.restaurant.phone, confirmation_text(p), kind="confirmation", pickup_id=pickup_id)
        await self._changed(pickup_id)

    # ---------- the sweeper ----------

    async def sweep(self, now: datetime | None = None) -> dict[str, int]:
        """One pass: expire stale offers and send due reminders. Returns counts for logging/tests."""
        now = now or now_local(self.tz)
        counts = {"reoffered": 0, "reminded": 0}
        for p in await self.repo.pickups_with_status("offered"):
            if p.offered_at is not None and now - p.offered_at >= self.timeout:
                log.info("pickup %s: no reply from volunteer %s in time", p.id, p.volunteer_id)
                await self.offer_next(p.id, now=now)
                counts["reoffered"] += 1
        for p in await self.repo.pickups_with_status("accepted"):
            if not p.reminder_sent and p.volunteer is not None and p.pickup_start - self.reminder_lead <= now < p.pickup_start:
                await self.texter.send(p.volunteer.phone, reminder_text(p), kind="reminder", pickup_id=p.id)
                await self.repo.update_pickup(p.id, reminder_sent=True)
                counts["reminded"] += 1
        return counts

    async def run_sweeper(self, interval_s: float = 30.0) -> None:
        """Loop forever (started at app startup, cancelled at shutdown)."""
        while True:
            try:
                await self.sweep()
            except Exception:
                log.exception("sweep failed")
            await asyncio.sleep(interval_s)
