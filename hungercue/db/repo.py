"""Repository: every database read/write the app performs, in one place.

Keeping SQL out of the call pipeline, tools and dashboard means each of those
can be tested with a throwaway SQLite database, and switching to Postgres
changes nothing here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from hungercue.db.models import Call, Pickup, Restaurant, Shelter, SmsMessage, Turn, Volunteer
from hungercue.db.session import Database
from hungercue.timeutil import now_local

M = TypeVar("M", Restaurant, Shelter, Volunteer)


@dataclass
class CallerProfile:
    """What we know about a phone number before the caller says a word."""

    kind: str  # restaurant | shelter | volunteer | "" (unknown)
    record_id: int | None = None
    name: str = ""
    address: str = ""
    summary: str = ""  # one line the LLM can use to greet a returning caller


class Repo:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ---------- returning-caller lookup ----------

    async def lookup_caller(self, phone: str) -> CallerProfile:
        """Check all three directories for this number.

        Lets the agent open with "Hi, is this another pickup from the bagel
        shop on 112th?" instead of asking who's calling.
        """
        if not phone:
            return CallerProfile(kind="")
        async with self.db.session() as s:
            r = await s.scalar(select(Restaurant).where(Restaurant.phone == phone))
            if r:
                last = await s.scalar(
                    select(Pickup).where(Pickup.restaurant_id == r.id).order_by(Pickup.created_at.desc()).limit(1)
                )
                summary = f"Restaurant '{r.name}' at {r.address}."
                if last:
                    summary += f" Last donation: {last.quantity} {last.items} on {last.pickup_start:%b %d}."
                return CallerProfile("restaurant", r.id, r.name, r.address, summary)
            sh = await s.scalar(select(Shelter).where(Shelter.phone == phone))
            if sh:
                summary = (
                    f"Shelter '{sh.name}' at {sh.address}. Capacity {sh.capacity}, needs {', '.join(sh.needs) or 'none listed'}, "
                    f"receives {sh.receiving_start}-{sh.receiving_end}."
                )
                return CallerProfile("shelter", sh.id, sh.name, sh.address, summary)
            v = await s.scalar(select(Volunteer).where(Volunteer.phone == phone))
            if v:
                summary = (
                    f"Volunteer {v.name}, neighborhoods {', '.join(v.neighborhoods) or 'unknown'}, "
                    f"{'has a car' if v.has_car else 'no car'}."
                )
                return CallerProfile("volunteer", v.id, v.name, "", summary)
        return CallerProfile(kind="")

    # ---------- directory upserts ----------

    async def _upsert(self, model: type[M], phone: str, fields: dict[str, Any]) -> M:
        """Insert or update by phone. Fields set to None are left unchanged,
        so a caller who only updates capacity doesn't wipe their address."""
        async with self.db.session() as s:
            row = await s.scalar(select(model).where(model.phone == phone))
            if row is None:
                row = model(phone=phone)
                s.add(row)
            for key, value in fields.items():
                if value is not None:
                    setattr(row, key, value)
            await s.flush()  # assigns row.id without ending the transaction
            return row

    async def upsert_restaurant(self, phone: str, **fields: Any) -> Restaurant:
        return await self._upsert(Restaurant, phone, fields)

    async def upsert_shelter(self, phone: str, **fields: Any) -> Shelter:
        return await self._upsert(Shelter, phone, fields)

    async def upsert_volunteer(self, phone: str, **fields: Any) -> Volunteer:
        return await self._upsert(Volunteer, phone, fields)

    async def active_shelters(self) -> list[Shelter]:
        async with self.db.session() as s:
            return list(await s.scalars(select(Shelter).where(Shelter.active.is_(True))))

    async def active_volunteers(self) -> list[Volunteer]:
        async with self.db.session() as s:
            return list(await s.scalars(select(Volunteer).where(Volunteer.active.is_(True))))

    async def get_volunteer(self, volunteer_id: int) -> Volunteer | None:
        async with self.db.session() as s:
            return await s.get(Volunteer, volunteer_id)

    async def volunteer_by_phone(self, phone: str) -> Volunteer | None:
        async with self.db.session() as s:
            return await s.scalar(select(Volunteer).where(Volunteer.phone == phone))

    # ---------- pickups ----------

    async def create_pickup(self, restaurant_id: int, **fields: Any) -> Pickup:
        async with self.db.session() as s:
            p = Pickup(restaurant_id=restaurant_id, **fields)
            s.add(p)
            await s.flush()
            return p

    async def get_pickup(self, pickup_id: int) -> Pickup | None:
        """Load a pickup together with its restaurant, shelter and volunteer.

        selectinload fetches the related rows up front; async SQLAlchemy can't
        lazily load them later outside the session.
        """
        async with self.db.session() as s:
            return await s.scalar(
                select(Pickup)
                .where(Pickup.id == pickup_id)
                .options(selectinload(Pickup.restaurant), selectinload(Pickup.shelter), selectinload(Pickup.volunteer))
            )

    async def update_pickup(self, pickup_id: int, **fields: Any) -> Pickup | None:
        async with self.db.session() as s:
            p = await s.get(Pickup, pickup_id)
            if p is None:
                return None
            for key, value in fields.items():
                setattr(p, key, value)
            return p

    async def pickups_between(self, start: datetime, end: datetime) -> list[Pickup]:
        async with self.db.session() as s:
            rows = await s.scalars(
                select(Pickup)
                .where(Pickup.pickup_start >= start, Pickup.pickup_start < end)
                .options(selectinload(Pickup.restaurant), selectinload(Pickup.shelter), selectinload(Pickup.volunteer))
                .order_by(Pickup.pickup_start)
            )
            return list(rows)

    async def pickups_with_status(self, *statuses: str) -> list[Pickup]:
        async with self.db.session() as s:
            rows = await s.scalars(
                select(Pickup)
                .where(Pickup.status.in_(statuses))
                .options(selectinload(Pickup.restaurant), selectinload(Pickup.shelter), selectinload(Pickup.volunteer))
            )
            return list(rows)

    async def offered_pickup_for_volunteer(self, volunteer_id: int) -> Pickup | None:
        """The pickup currently waiting on this volunteer's YES, if any."""
        async with self.db.session() as s:
            return await s.scalar(
                select(Pickup)
                .where(Pickup.volunteer_id == volunteer_id, Pickup.status == "offered")
                .order_by(Pickup.offered_at.desc())
                .limit(1)
            )

    # ---------- calls, transcripts and latency ----------

    async def start_call(self, call_sid: str, from_number: str) -> Call:
        async with self.db.session() as s:
            existing = await s.scalar(select(Call).where(Call.call_sid == call_sid))
            if existing:
                return existing
            c = Call(call_sid=call_sid, from_number=from_number)
            s.add(c)
            await s.flush()
            return c

    async def add_transcript(self, call_sid: str, role: str, text: str) -> None:
        async with self.db.session() as s:
            c = await s.scalar(select(Call).where(Call.call_sid == call_sid))
            if c is None:
                return
            # Assign a *new* list: SQLAlchemy only notices JSON changes on reassignment.
            c.transcript = [*c.transcript, {"role": role, "text": text, "at": now_local().isoformat()}]

    async def update_call(self, call_sid: str, **fields: Any) -> None:
        async with self.db.session() as s:
            c = await s.scalar(select(Call).where(Call.call_sid == call_sid))
            if c is None:
                return
            for key, value in fields.items():
                setattr(c, key, value)

    async def get_call(self, call_sid: str) -> Call | None:
        async with self.db.session() as s:
            return await s.scalar(select(Call).where(Call.call_sid == call_sid))

    async def recent_calls(self, limit: int = 50) -> list[Call]:
        async with self.db.session() as s:
            return list(await s.scalars(select(Call).order_by(Call.started_at.desc()).limit(limit)))

    async def calls_with_audio_before(self, cutoff: datetime) -> list[Call]:
        async with self.db.session() as s:
            rows = await s.scalars(
                select(Call).where(Call.audio_path != "", Call.keep_audio.is_(False), Call.started_at < cutoff)
            )
            return list(rows)

    async def add_turn(self, call_sid: str, turn_index: int, **timings: Any) -> Turn:
        async with self.db.session() as s:
            t = Turn(call_sid=call_sid, turn_index=turn_index, **timings)
            s.add(t)
            await s.flush()
            return t

    async def turns_since(self, since: datetime) -> list[Turn]:
        async with self.db.session() as s:
            rows = await s.scalars(select(Turn).where(Turn.created_at >= since).order_by(Turn.created_at))
            return list(rows)

    # ---------- SMS log ----------

    async def log_sms(self, direction: str, phone: str, body: str, kind: str = "", pickup_id: int | None = None) -> None:
        async with self.db.session() as s:
            s.add(SmsMessage(direction=direction, phone=phone, body=body, kind=kind, pickup_id=pickup_id))

    async def sms_for_pickup(self, pickup_id: int) -> list[SmsMessage]:
        async with self.db.session() as s:
            rows = await s.scalars(select(SmsMessage).where(SmsMessage.pickup_id == pickup_id).order_by(SmsMessage.id))
            return list(rows)
