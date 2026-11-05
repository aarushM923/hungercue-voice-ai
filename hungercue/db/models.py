"""Database tables (SQLAlchemy 2 ORM).

Each class is one table. `Mapped[...]` type hints tell SQLAlchemy the column
type, so the same models work on SQLite (local/tests) and Postgres (production).

Time convention: every timestamp is a *naive local time* in the configured
timezone (America/New_York by default). Pickups are local events ("between 3
and 5pm"), and coordinators read local times, so storing local time avoids
off-by-four-hours bugs in reminders and the dashboard.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from hungercue.timeutil import now_local


class Base(DeclarativeBase):
    """Parent class that collects every table definition for create_all()."""


class Restaurant(Base):
    __tablename__ = "restaurants"

    id: Mapped[int] = mapped_column(primary_key=True)
    phone: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    contact_name: Mapped[str] = mapped_column(String(200), default="")
    address: Mapped[str] = mapped_column(String(300), default="")
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    language: Mapped[str] = mapped_column(String(5), default="en")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_local)

    pickups: Mapped[list[Pickup]] = relationship(back_populates="restaurant")


class Shelter(Base):
    __tablename__ = "shelters"

    id: Mapped[int] = mapped_column(primary_key=True)
    phone: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    contact_name: Mapped[str] = mapped_column(String(200), default="")
    address: Mapped[str] = mapped_column(String(300), default="")
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    capacity: Mapped[int] = mapped_column(Integer, default=0)  # meals/servings it can take today
    needs: Mapped[list[str]] = mapped_column(JSON, default=list)  # e.g. ["produce", "prepared meals"]
    dietary_restrictions: Mapped[list[str]] = mapped_column(JSON, default=list)  # e.g. ["halal", "no pork"]
    receiving_start: Mapped[str] = mapped_column(String(5), default="09:00")  # HH:MM local
    receiving_end: Mapped[str] = mapped_column(String(5), default="17:00")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    language: Mapped[str] = mapped_column(String(5), default="en")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_local, onupdate=now_local)


class Volunteer(Base):
    __tablename__ = "volunteers"

    id: Mapped[int] = mapped_column(primary_key=True)
    phone: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    neighborhoods: Mapped[list[str]] = mapped_column(JSON, default=list)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)  # centre of first neighbourhood
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    has_car: Mapped[bool] = mapped_column(Boolean, default=False)
    # List of {"day": "mon".."sun", "start": "HH:MM", "end": "HH:MM"} windows.
    availability: Mapped[list[dict[str, str]]] = mapped_column(JSON, default=list)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    language: Mapped[str] = mapped_column(String(5), default="en")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_local, onupdate=now_local)


class Pickup(Base):
    """One donation to collect. Status moves through:

    pending_match -> offered (SMS sent to a volunteer) -> accepted -> completed
                  -> unmatched (no shelter/volunteer fits; a coordinator steps in)
    """

    __tablename__ = "pickups"

    id: Mapped[int] = mapped_column(primary_key=True)
    restaurant_id: Mapped[int] = mapped_column(ForeignKey("restaurants.id"))
    items: Mapped[str] = mapped_column(Text, default="")
    quantity: Mapped[str] = mapped_column(String(200), default="")
    servings: Mapped[int] = mapped_column(Integer, default=0)  # rough count, used against shelter capacity
    food_tags: Mapped[list[str]] = mapped_column(JSON, default=list)  # e.g. ["prepared meals", "contains pork"]
    pickup_start: Mapped[datetime] = mapped_column(DateTime)
    pickup_end: Mapped[datetime] = mapped_column(DateTime)
    address: Mapped[str] = mapped_column(String(300), default="")
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="pending_match", index=True)
    shelter_id: Mapped[int | None] = mapped_column(ForeignKey("shelters.id"), nullable=True)
    volunteer_id: Mapped[int | None] = mapped_column(ForeignKey("volunteers.id"), nullable=True)
    # Volunteers already offered this pickup, so the 15-minute fallback skips them.
    offered_volunteer_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    offered_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    match_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    reminder_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    call_sid: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_local)

    restaurant: Mapped[Restaurant] = relationship(back_populates="pickups")
    shelter: Mapped[Shelter | None] = relationship()
    volunteer: Mapped[Volunteer | None] = relationship()


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    call_sid: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    from_number: Mapped[str] = mapped_column(String(20), default="")
    caller_type: Mapped[str] = mapped_column(String(20), default="")  # restaurant | shelter | volunteer
    language: Mapped[str] = mapped_column(String(5), default="en")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=now_local)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # List of {"role": "caller"|"agent", "text": str, "at": iso time}.
    transcript: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    outcome: Mapped[str] = mapped_column(String(30), default="")  # pickup_booked, transferred, voicemail, ...
    audio_path: Mapped[str] = mapped_column(String(300), default="")
    voicemail_url: Mapped[str] = mapped_column(String(300), default="")  # Twilio recording, if left
    keep_audio: Mapped[bool] = mapped_column(Boolean, default=False)  # caller agreed to longer retention
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)


class Turn(Base):
    """Latency breakdown for one agent reply, so we can show where time goes."""

    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(primary_key=True)
    call_sid: Mapped[str] = mapped_column(String(64), index=True)
    turn_index: Mapped[int] = mapped_column(Integer)
    endpoint_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    stt_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_first_token_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tts_first_byte_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    v2v_ms: Mapped[float | None] = mapped_column(Float, nullable=True)  # voice-to-voice total
    interrupted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_local)


class SmsMessage(Base):
    """Log of every text sent or received, for the dashboard and debugging."""

    __tablename__ = "sms_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    direction: Mapped[str] = mapped_column(String(10))  # outbound | inbound
    phone: Mapped[str] = mapped_column(String(20), index=True)
    body: Mapped[str] = mapped_column(Text)
    pickup_id: Mapped[int | None] = mapped_column(ForeignKey("pickups.id"), nullable=True)
    kind: Mapped[str] = mapped_column(String(30), default="")  # confirmation, offer, reminder, reply
    # sent | pending_verification (SMS disabled until toll-free verification) | failed | received
    status: Mapped[str] = mapped_column(String(30), default="sent")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_local)
