"""Matching service: connects the pure scoring code to the database."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from hungercue.config import Settings
from hungercue.db.models import Pickup, Shelter, Volunteer
from hungercue.db.repo import Repo
from hungercue.matching.assign import BatchResult, greedy_best, hungarian_batch
from hungercue.matching.scoring import DonationInfo, PairScore, ShelterInfo, VolunteerInfo, Weights

log = logging.getLogger(__name__)

ACTIVE = ("offered", "accepted")  # statuses that hold a shelter's capacity and a volunteer's time


def _point(lat: float | None, lon: float | None) -> tuple[float, float] | None:
    return (lat, lon) if lat is not None and lon is not None else None


def donation_info(p: Pickup) -> DonationInfo:
    return DonationInfo(p.id, p.pickup_start, p.pickup_end, p.servings, list(p.food_tags), _point(p.lat, p.lon))


class Matcher:
    def __init__(self, repo: Repo, settings: Settings) -> None:
        self.repo = repo
        self.weights = Weights(
            settings.match_w_fit, settings.match_w_overlap, settings.match_w_dist_dv, settings.match_w_dist_vs
        )

    async def _candidates(self, day: datetime, skip_pickup: int | None) -> tuple[list[ShelterInfo], list[VolunteerInfo]]:
        """Shelters with today's remaining capacity, volunteers with today's commitments."""
        start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        same_day = [p for p in await self.repo.pickups_between(start, start + timedelta(days=1))
                    if p.status in ACTIVE and p.id != skip_pickup]
        committed: dict[int, int] = {}
        busy: dict[int, list[tuple[datetime, datetime]]] = {}
        for p in same_day:
            if p.shelter_id is not None:
                committed[p.shelter_id] = committed.get(p.shelter_id, 0) + p.servings
            if p.volunteer_id is not None:
                busy.setdefault(p.volunteer_id, []).append((p.pickup_start, p.pickup_end))
        shelters = [self._shelter(s, committed.get(s.id, 0)) for s in await self.repo.active_shelters()]
        volunteers = [self._volunteer(v, busy.get(v.id, [])) for v in await self.repo.active_volunteers()]
        return shelters, volunteers

    @staticmethod
    def _shelter(s: Shelter, committed: int) -> ShelterInfo:
        return ShelterInfo(s.id, s.receiving_start, s.receiving_end, s.capacity, committed,
                           list(s.needs), list(s.dietary_restrictions), _point(s.lat, s.lon))

    @staticmethod
    def _volunteer(v: Volunteer, busy: list[tuple[datetime, datetime]]) -> VolunteerInfo:
        return VolunteerInfo(v.id, list(v.availability), v.has_car, _point(v.lat, v.lon), busy)

    async def match(self, pickup_id: int) -> PairScore | None:
        """Pick the best shelter + volunteer for one pickup and save it.

        Volunteers already offered this pickup (declined or timed out) are skipped,
        which is how the 15-minute fallback finds the next-best volunteer.
        """
        p = await self.repo.get_pickup(pickup_id)
        if p is None:
            return None
        shelters, volunteers = await self._candidates(p.pickup_start, skip_pickup=p.id)
        best = greedy_best(donation_info(p), shelters, volunteers, self.weights, set(p.offered_volunteer_ids))
        if best is None:
            await self.repo.update_pickup(p.id, status="unmatched", volunteer_id=None)
            log.info("pickup %s: no feasible shelter/volunteer", p.id)
            return None
        await self.repo.update_pickup(p.id, shelter_id=best.shelter_id, volunteer_id=best.volunteer_id,
                                      match_score=best.score)
        log.info("pickup %s matched: %s", p.id, best)
        return best

    async def rebalance_day(self, day: datetime) -> tuple[BatchResult, list[int]]:
        """Hungarian re-optimisation of a day's not-yet-accepted pickups.

        Only pickups no volunteer has accepted are moved, so nobody who said
        YES gets bumped. Returns the batch result plus the ids of pickups that
        need a NEW offer text: an offer still going to the same volunteer is
        left as it was, so nobody gets the same text twice.
        """
        start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        todays = await self.repo.pickups_between(start, start + timedelta(days=1))
        movable = [p for p in todays if p.status in ("pending_match", "unmatched", "offered")]
        before = {p.id: (p.status, p.volunteer_id) for p in movable}
        # Free the movable pickups' holds before computing capacity and busy times.
        for p in movable:
            await self.repo.update_pickup(p.id, status="pending_match")
        shelters, volunteers = await self._candidates(start, skip_pickup=None)
        result = hungarian_batch([donation_info(p) for p in movable], shelters, volunteers, self.weights)
        needs_offer: list[int] = []
        for p in movable:
            pair = result.assignments.get(p.id)
            if pair is None:
                await self.repo.update_pickup(p.id, status="unmatched", volunteer_id=None)
            elif before[p.id] == ("offered", pair.volunteer_id):
                # Same volunteer already has the offer: keep its clock and history.
                await self.repo.update_pickup(p.id, status="offered", shelter_id=pair.shelter_id, match_score=pair.score)
            else:
                await self.repo.update_pickup(p.id, shelter_id=pair.shelter_id, volunteer_id=pair.volunteer_id,
                                              match_score=pair.score, offered_volunteer_ids=[])
                needs_offer.append(p.id)
        return result, needs_offer
