"""Choosing who gets which donation.

Greedy (used live, right after a call): score every (shelter, volunteer)
pair for the new donation and take the best. Fast, simple, and it can
re-run instantly with the declining volunteer excluded.

Hungarian (batch, the stretch goal): with a whole day's donations at once,
greedy can be short-sighted. It gives donation A the volunteer that donation
B *needed* (say the only one with a car near B), even though A had another
good option. The Hungarian algorithm (scipy.optimize.linear_sum_assignment)
finds the one-volunteer-per-donation assignment with the best TOTAL score.
It's the classic "assignment problem", solved exactly in O(n^3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from hungercue.matching.scoring import DonationInfo, PairScore, ShelterInfo, VolunteerInfo, Weights, score_pair


def rank_pairs(
    d: DonationInfo,
    shelters: list[ShelterInfo],
    volunteers: list[VolunteerInfo],
    w: Weights,
    exclude_volunteers: set[int] | None = None,
) -> list[PairScore]:
    """All feasible pairs for one donation, best first."""
    exclude = exclude_volunteers or set()
    pairs = [
        p
        for v in volunteers
        if v.id not in exclude
        for s in shelters
        if (p := score_pair(d, s, v, w)) is not None
    ]
    # Ties broken by ids so results are deterministic (important for tests and audits).
    return sorted(pairs, key=lambda p: (-p.score, p.volunteer_id, p.shelter_id))


def greedy_best(
    d: DonationInfo,
    shelters: list[ShelterInfo],
    volunteers: list[VolunteerInfo],
    w: Weights,
    exclude_volunteers: set[int] | None = None,
) -> PairScore | None:
    ranked = rank_pairs(d, shelters, volunteers, w, exclude_volunteers)
    return ranked[0] if ranked else None


@dataclass
class BatchResult:
    assignments: dict[int, PairScore]  # donation id -> chosen pair
    unassigned: list[int]  # donation ids no feasible pair could serve
    total_score: float


INFEASIBLE = 1e9  # cost for impossible pairings; real costs are tiny next to it


def hungarian_batch(
    donations: list[DonationInfo],
    shelters: list[ShelterInfo],
    volunteers: list[VolunteerInfo],
    w: Weights,
) -> BatchResult:
    """Assign each donation a distinct volunteer, maximising the total score.

    Matrix: rows = donations, columns = volunteers; each cell holds the best
    shelter's score for that (donation, volunteer). linear_sum_assignment
    *minimises* cost, so we negate scores. Infeasible cells get a huge cost
    and are dropped afterwards. Shelter capacity is shared between donations,
    so after solving we re-check it in score order and move an over-full
    donation to that volunteer's next-best shelter.
    """
    if not donations or not volunteers:
        return BatchResult({}, [d.id for d in donations], 0.0)

    best: dict[tuple[int, int], list[PairScore]] = {}
    cost = np.full((len(donations), len(volunteers)), INFEASIBLE)
    for i, d in enumerate(donations):
        for j, v in enumerate(volunteers):
            options = rank_pairs(d, shelters, [v], w)
            if options:
                best[(i, j)] = options
                cost[i, j] = -options[0].score

    rows, cols = linear_sum_assignment(cost)

    chosen: list[tuple[int, list[PairScore]]] = [
        (i, best[(i, j)]) for i, j in zip(rows, cols) if cost[i, j] < INFEASIBLE
    ]
    # Re-check shared shelter capacity, best-scoring assignments first.
    by_id = {s.id: s for s in shelters}
    used: dict[int, int] = {}
    assignments: dict[int, PairScore] = {}
    for i, options in sorted(chosen, key=lambda item: -item[1][0].score):
        d = donations[i]
        for pair in options:
            s = by_id[pair.shelter_id]
            room = s.capacity - s.committed - used.get(s.id, 0) if s.capacity > 0 else None
            if room is None or room >= max(1, d.servings):
                assignments[d.id] = pair
                used[s.id] = used.get(s.id, 0) + d.servings
                break
    unassigned = [d.id for d in donations if d.id not in assignments]
    return BatchResult(assignments, unassigned, round(sum(p.score for p in assignments.values()), 4))


def greedy_batch(
    donations: list[DonationInfo],
    shelters: list[ShelterInfo],
    volunteers: list[VolunteerInfo],
    w: Weights,
) -> BatchResult:
    """Greedy over a batch, in arrival order (what happens live), for comparison."""
    taken: set[int] = set()
    assignments: dict[int, PairScore] = {}
    for d in donations:
        pick = greedy_best(d, shelters, volunteers, w, exclude_volunteers=taken)
        if pick:
            assignments[d.id] = pick
            taken.add(pick.volunteer_id)
    unassigned = [d.id for d in donations if d.id not in assignments]
    return BatchResult(assignments, unassigned, round(sum(p.score for p in assignments.values()), 4))
