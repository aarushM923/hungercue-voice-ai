"""Matching engine tests: hard rules, scoring, greedy vs Hungarian."""

from datetime import datetime

from hungercue.matching.assign import greedy_batch, greedy_best, hungarian_batch, rank_pairs
from hungercue.matching.scoring import (
    DonationInfo, ShelterInfo, VolunteerInfo, Weights, dietary_conflict, fit, receiving_overlap,
    score_pair, volunteer_available,
)

TUE_3PM = datetime(2026, 11, 3, 15)  # a Tuesday
TUE_5PM = datetime(2026, 11, 3, 17)
HARLEM = (40.8116, -73.9465)
UWS = (40.7870, -73.9754)
BROOKLYN = (40.6782, -73.9442)
W = Weights()


def donation(id: int = 1, tags: list[str] | None = None, servings: int = 30, loc=HARLEM, start=TUE_3PM, end=TUE_5PM):
    return DonationInfo(id, start, end, servings, tags or ["bread"], loc)


def test_dietary_rules() -> None:
    assert dietary_conflict(["contains pork"], ["halal"])
    assert dietary_conflict(["contains meat"], ["halal"])
    assert not dietary_conflict(["contains meat", "halal"], ["halal"])
    assert dietary_conflict(["contains pork"], ["No Pork"])
    assert dietary_conflict(["contains meat"], ["vegetarian"])
    assert dietary_conflict(["contains nuts"], ["nut free"])
    assert dietary_conflict(["contains meat"], ["kosher"]) and not dietary_conflict(["bread"], ["kosher"])
    assert not dietary_conflict(["bread", "vegetarian"], ["halal", "vegetarian", "nut free"])


def test_fit_needs_synonyms_and_capacity() -> None:
    d = donation(tags=["bagels"], servings=40)
    assert fit(d, ShelterInfo(1, needs=["bread"])) == 1.0  # "bagels" counts as bread
    assert fit(d, ShelterInfo(1, needs=["produce"])) == 0.3
    assert fit(d, ShelterInfo(1)) == 0.7
    assert fit(d, ShelterInfo(1, needs=["bread"], capacity=50, committed=30)) == 0.25  # room for 20 of 40, squared


def test_receiving_overlap_and_volunteer_availability() -> None:
    d = donation()
    assert receiving_overlap(d, ShelterInfo(1, receiving_start="16:00", receiving_end="20:00")) == 1.0
    assert receiving_overlap(d, ShelterInfo(1, receiving_start="18:00", receiving_end="20:00")) == 0.0
    tue = [{"day": "tue", "start": "14:00", "end": "18:00"}]
    assert volunteer_available(d, VolunteerInfo(1, availability=tue))
    assert not volunteer_available(d, VolunteerInfo(1, availability=[{"day": "wed", "start": "14:00", "end": "18:00"}]))
    assert not volunteer_available(d, VolunteerInfo(1, availability=[{"day": "tue", "start": "16:50", "end": "18:00"}]))
    assert volunteer_available(d, VolunteerInfo(1))  # no hours given: allowed
    assert not volunteer_available(d, VolunteerInfo(1, busy=[(TUE_3PM, TUE_5PM)]))


def test_score_pair_hard_rules_return_none() -> None:
    d = donation(tags=["prepared meals", "contains pork"])
    v = VolunteerInfo(1, has_car=True, location=HARLEM)
    assert score_pair(d, ShelterInfo(1, restrictions=["halal"]), v, W) is None
    assert score_pair(d, ShelterInfo(1, receiving_start="06:00", receiving_end="09:00"), v, W) is None
    assert score_pair(d, ShelterInfo(1, capacity=10, committed=10), v, W) is None
    assert score_pair(d, ShelterInfo(1), v, W) is not None


def test_closer_volunteer_and_shelter_score_higher() -> None:
    d = donation(loc=HARLEM)
    near_shelter, far_shelter = ShelterInfo(1, location=UWS), ShelterInfo(2, location=BROOKLYN)
    near_vol, far_vol = VolunteerInfo(1, has_car=True, location=HARLEM), VolunteerInfo(2, has_car=True, location=BROOKLYN)
    best = greedy_best(d, [far_shelter, near_shelter], [far_vol, near_vol], W)
    assert (best.shelter_id, best.volunteer_id) == (1, 1)
    p = score_pair(d, near_shelter, near_vol, W)
    expected = W.fit * p.fit + W.overlap * p.overlap_h - W.dist_dv * p.dist_dv_min - W.dist_vs * p.dist_vs_min
    assert abs(p.score - expected) < 0.01  # the spec's formula, term by term
    # Transit is slower than driving, so the same trip scores lower without a car.
    no_car = score_pair(d, near_shelter, VolunteerInfo(3, has_car=False, location=HARLEM), W)
    assert no_car.score < p.score


def test_exclusions_and_no_feasible_pair() -> None:
    d = donation()
    shelters = [ShelterInfo(1, location=UWS)]
    vols = [VolunteerInfo(1, location=HARLEM), VolunteerInfo(2, location=BROOKLYN)]
    assert greedy_best(d, shelters, vols, W, exclude_volunteers={1}).volunteer_id == 2
    assert greedy_best(d, shelters, vols, W, exclude_volunteers={1, 2}) is None
    assert len(rank_pairs(d, shelters, vols, W)) == 2


def test_hungarian_beats_greedy_when_greedy_is_short_sighted() -> None:
    """Donation A arrives first and greedily takes Ana, the ONLY volunteer free
    for donation B. Hungarian gives A to Ben instead so both get picked up."""
    a = donation(1, loc=HARLEM)
    b = donation(2, loc=HARLEM, start=datetime(2026, 11, 3, 19), end=datetime(2026, 11, 3, 20))
    shelters = [ShelterInfo(1, location=UWS, receiving_start="09:00", receiving_end="21:00")]
    ana = VolunteerInfo(1, has_car=True, location=HARLEM)  # free all day, very close
    ben = VolunteerInfo(2, has_car=True, location=UWS,  # a bit farther, only free in the afternoon
                        availability=[{"day": "tue", "start": "14:00", "end": "18:00"}])
    greedy = greedy_batch([a, b], shelters, [ana, ben], W)
    hung = hungarian_batch([a, b], shelters, [ana, ben], W)
    assert greedy.assignments[1].volunteer_id == 1 and greedy.unassigned == [2]
    assert {d: p.volunteer_id for d, p in hung.assignments.items()} == {1: 2, 2: 1}
    assert hung.unassigned == [] and hung.total_score > greedy.total_score


def test_hungarian_respects_shared_shelter_capacity() -> None:
    a, b = donation(1, servings=30), donation(2, servings=30)
    small = ShelterInfo(1, location=UWS, capacity=40)  # can't take both
    other = ShelterInfo(2, location=BROOKLYN)
    vols = [VolunteerInfo(1, location=HARLEM), VolunteerInfo(2, location=HARLEM)]
    res = hungarian_batch([a, b], [small, other], vols, W)
    assert sorted(p.shelter_id for p in res.assignments.values()) == [1, 2]


def test_hungarian_edge_cases() -> None:
    assert hungarian_batch([], [], [], W).assignments == {}
    assert hungarian_batch([donation()], [ShelterInfo(1)], [], W).unassigned == [1]
    impossible = donation(tags=["contains pork"])
    res = hungarian_batch([impossible], [ShelterInfo(1, restrictions=["halal"])], [VolunteerInfo(1)], W)
    assert res.unassigned == [1]
