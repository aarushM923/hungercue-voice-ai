"""The 100 test calls: who is calling, what they know, and how hard the audio is.

Each scenario has GROUND TRUTH (the facts the simulated caller is given) and
CONDITIONS (noise, mumbling, interruptions, topic changes, voice/accent,
language). After the call we compare what the system stored against the
ground truth to get slot accuracy.

Scenarios are generated from a fixed random seed, so the same 100 calls are
replayed on every run. That matters: comparing two runs is only fair if the
callers are identical.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCENARIO_FILE = Path(__file__).parent / "scenarios.json"
SEED = 2026

# Caller voices (Deepgram Aura). Different voices stand in for different accents and timbres.
EN_VOICES = ["aura-2-thalia-en", "aura-2-apollo-en", "aura-2-helena-en", "aura-2-arcas-en",
             "aura-2-andromeda-en", "aura-2-orpheus-en", "aura-2-luna-en", "aura-2-zeus-en"]
ES_VOICES = ["aura-2-celeste-es", "aura-2-nestor-es", "aura-2-estrella-es", "aura-2-alvaro-es"]

# Item: (how the caller says it in English, in Spanish, keyword groups that must appear in what's stored).
# Each inner list is alternatives (any one counts), so a Spanish caller's "arroz" matches "rice".
ITEMS: list[tuple[str, str, list[list[str]]]] = [
    ("bagels and cream cheese", "bagels con queso crema", [["bagel"], ["cream cheese", "queso crema"]]),
    ("rice and beans", "arroz con frijoles", [["rice", "arroz"], ["bean", "frijol"]]),
    ("chicken soup", "sopa de pollo", [["soup", "sopa"], ["chicken", "pollo"]]),
    ("sandwiches", "sándwiches", [["sandwich", "sándwich"]]),
    ("fresh salads", "ensaladas frescas", [["salad", "ensalada"]]),
    ("pastries and muffins", "pasteles y muffins", [["pastr", "pastel", "pastri"], ["muffin"]]),
    ("pasta with vegetables", "pasta con verduras", [["pasta"], ["vegetable", "verdura", "veggie"]]),
    ("loaves of bread", "panes", [["bread", "pan", "loaves", "loaf"]]),
    ("fruit cups", "vasos de fruta", [["fruit", "fruta"]]),
    ("tacos", "tacos", [["taco"]]),
]
QUANTITIES = [(3, "trays", "bandejas"), (2, "large boxes", "cajas grandes"), (4, "bags", "bolsas"),
              (5, "trays", "bandejas"), (6, "boxes", "cajas"), (10, "containers", "recipientes")]
NUM_WORDS_ES = {2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 10: "diez"}
STREETS = [("212", "West 112th Street"), ("75", "Lenox Avenue"), ("1410", "Amsterdam Avenue"),
           ("38", "Bedford Avenue"), ("2210", "Broadway"), ("519", "Atlantic Avenue"),
           ("88", "Orchard Street"), ("301", "Malcolm X Boulevard"), ("47", "Steinway Street"),
           ("162", "Graham Avenue")]
BUSINESSES = ["Sunrise Bagels", "Casa Lupe", "Green Leaf Deli", "Harlem Kitchen", "Nonna's Pasta",
              "Corner Bakery", "Taqueria El Sol", "Fresh Bowl", "Mama's Soul Food", "Golden Wok"]
WINDOWS = [("15:00", "17:00"), ("14:00", "16:00"), ("16:30", "18:00"), ("18:00", "20:00"), ("13:00", "15:30")]
SHELTERS = ["Hope House", "Riverside Family Shelter", "St. Mark's Pantry", "Bridge Women's Center",
            "Uptown Youth Shelter"]
NEEDS = [["produce"], ["prepared meals"], ["bread"], ["prepared meals", "produce"], ["dairy"]]
DIETARY = [[], ["halal"], ["no pork"], ["vegetarian"], ["nut free"]]
HOURS = [("09:00", "17:00"), ("10:00", "19:00"), ("12:00", "20:00"), ("08:00", "14:00")]
NEIGHBORHOODS = ["Harlem", "Washington Heights", "Astoria", "Williamsburg", "Bed Stuy", "Upper West Side",
                 "East Harlem", "Park Slope", "Jackson Heights", "South Bronx"]
NAMES = ["Maria", "James", "Aisha", "Carlos", "Wei", "Priya", "Tom", "Fatima", "Luis", "Grace", "Omar", "Rosa"]
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
NOISES = ["kitchen", "street", "cafe"]


@dataclass
class Conditions:
    voice: str
    noise: str = "none"  # none | kitchen | street | cafe
    snr_db: float = 30.0  # signal-to-noise ratio; lower = noisier
    mumble: bool = False  # quiet, muffled speech
    interrupt: bool = False  # caller talks over the agent once
    topic_change: bool = False  # caller asks something unrelated mid-call


@dataclass
class Scenario:
    id: str
    caller_type: str  # restaurant | shelter | volunteer
    language: str  # en | es
    phone: str
    name: str
    truth: dict[str, Any]  # the facts to capture (see score.py for how each is checked)
    conditions: Conditions
    returning: bool = False  # restaurant already in the database: agent should recognise it
    said: dict[str, str] = field(default_factory=dict)  # how the caller phrases facts (fed to the caller LLM)


def _restaurant(i: int, rng: random.Random, lang: str, returning: bool) -> tuple[dict[str, Any], dict[str, str]]:
    item_en, item_es, keywords = rng.choice(ITEMS)
    qty, unit_en, unit_es = rng.choice(QUANTITIES)
    number, street = rng.choice(STREETS)
    start, end = rng.choice(WINDOWS)
    day = rng.choice(["today", "tomorrow"])
    truth = {
        "business_name": BUSINESSES[i % len(BUSINESSES)],
        "item_keywords": keywords,
        "quantity_number": qty,
        "pickup_day": day,
        "pickup_start": start,
        "pickup_end": end,
        "address_number": number,
        "address_street": street,
    }
    if lang == "es":
        said = {"items": item_es, "quantity": f"{NUM_WORDS_ES[qty]} {unit_es}",
                "day": "hoy" if day == "today" else "mañana", "address": f"{number} {street}"}
    else:
        said = {"items": item_en, "quantity": f"{qty} {unit_en}", "day": day, "address": f"{number} {street}"}
    said["window"] = f"{start} to {end} (24-hour clock; say it naturally, e.g. 'three to five pm')"
    if returning:
        said["note"] = "You have donated before; the agent may already know your business and address."
    return truth, said


def _shelter(i: int, rng: random.Random) -> tuple[dict[str, Any], dict[str, str]]:
    number, street = rng.choice(STREETS)
    start, end = rng.choice(HOURS)
    truth = {
        "shelter_name": SHELTERS[i % len(SHELTERS)],
        "capacity": rng.choice([40, 60, 80, 120, 150]),
        "needs": rng.choice(NEEDS),
        "dietary_restrictions": rng.choice(DIETARY),
        "receiving_start": start,
        "receiving_end": end,
        "address_number": number,
        "address_street": street,
    }
    return truth, {"hours": f"{start} to {end} (say it naturally)", "address": f"{number} {street}"}


def _volunteer(rng: random.Random) -> tuple[dict[str, Any], dict[str, str]]:
    days = sorted(rng.sample(DAYS, rng.choice([1, 2, 3])), key=DAYS.index)
    start, end = rng.choice([("09:00", "12:00"), ("13:00", "17:00"), ("17:00", "20:00"), ("10:00", "16:00")])
    hoods = rng.sample(NEIGHBORHOODS, rng.choice([1, 2]))
    truth = {
        "availability_days": days,
        "availability_start": start,
        "availability_end": end,
        "neighborhoods": hoods,
        "has_car": rng.random() < 0.5,
    }
    return truth, {"availability": f"{', '.join(days)} from {start} to {end} (say it naturally)"}


def generate(n: int = 100, seed: int = SEED) -> list[Scenario]:
    """Deterministic mix: 50% restaurants, 25% shelters, 25% volunteers;
    ~15% Spanish, up to 8 returning restaurants, and a spread of hard audio conditions."""
    rng = random.Random(seed)
    plan = ["restaurant"] * (n // 2) + ["shelter"] * (n // 4)
    plan += ["volunteer"] * (n - len(plan))
    rng.shuffle(plan)
    scenarios = []
    returning_left = max(1, n * 8 // 100)
    for i, kind in enumerate(plan):
        lang = "es" if rng.random() < 0.15 else "en"
        returning = kind == "restaurant" and returning_left > 0 and rng.random() < 0.25
        returning_left -= int(returning)
        noisy = rng.random() < 0.4
        cond = Conditions(
            voice=rng.choice(ES_VOICES if lang == "es" else EN_VOICES),
            noise=rng.choice(NOISES) if noisy else "none",
            snr_db=rng.choice([20.0, 15.0, 10.0]) if noisy else 30.0,
            mumble=rng.random() < 0.15,
            interrupt=rng.random() < 0.25,
            topic_change=rng.random() < 0.15,
        )
        if kind == "restaurant":
            truth, said = _restaurant(i, rng, lang, returning)
        elif kind == "shelter":
            truth, said = _shelter(i, rng)
        else:
            truth, said = _volunteer(rng)
        scenarios.append(Scenario(
            id=f"S{i + 1:03d}", caller_type=kind, language=lang, phone=f"+1555{7000000 + i:07d}",
            name=rng.choice(NAMES), truth=truth, conditions=cond, returning=returning, said=said,
        ))
    return scenarios


def smoke_subset(scenarios: list[Scenario], size: int = 20) -> list[Scenario]:
    """A fixed 20-call slice for CI: 50% restaurants, 25% shelters, 25% volunteers,
    evenly spaced within each type so it also spans languages and conditions."""
    quota = {"restaurant": size // 2, "shelter": size // 4, "volunteer": size - size // 2 - size // 4}
    picked: list[Scenario] = []
    for kind, k in quota.items():
        group = [s for s in scenarios if s.caller_type == kind]
        step = max(1, len(group) // k)
        picked += group[::step][:k]
    return sorted(picked, key=lambda s: s.id)


def save(scenarios: list[Scenario], path: Path = SCENARIO_FILE) -> None:
    path.write_text(json.dumps([asdict(s) for s in scenarios], indent=1, ensure_ascii=False))


def load(path: Path = SCENARIO_FILE) -> list[Scenario]:
    raw = json.loads(path.read_text())
    return [Scenario(**{**r, "conditions": Conditions(**r["conditions"])}) for r in raw]
