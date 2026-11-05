"""Tests for the eval suite itself: scenarios, audio, scoring, report, and a full offline harness run."""

import asyncio
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from evals import report
from evals.audio import FRAME, background_frame, degrade, make_noise, mulaw_frames, rms
from evals.caller import HANGUP, ScriptedCaller, persona_prompt
from evals.harness import run_scenario
from evals.scenarios import Conditions, Scenario, generate, load, smoke_subset
from evals.score import (
    address_ok, completed, keywords_ok, numbers_in, score_restaurant, score_shelter, score_volunteer, street_key,
)
from hungercue.config import Settings
from hungercue.db.models import Pickup, Shelter, Volunteer
from hungercue.db.repo import Repo
from hungercue.pipeline.call_session import Services
from hungercue.state.session_store import SessionStore, make_redis
from tests.fakes import LOUD, FakeOpenAI, FakeSTT, FakeTTS, loudness_vad, text_chunks, tool_chunks

CALL_DATE = datetime(2026, 11, 3, 10)


# ---------------------------------------------------------------- scenarios

def test_generation_is_deterministic_and_balanced() -> None:
    a, b = generate(), generate()
    assert [asdict(x) for x in a] == [asdict(x) for x in b]
    kinds = [s.caller_type for s in a]
    assert (kinds.count("restaurant"), kinds.count("shelter"), kinds.count("volunteer")) == (50, 25, 25)
    assert len({s.phone for s in a}) == 100
    assert 5 <= sum(s.language == "es" for s in a) <= 30
    assert sum(s.conditions.interrupt for s in a) >= 10 and sum(s.conditions.noise != "none" for s in a) >= 20


def test_committed_scenarios_match_generator() -> None:
    # The checked-in file is what CI replays; it must not drift from the code.
    assert [asdict(x) for x in load()] == [asdict(x) for x in generate()]


def test_smoke_subset_is_stratified() -> None:
    smoke = smoke_subset(generate())
    kinds = [s.caller_type for s in smoke]
    assert len(smoke) == 20 and (kinds.count("restaurant"), kinds.count("shelter"), kinds.count("volunteer")) == (10, 5, 5)


def test_persona_prompt_contains_facts_and_rules() -> None:
    s = next(x for x in generate() if x.caller_type == "restaurant" and x.language == "es")
    p = persona_prompt(s, CALL_DATE)
    assert "Spanish" in p and s.said["items"] in p and HANGUP in p and "one or two at a time" in p


# ---------------------------------------------------------------- audio

def test_noise_hits_target_snr() -> None:
    rng = np.random.default_rng(0)
    speech = (8000 * np.sin(np.linspace(0, 400, 16000))).astype(np.int16)
    noisy = degrade(speech, "cafe", 10.0, False, rng)
    noise = noisy.astype(float) - speech
    snr = 20 * np.log10(rms(speech) / rms(noise))
    assert 9.0 < snr < 11.0
    for kind in ("kitchen", "street", "cafe"):
        assert abs(rms(make_noise(kind, 8000, rng)) - 1.0) < 0.05  # unit loudness before scaling


def test_mumble_is_quieter_and_frames_are_twilio_sized() -> None:
    rng = np.random.default_rng(0)
    speech = (8000 * np.sin(np.linspace(0, 400, 1000))).astype(np.int16)
    assert rms(degrade(speech, "none", 30, True, rng)) < rms(speech) / 2
    frames = mulaw_frames(b"\x00" * 330)
    assert [len(f) for f in frames] == [FRAME, FRAME, FRAME] and frames[-1].endswith(b"\xff")
    assert len(background_frame("street", 15, rng)) == FRAME


# ---------------------------------------------------------------- scoring

def test_field_matchers() -> None:
    assert numbers_in("three trays") == {3} and numbers_in("tres bandejas") == {3} and numbers_in("10 boxes") == {10}
    assert street_key("West 112th Street") == "112" and street_key("Malcolm X Boulevard") == "malcolm"
    assert address_ok("212 W 112th St, New York", "212", "West 112th Street")
    assert not address_ok("213 W 112th St", "212", "West 112th Street")
    assert keywords_ok("Arroz y frijoles negros", [["rice", "arroz"], ["bean", "frijol"]])
    assert not keywords_ok("rice", [["rice"], ["bean"]])


def test_score_restaurant_fields() -> None:
    truth = {"business_name": "Casa Lupe", "item_keywords": [["rice", "arroz"], ["bean", "frijol"]],
             "quantity_number": 3, "pickup_day": "tomorrow", "pickup_start": "15:00", "pickup_end": "17:00",
             "address_number": "75", "address_street": "Lenox Avenue"}
    p = Pickup(items="rice and beans", quantity="three trays", address="75 Lenox Ave",
               pickup_start=datetime(2026, 11, 4, 15), pickup_end=datetime(2026, 11, 4, 16, 30))
    from hungercue.db.models import Restaurant
    fields = {f.field: f.ok for f in score_restaurant(truth, p, Restaurant(name="Casa Lupe"), CALL_DATE, False)}
    assert fields == {"items": True, "quantity": True, "pickup_date": True, "pickup_start": True,
                      "pickup_end": False, "address": True, "business_name": True}
    missing = score_restaurant(truth, None, None, CALL_DATE, True)
    assert not any(f.ok for f in missing) and "business_name" not in {f.field for f in missing}
    assert completed("restaurant", p, None, None) and not completed("restaurant", None, None, None)


def test_score_shelter_and_volunteer() -> None:
    truth = {"shelter_name": "Hope House", "capacity": 60, "needs": ["prepared meals"], "dietary_restrictions": ["halal"],
             "receiving_start": "10:00", "receiving_end": "19:00", "address_number": "38", "address_street": "Bedford Avenue"}
    s = Shelter(name="Hope House", capacity=60, needs=["hot food"], dietary_restrictions=["Halal"],
                receiving_start="10:00", receiving_end="19:00", address="38 Bedford Ave")
    assert all(f.ok for f in score_shelter(truth, s))  # "hot food" is a synonym for prepared meals
    s.dietary_restrictions = ["halal", "vegetarian"]  # an invented restriction is wrong too
    assert {f.field for f in score_shelter(truth, s) if not f.ok} == {"dietary_restrictions"}

    vt = {"availability_days": ["sat", "sun"], "availability_start": "09:00", "availability_end": "12:00",
          "neighborhoods": ["Bed Stuy"], "has_car": True}
    v = Volunteer(availability=[{"day": "sun", "start": "09:00", "end": "12:00"}, {"day": "sat", "start": "09:00", "end": "12:00"}],
                  neighborhoods=["Bedford-Stuyvesant"], has_car=True)
    assert all(f.ok for f in score_volunteer(vt, v))
    assert not any(f.ok for f in score_volunteer(vt, None))


# ---------------------------------------------------------------- report

def fake_result(sid: str, ok: list[bool], v2v: list[float], kind: str = "restaurant", **kw) -> dict:
    base = {"scenario_id": sid, "caller_type": kind, "language": "en", "classified_as": kind,
            "conditions": asdict(Conditions(voice="v")), "fields": [{"field": f"f{i}", "ok": o, "expected": 1, "got": 1} for i, o in enumerate(ok)],
            "completed": all(ok), "server_turns": [{"v2v_ms": x, "stt_ms": 100} for x in v2v], "client_v2v_ms": v2v,
            "barge_ins": [], "agent_cost_usd": 0.04, "eval_cost_usd": 0.01, "error": ""}
    return {**base, **kw}


def test_aggregate_and_compare() -> None:
    results = [fake_result("S1", [True, True, False], [600, 700]),
               fake_result("S2", [True, True, True], [800, 900], kind="shelter", classified_as="volunteer",
                           barge_ins=[{"reaction_ms": 300, "success": True}, {"reaction_ms": None, "success": False}])]
    m = report.aggregate(results)
    assert m["slot_accuracy"] == round(5 / 6, 4) and m["classification_accuracy"] == 0.5
    assert m["task_completion_rate"] == 0.5 and m["latency_ms"]["client_p50"] == 700
    assert m["barge_in"]["success_rate"] == 0.5 and m["cost_usd"]["agent_per_call"] == 0.04
    assert m["by_condition"]["shelter"]["calls"] == 1
    assert report.compare(m, m) == []
    worse = {**m, "slot_accuracy": m["slot_accuracy"] - 0.05,
             "latency_ms": {**m["latency_ms"], "client_p50": 700 * 1.2}}
    problems = report.compare(worse, m)
    assert len(problems) == 2 and "slot accuracy" in problems[0]
    md = report.to_markdown(m, results, "Test")
    assert "| Slot accuracy | 83.3% (5/6 fields) |" in md and "Missed fields" in md


# ---------------------------------------------------------------- the harness, end to end, offline

class CallerTTS:
    """Caller 'speech' is 0.5 s of loud audio, which the loudness VAD hears as speech."""

    async def synthesize(self, text: str, language: str = "en"):
        yield LOUD * 25


async def test_harness_runs_a_full_call_and_scores_it(repo: Repo, tmp_path: Path) -> None:
    scenario = Scenario(
        id="S900", caller_type="restaurant", language="en", phone="+15557000900", name="Maria",
        truth={"business_name": "Casa Lupe", "item_keywords": [["rice"], ["bean"]], "quantity_number": 3,
               "pickup_day": "today", "pickup_start": "15:00", "pickup_end": "17:00",
               "address_number": "75", "address_street": "Lenox Avenue"},
        conditions=Conditions(voice="v", interrupt=True),
    )
    donation = {"business_name": "Casa Lupe", "contact_name": "Maria", "items": "rice and beans", "quantity": "3 trays",
                "servings_estimate": 30, "food_tags": [], "pickup_date": f"{CALL_DATE:%Y-%m-%d}", "pickup_start": "15:00",
                "pickup_end": "17:00", "address": "75 Lenox Ave", "notes": None, "confirmed_by_caller": True}
    # Agent script: classify, ask a question (long reply, so the caller can interrupt it),
    # answer the interruption, then book and say goodbye.
    agent = FakeOpenAI([
        tool_chunks("set_caller_type", {"caller_type": "restaurant"}),
        text_chunks("Great, thank you so much for calling. What food do you have and where should we pick it up? "),
        text_chunks("Of course, take your time and tell me the food and the pickup details whenever you are ready. "),
        text_chunks("No problem at all. "),
        tool_chunks("log_donation", donation),
        text_chunks("You're all set, thanks! "),
    ])
    stt = FakeSTT()
    cfg = Settings(_env_file=None, recordings_dir=str(tmp_path / "rec"))
    store = SessionStore(make_redis(""))
    agent_tts = FakeTTS(chunks_per_text=100)  # 100 x 20 ms = 2 s per sentence, longer than the 1.2 s interrupt point

    def services(actions):
        return Services(cfg, repo, store, agent, agent_tts, lambda: stt, loudness_vad, actions=actions)

    async def hear(text, utt) -> None:  # "transcribe" each caller utterance once it's fully sent
        await utt.ended.wait()
        await asyncio.sleep(0.25)  # trailing silence, as a real STT endpoint would need
        stt.say(text)

    caller = ScriptedCaller(["Hi, I want to donate food.", "Rice and beans, three trays.", "Seventy five Lenox, three to five."])
    result = await asyncio.wait_for(
        run_scenario(scenario, services, caller, CallerTTS(), repo, CALL_DATE, on_utterance=hear), 60)

    assert result.error == "", result.error
    assert result.classified_as == "restaurant" and result.completed
    fields = {f["field"]: f["ok"] for f in result.fields}
    assert fields["items"] and fields["quantity"] and fields["address"] and fields["pickup_start"]
    assert len(result.client_v2v_ms) >= 2 and all(0 < v < 5000 for v in result.client_v2v_ms)
    assert result.barge_ins and result.barge_ins[0]["success"], result.barge_ins
    assert caller.heard[0].startswith("Hi, this is HungerCue")  # the caller heard the greeting first
    assert result.agent_cost_usd > 0
