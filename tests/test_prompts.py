from datetime import datetime

from hungercue.db.repo import CallerProfile
from hungercue.llm.prompts import build_system_prompt, greeting, short_street


def test_system_prompt_includes_date_and_caller() -> None:
    p = build_system_prompt(datetime(2026, 11, 3, 14, 5), CallerProfile("restaurant", 1, "Bagel Shop", "", "Restaurant 'Bagel Shop'."), "es", "+1212")
    assert "2026-11-03 14:05 (Tuesday)" in p and "RETURNING RESTAURANT" in p and "Spanish" in p
    assert "Unknown caller" in build_system_prompt(datetime(2026, 1, 1), CallerProfile(""), "en", "")


def test_greetings() -> None:
    bagel = CallerProfile("restaurant", 1, "the bagel shop", "200 W 112th St, New York")
    assert greeting(bagel) == "Hi, this is HungerCue. Is this another pickup from the bagel shop on 112th?"
    assert "donate food" in greeting(CallerProfile(""))
    assert greeting(CallerProfile("volunteer", 2, "Ana")).startswith("Hi Ana")
    assert greeting(CallerProfile(""), "es").startswith("Hola")
    assert "the bagel shop" in greeting(bagel, "es")
    assert short_street("Main Street") == "Main Street"
