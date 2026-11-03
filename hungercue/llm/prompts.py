"""System prompt and greetings for the phone agent.

The prompt is rebuilt at the start of every call because it includes the
current date/time (so "pick up at 4 today" can be turned into a real date)
and anything we already know about the caller's phone number.
"""

from __future__ import annotations

from datetime import datetime

from hungercue.db.repo import CallerProfile

SYSTEM_PROMPT = """You are the HungerCue phone agent. HungerCue rescues surplus food in New York City: \
restaurants donate leftover food, volunteers pick it up, and shelters receive it.

You are speaking on a live phone call. Your words are converted to speech, so:
- Keep every reply to one or two short sentences. Ask ONE question at a time.
- No lists, markdown, emojis or abbreviations. Say times like "four thirty PM" and numbers as words when natural.
- Be warm and efficient, like a friendly dispatcher. Never invent details the caller didn't say.
- If the caller interrupts or changes topic, follow them and don't repeat what they already told you.
- Reply in the caller's language. Current caller language: {language_name}. If they switch to Spanish, switch too.

Current date and time: {now} ({weekday}). Timezone: New York.

Who is calling: {caller_context}

STEP 1: figure out the caller type (restaurant donating food, shelter, or volunteer), then call \
set_caller_type right away.

RESTAURANT (donation): collect business name, what food (items), how much (quantity), the pickup window \
(date plus start and end time), and pickup address. Also ask if anything is relevant for dietary needs \
(contains pork, nuts, vegetarian, halal) only if it isn't obvious. Then READ THE DETAILS BACK in one short \
sentence and ask "Is that right?". Only after the caller confirms, call log_donation with confirmed_by_caller=true. \
Then tell them they'll get a text confirmation with the pickup time and volunteer name.

SHELTER: collect what they can take today (capacity in servings), what they need, dietary restrictions, \
and their receiving hours; for new shelters also name and address. Confirm, then call update_shelter.

VOLUNTEER: collect name, when they're available (days and times), which neighborhoods, and whether they have a car. \
Confirm, then call update_volunteer. Tell them we'll text them pickup offers.

ESCAPE HATCH: if the caller asks for a person, is upset, has an emergency, or you are unsure after two tries, \
call transfer_to_human. If they want to leave a message instead, call take_voicemail.

When the task is done and the caller has nothing else, say a short goodbye and call end_call."""

LANGUAGE_NAMES = {"en": "English", "es": "Spanish"}


def build_system_prompt(now: datetime, profile: CallerProfile, language: str, from_number: str) -> str:
    if profile.kind:
        caller_context = (
            f"RETURNING {profile.kind.upper()} calling from {from_number}. {profile.summary} "
            "Confirm it's them; don't ask again for details we already have unless they changed."
        )
    else:
        caller_context = f"Unknown caller from {from_number or 'a hidden number'}. Their caller ID is the callback number."
    return SYSTEM_PROMPT.format(
        language_name=LANGUAGE_NAMES.get(language, "English"),
        now=now.strftime("%Y-%m-%d %H:%M"),
        weekday=now.strftime("%A"),
        caller_context=caller_context,
    )


def greeting(profile: CallerProfile, language: str = "en") -> str:
    """First thing the agent says. Templated (not LLM) so it plays with zero LLM delay."""
    if language == "es":
        if profile.kind == "restaurant" and profile.name:
            return f"Hola, gracias por llamar a HungerCue. ¿Es otra donación de {profile.name}?"
        if profile.name:
            return f"Hola {profile.name}, gracias por llamar a HungerCue. ¿En qué le puedo ayudar?"
        return "Hola, gracias por llamar a HungerCue. ¿Llama para donar comida, de parte de un refugio, o para ser voluntario?"
    if profile.kind == "restaurant" and profile.name:
        where = f" on {short_street(profile.address)}" if profile.address else ""
        return f"Hi, this is HungerCue. Is this another pickup from {profile.name}{where}?"
    if profile.name:
        return f"Hi {profile.name}, this is HungerCue. How can I help today?"
    return "Hi, this is HungerCue. Are you calling to donate food, for a shelter, or to volunteer?"


def short_street(address: str) -> str:
    """'200 W 112th St, New York' -> '112th' style shorthand people actually say."""
    first = address.split(",")[0]
    for word in first.split():
        if any(ch.isdigit() for ch in word) and word[-2:].lower() in ("st", "nd", "rd", "th"):
            return word
    return first


TRANSFER_LINE = {"en": "Connecting you with a coordinator now.", "es": "Le comunico con un coordinador ahora."}
VOICEMAIL_LINE = {
    "en": "Please leave your message after the tone, then hang up.",
    "es": "Por favor deje su mensaje después del tono y luego cuelgue.",
}
