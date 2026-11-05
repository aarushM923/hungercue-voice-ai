"""The simulated caller: an LLM role-playing a restaurant, shelter or volunteer.

It receives the scenario's facts (the ground truth) and talks like a busy
person on the phone. To keep the test honest it is told to reveal details
only when asked, not to dump everything in one breath (that would make the
agent's job artificially easy).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from evals.scenarios import Scenario

HANGUP = "<hangup>"
INTERJECTIONS = {
    "en": "Sorry, sorry, hold on one second.",
    "es": "Perdón, perdón, espere un segundo.",
}
CALLER_MODEL_PRICES = (0.40, 1.60)  # gpt-4.1-mini USD per 1M input / output tokens


def persona_prompt(s: Scenario, today: datetime) -> str:
    t, said = s.truth, s.said
    lang = "Spanish (you speak only Spanish)" if s.language == "es" else "English"
    lines = [
        f"You are {s.name}, calling HungerCue, a food rescue service in New York, on the phone. Speak {lang}.",
        f"Today is {today:%A %B %d}.",
    ]
    if s.caller_type == "restaurant":
        lines += [
            f"You work at the restaurant {t['business_name']} and want to donate surplus food for pickup.",
            f"Food: {said['items']}. Amount: {said['quantity']}. Pickup {said['day']}, window {said['window']}.",
            f"Pickup address: {said['address']}.",
        ]
        if said.get("note"):
            lines.append(said["note"])
    elif s.caller_type == "shelter":
        lines += [
            f"You run intake at the shelter {t['shelter_name']}, {said['address']}, and want to update what you can receive.",
            f"Today you can take about {t['capacity']} servings. You need: {', '.join(t['needs'])}.",
            "Dietary rules: " + (", ".join(t["dietary_restrictions"]) if t["dietary_restrictions"] else "none") + ".",
            f"You receive deliveries {said['hours']}.",
        ]
    else:
        lines += [
            "You want to sign up as a volunteer driver/picker-upper for food pickups.",
            f"You're available {said['availability']}.",
            f"Neighborhoods you can cover: {', '.join(t['neighborhoods'])}.",
            "You " + ("have a car." if t["has_car"] else "do NOT have a car (you take the subway)."),
        ]
    lines += [
        "",
        "How to talk:",
        "- This is a phone call. Short, natural sentences, like a busy real person. No lists, no markdown.",
        "- Start by saying why you're calling in one sentence. Then give details only when asked, one or two at a time.",
        "- If the agent reads details back and they're right, say yes. If anything is wrong, correct it clearly.",
        "- Never invent facts beyond the ones above. If asked something you don't know, say you're not sure.",
    ]
    if s.conditions.topic_change:
        lines.append("- Early in the call, briefly ask an unrelated question (e.g. whether HungerCue also takes clothing donations), then get back on track.")
    lines.append(f"- When the agent has everything and says goodbye (or you're clearly done), reply with exactly {HANGUP}")
    return "\n".join(lines)


class CallerBrain(Protocol):
    async def reply(self, agent_text: str) -> str: ...
    def note_said(self, text: str) -> None: ...
    @property
    def cost_usd(self) -> float: ...


class LLMCaller:
    def __init__(self, client: Any, model: str, scenario: Scenario, today: datetime) -> None:
        self.client = client
        self.model = model
        # Roles are flipped: from the caller-LLM's point of view, the agent is the "user".
        self.messages: list[dict[str, str]] = [{"role": "system", "content": persona_prompt(scenario, today)}]
        self.in_tokens = 0
        self.out_tokens = 0

    async def reply(self, agent_text: str) -> str:
        self.messages.append({"role": "user", "content": agent_text or "(silence)"})
        r = await self.client.chat.completions.create(model=self.model, messages=self.messages, temperature=0.7)
        if r.usage:
            self.in_tokens += r.usage.prompt_tokens
            self.out_tokens += r.usage.completion_tokens
        text = (r.choices[0].message.content or "").strip()
        self.messages.append({"role": "assistant", "content": text})
        return HANGUP if HANGUP in text else text

    def note_said(self, text: str) -> None:
        """Record a line the caller said outside reply() (the scripted interruption)."""
        self.messages.append({"role": "assistant", "content": text})

    @property
    def cost_usd(self) -> float:
        p_in, p_out = CALLER_MODEL_PRICES
        return (self.in_tokens * p_in + self.out_tokens * p_out) / 1_000_000


class ScriptedCaller:
    """Fixed lines, for offline tests of the harness itself."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = list(lines)
        self.heard: list[str] = []

    async def reply(self, agent_text: str) -> str:
        self.heard.append(agent_text)
        return self.lines.pop(0) if self.lines else HANGUP

    def note_said(self, text: str) -> None:
        pass

    @property
    def cost_usd(self) -> float:
        return 0.0
