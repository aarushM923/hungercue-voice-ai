"""Who is calling? Classified in the background, off the reply's critical path.

The first version had the agent call a set_caller_type tool before replying.
That cost a whole extra LLM round trip (~0.5-1 s) on the first reply of every
call, in the time the caller is waiting. Classification is bookkeeping, so now
a small separate LLM call runs *in parallel* with the reply after each caller
turn, until the type is known. The tools that save data (log_donation,
update_shelter, update_volunteer) also set the type definitively.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

TYPES = ("restaurant", "shelter", "volunteer")
# Returns (caller type or "", token usage) so the call can count the classifier in its cost.
Classifier = Callable[[list[str]], Awaitable[tuple[str, Any]]]

PROMPT = (
    "Classify the caller of a food-rescue phone line from what they said. "
    "restaurant = a business donating surplus food; shelter = a shelter or pantry that receives food; "
    "volunteer = a person offering to pick up or deliver food. "
    "Answer unknown if it isn't clear yet."
)
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "caller_type",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"caller_type": {"type": "string", "enum": [*TYPES, "unknown"]}},
            "required": ["caller_type"],
            "additionalProperties": False,
        },
    },
}


def make_classifier(client: Any, model: str) -> Classifier:
    """Build a classifier function around an OpenAI client."""

    async def classify(caller_lines: list[str]) -> tuple[str, Any]:
        text = "\n".join(f"Caller: {line}" for line in caller_lines[-4:])
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": text}],
            response_format=SCHEMA,  # structured output: always valid JSON with one of the four values
            temperature=0,
            max_completion_tokens=20,
        )
        try:
            value = json.loads(r.choices[0].message.content or "{}").get("caller_type", "")
        except json.JSONDecodeError:
            value = ""
        return (value if value in TYPES else ""), r.usage

    return classify
