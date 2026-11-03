"""Reusable fakes for tests (and the offline parts of the eval harness)."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS
from typing import Any


def text_chunks(text: str, size: int = 7) -> list[Any]:
    """Split text into streaming chunks like the OpenAI API sends."""
    parts = [text[i : i + size] for i in range(0, len(text), size)]
    return [NS(usage=None, choices=[NS(delta=NS(content=p, tool_calls=None))]) for p in parts]


def tool_chunks(name: str, args: dict[str, Any], call_id: str = "call_1") -> list[Any]:
    """A tool call whose JSON arguments arrive in 3 fragments, as in real streams."""
    raw = json.dumps(args)
    third = len(raw) // 3 + 1
    frags = [raw[i : i + third] for i in range(0, len(raw), third)]
    out = [NS(usage=None, choices=[NS(delta=NS(content=None, tool_calls=[
        NS(index=0, id=call_id, function=NS(name=name, arguments=frags[0]))]))])]
    for f in frags[1:]:
        out.append(NS(usage=None, choices=[NS(delta=NS(content=None, tool_calls=[
            NS(index=0, id=None, function=NS(name=None, arguments=f))]))]))
    return out


def usage_chunk(prompt: int = 100, completion: int = 20, cached: int = 0) -> Any:
    return NS(usage=NS(prompt_tokens=prompt, completion_tokens=completion,
                       prompt_tokens_details=NS(cached_tokens=cached)), choices=[])


class FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self.chunks:
                yield c
        return gen()


class FakeOpenAI:
    """Replays one scripted list of chunks per create() call and records requests."""

    def __init__(self, scripts: list[list[Any]]) -> None:
        self.scripts = list(scripts)
        self.requests: list[dict[str, Any]] = []
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **kwargs: Any) -> FakeStream:
        # Snapshot messages: the agent mutates the list after the call.
        self.requests.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        return FakeStream(self.scripts.pop(0))
