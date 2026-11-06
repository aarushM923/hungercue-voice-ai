"""The conversational agent: streaming LLM replies with tool calls.

Flow for one caller turn:
  1. Send the conversation so far to GPT-4.1-mini with stream=True.
  2. As text tokens arrive, a SentenceChunker groups them into speakable
     pieces; each piece is yielded immediately so TTS can start talking
     before the model has finished writing.
  3. If the model asks for tools instead, run them, append the results to
     the conversation, and ask the model again (it then tells the caller).

Barge-in: if the caller interrupts, the orchestrator cancels the task that
consumes this generator. The `finally` block records the partial reply so the
model knows what the caller actually heard.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from hungercue.llm.tools import TOOL_SCHEMAS, ToolContext, run_tool

MAX_TOOL_ROUNDS = 4  # safety net against a model that keeps calling tools forever

# When the model calls a tool BEFORE saying anything, the caller would otherwise sit
# in silence for two LLM round trips (call the tool, then write the reply). The
# first eval showed those turns were the entire slow tail (1.0-1.9 s). Like a human
# dispatcher, we say a short line immediately while the work happens.
SAVING_TOOLS = {"log_donation", "update_shelter", "update_volunteer"}
FILLER = {"en": "Got it, one moment.", "es": "Entendido, un momento."}
# Ending actions need no second round at all: a fixed closing line says it all.
CLOSING = {
    "end_call": {"en": "Thanks for calling HungerCue. Goodbye!", "es": "Gracias por llamar a HungerCue. ¡Adiós!"},
    "transfer_to_human": {"en": "Connecting you with a coordinator now.", "es": "Le comunico con un coordinador ahora."},
    "take_voicemail": {"en": "Sure.", "es": "Claro."},
}

# USD per 1M tokens for gpt-4.1-mini (input, cached input, output). Used for cost-per-call.
PRICES = {"gpt-4.1-mini": (0.40, 0.10, 1.60)}


class ChatClient(Protocol):
    """The slice of AsyncOpenAI we use (lets tests pass a scripted fake)."""

    @property
    def chat(self) -> Any: ...


@dataclass
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0

    def cost_usd(self, model: str) -> float:
        p_in, p_cached, p_out = PRICES.get(model, PRICES["gpt-4.1-mini"])
        fresh = self.input_tokens - self.cached_tokens
        return (fresh * p_in + self.cached_tokens * p_cached + self.output_tokens * p_out) / 1_000_000


class SentenceChunker:
    """Split streaming text into pieces worth sending to TTS.

    Rules: cut after sentence-ending punctuation; for the FIRST piece also cut
    at a comma once there are a few words, because getting any audio out fast
    matters most for perceived latency. Later pieces wait for full sentences,
    which sound more natural.
    """

    SENTENCE_END = re.compile(r"(?<=[.!?¿¡])\s+")  # punctuation followed by whitespace
    FIRST_CLAUSE = re.compile(r"(?<=[,;:])\s+")

    def __init__(self, first_min_chars: int = 18) -> None:
        self.buffer = ""
        self.first_done = False
        self.first_min_chars = first_min_chars

    def push(self, text: str) -> list[str]:
        self.buffer += text
        out: list[str] = []
        while True:
            m = self.SENTENCE_END.search(self.buffer)
            if not self.first_done:
                # First comma that has at least `first_min_chars` of text before it.
                c = next((x for x in self.FIRST_CLAUSE.finditer(self.buffer) if x.start() >= self.first_min_chars), None)
                if c and (m is None or c.start() < m.start()):
                    m = c
            if m is None:
                return out
            piece, self.buffer = self.buffer[: m.start()].strip(), self.buffer[m.end() :]
            if piece:
                out.append(piece)
                self.first_done = True

    def flush(self) -> list[str]:
        piece, self.buffer = self.buffer.strip(), ""
        return [piece] if piece else []


class Agent:
    def __init__(self, client: ChatClient, model: str, ctx: ToolContext, temperature: float = 0.3) -> None:
        self.client = client
        self.model = model
        self.ctx = ctx
        self.temperature = temperature
        self.usage = Usage()
        self.first_token_at: float | None = None  # loop time of the first text token this turn

    async def respond(self, messages: list[dict[str, Any]]) -> AsyncIterator[str]:
        """Yield speakable text pieces for the next agent turn. Mutates `messages`."""
        self.first_token_at = None
        loop = asyncio.get_running_loop()
        said_anything = False  # across rounds of this one reply
        for _round in range(MAX_TOOL_ROUNDS + 1):
            chunker = SentenceChunker()
            spoken: list[str] = []
            tool_calls: dict[int, dict[str, str]] = {}
            completed = False
            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    temperature=self.temperature,
                    stream=True,
                    stream_options={"include_usage": True},  # final chunk reports token counts
                    parallel_tool_calls=False,  # one tool at a time keeps conversation order simple
                )
                async for chunk in stream:
                    if chunk.usage:
                        self._add_usage(chunk.usage)
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        if self.first_token_at is None:
                            self.first_token_at = loop.time()
                        for piece in chunker.push(delta.content):
                            spoken.append(piece)
                            yield piece
                    # Tool call arguments stream in fragments; stitch them by index.
                    for tc in delta.tool_calls or []:
                        slot = tool_calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
                for piece in chunker.flush():
                    spoken.append(piece)
                    yield piece
                completed = True
            finally:
                # Runs on normal completion AND on cancellation (barge-in).
                text = " ".join(spoken)
                if not completed:
                    text = (text + " [interrupted by caller]").strip()
                if text or tool_calls:
                    msg: dict[str, Any] = {"role": "assistant", "content": text or None}
                    if tool_calls and completed:
                        msg["tool_calls"] = [
                            {"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": t["arguments"]}}
                            for t in tool_calls.values()
                        ]
                    messages.append(msg)

            said_anything = said_anything or bool(spoken)
            if not tool_calls:
                return
            names = [t["name"] for t in tool_calls.values()]
            lang = self.ctx.state.language if self.ctx.state.language in FILLER else "en"
            ending = next((n for n in names if n in CLOSING), None)
            extra = ""
            if not said_anything and ending:
                extra = CLOSING[ending][lang]
            elif not said_anything and SAVING_TOOLS & set(names):
                extra = FILLER[lang]
            if extra:
                if self.first_token_at is None:
                    self.first_token_at = loop.time()  # the moment we had something to say
                # Record it on the assistant message that holds the tool call, so the
                # history matches what the caller heard.
                messages[-1]["content"] = extra
                said_anything = True
                yield extra
            await self._run_tools(messages, list(tool_calls.values()))
            if ending:
                return  # goodbye / transfer / voicemail: nothing more for the model to say
        # Too many tool rounds: hand back something safe to say.
        yield "Sorry, let me connect you with a coordinator."
        self.ctx.pending_action = "transferred"

    async def _run_tools(self, messages: list[dict[str, Any]], calls: list[dict[str, str]]) -> None:
        """Run tools and append their results.

        Shielded from cancellation: if the caller barges in while a pickup is
        being written, we still finish the write and record the result, so the
        database and the conversation history never disagree.
        """

        async def work() -> list[dict[str, Any]]:
            results = []
            for c in calls:
                content = await run_tool(self.ctx, c["name"], c["arguments"])
                results.append({"role": "tool", "tool_call_id": c["id"], "content": content})
            return results

        task = asyncio.ensure_future(work())
        try:
            results = await asyncio.shield(task)
        except asyncio.CancelledError:
            messages.extend(await task)  # finish the in-flight write, then let the cancel proceed
            raise
        messages.extend(results)

    def _add_usage(self, usage: Any) -> None:
        self.usage.input_tokens += usage.prompt_tokens or 0
        self.usage.output_tokens += usage.completion_tokens or 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None and getattr(details, "cached_tokens", None):
            self.usage.cached_tokens += details.cached_tokens
