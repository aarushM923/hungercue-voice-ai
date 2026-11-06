"""Agent loop tests with a scripted fake OpenAI client."""

import asyncio

from hungercue.db.repo import Repo
from hungercue.llm.agent import Agent, SentenceChunker, Usage
from hungercue.llm.tools import ToolContext
from hungercue.state.session_store import CallState
from tests.fakes import FakeOpenAI, text_chunks, tool_chunks, usage_chunk

DONATION = {
    "business_name": "Bagel Shop", "contact_name": None, "items": "bagels", "quantity": "3 bags",
    "servings_estimate": 40, "food_tags": ["Bread"], "pickup_date": "2026-11-04", "pickup_start": "15:00",
    "pickup_end": "17:00", "address": "200 W 112th St", "notes": None, "confirmed_by_caller": True,
}


def ctx_for(repo: Repo) -> ToolContext:
    return ToolContext(repo=repo, state=CallState(call_sid="CA1", from_number="+12125550100"))


def test_chunker_first_clause_then_sentences() -> None:
    c = SentenceChunker()
    out = c.push("Great, thanks so much for calling us, ") + c.push("we can help. When is pickup? ") + c.push("Ok")
    assert out == ["Great, thanks so much for calling us,", "we can help.", "When is pickup?"]
    assert c.flush() == ["Ok"] and c.flush() == []


def test_chunker_short_first_clause_waits() -> None:
    c = SentenceChunker()
    assert c.push("Hi, ") == []  # too short to send alone
    assert c.push("how are you? ") == ["Hi, how are you?"]


def test_usage_cost() -> None:
    u = Usage(input_tokens=1_000_000, cached_tokens=500_000, output_tokens=100_000)
    assert abs(u.cost_usd("gpt-4.1-mini") - (0.20 + 0.05 + 0.16)) < 1e-9


async def test_text_reply_streams_pieces(repo: Repo) -> None:
    fake = FakeOpenAI([text_chunks("Sure thing. What food do you have? ") + [usage_chunk(120, 15, 100)]])
    agent = Agent(fake, "gpt-4.1-mini", ctx_for(repo))
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "I want to donate"}]
    pieces = [p async for p in agent.respond(messages)]
    assert pieces == ["Sure thing.", "What food do you have?"]
    assert messages[-1] == {"role": "assistant", "content": "Sure thing. What food do you have?"}
    assert agent.usage.input_tokens == 120 and agent.usage.cached_tokens == 100
    assert agent.first_token_at is not None
    req = fake.requests[0]
    assert req["stream"] and req["model"] == "gpt-4.1-mini" and len(req["tools"]) == 6


async def test_tool_call_books_pickup_then_speaks(repo: Repo) -> None:
    booked: list[int] = []

    async def on_donation(pid: int) -> None:
        booked.append(pid)

    ctx = ctx_for(repo)
    ctx.on_donation = on_donation
    await repo.start_call("CA1", "+12125550100")
    fake = FakeOpenAI([tool_chunks("log_donation", DONATION), text_chunks("You're all set! ")])
    agent = Agent(fake, "gpt-4.1-mini", ctx)
    messages = [{"role": "user", "content": "yes that's right"}]
    pieces = [p async for p in agent.respond(messages)]
    # The model called the tool before saying anything, so a filler plays at once
    # instead of two LLM round trips of silence.
    assert pieces == ["Got it, one moment.", "You're all set!"]
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "log_donation"
    assert messages[1]["content"] == "Got it, one moment."  # history matches what was heard
    assert '"ok": true' in messages[2]["content"]
    p = await repo.get_pickup(booked[0])
    assert p.items == "bagels" and p.food_tags == ["bread"] and p.pickup_start.hour == 15
    assert p.restaurant.phone == "+12125550100" and ctx.state.outcome == "pickup_booked"
    assert (await repo.get_call("CA1")).outcome == "pickup_booked"
    # The second request must include the tool result so the model can respond to it.
    assert fake.requests[1]["messages"][-1]["role"] == "tool"


async def test_barge_in_records_partial_reply(repo: Repo) -> None:
    fake = FakeOpenAI([text_chunks("First sentence here. Second sentence that never plays. ")])
    agent = Agent(fake, "gpt-4.1-mini", ctx_for(repo))
    messages: list = [{"role": "user", "content": "hi"}]
    gen = agent.respond(messages)
    assert await gen.__anext__() == "First sentence here."
    await gen.aclose()  # what cancellation does to the generator
    assert messages[-1]["content"] == "First sentence here. [interrupted by caller]"


async def test_cancel_during_tool_still_finishes_write(repo: Repo) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    ctx = ctx_for(repo)

    async def slow_hook(pid: int) -> None:
        started.set()
        await release.wait()

    ctx.on_donation = slow_hook
    fake = FakeOpenAI([tool_chunks("log_donation", DONATION), text_chunks("Done. ")])
    agent = Agent(fake, "gpt-4.1-mini", ctx)
    messages: list = [{"role": "user", "content": "yes"}]

    async def consume() -> None:
        async for _ in agent.respond(messages):
            pass

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]  # history stays consistent
    assert ctx.state.pickup_id is not None


async def test_runaway_tool_loop_falls_back_to_transfer(repo: Repo) -> None:
    scripts = [tool_chunks("no_such_tool", {}, f"c{i}") for i in range(5)]
    ctx = ctx_for(repo)
    agent = Agent(FakeOpenAI(scripts), "gpt-4.1-mini", ctx)
    pieces = [p async for p in agent.respond([{"role": "user", "content": "hi"}])]
    assert "coordinator" in pieces[-1] and ctx.pending_action == "transferred"


async def test_no_filler_when_model_already_spoke(repo: Repo) -> None:
    from tests.fakes import FakeStream
    fake = FakeOpenAI([text_chunks("Booking that now. ") + tool_chunks("log_donation", DONATION), text_chunks("Done! ")])
    agent = Agent(fake, "gpt-4.1-mini", ctx_for(repo))
    pieces = [p async for p in agent.respond([{"role": "user", "content": "yes"}])]
    assert pieces == ["Booking that now.", "Done!"]


async def test_ending_action_speaks_closing_line_in_callers_language(repo: Repo) -> None:
    ctx = ctx_for(repo)
    ctx.state.language = "es"
    fake = FakeOpenAI([tool_chunks("end_call", {"reason": "done"}), text_chunks("never used ")])
    agent = Agent(fake, "gpt-4.1-mini", ctx)
    pieces = [p async for p in agent.respond([{"role": "user", "content": "eso es todo"}])]
    assert pieces == ["Gracias por llamar a HungerCue. ¡Adiós!"]
    assert len(fake.requests) == 1 and ctx.pending_action == "end_call"  # no second LLM round
