"""Background caller classification."""

from types import SimpleNamespace as NS

from hungercue.llm.classify import make_classifier


class FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.kwargs: dict = {}
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **kwargs):
        self.kwargs = kwargs
        return NS(choices=[NS(message=NS(content=self.content))], usage=NS(prompt_tokens=80, completion_tokens=5))


async def test_returns_type_and_usage() -> None:
    client = FakeClient('{"caller_type": "shelter"}')
    guess, usage = await make_classifier(client, "gpt-4.1-mini")(["hi", "I run intake at Hope House"])
    assert guess == "shelter" and usage.prompt_tokens == 80
    assert client.kwargs["response_format"]["json_schema"]["strict"] is True
    assert "Caller: I run intake at Hope House" in client.kwargs["messages"][1]["content"]


async def test_unknown_and_garbage_become_empty() -> None:
    assert (await make_classifier(FakeClient('{"caller_type": "unknown"}'), "m")(["hello"]))[0] == ""
    assert (await make_classifier(FakeClient("not json"), "m")(["hello"]))[0] == ""
