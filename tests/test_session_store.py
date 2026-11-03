from hungercue.state.session_store import CallState, SessionStore, make_redis


async def test_save_load_delete() -> None:
    store = SessionStore(make_redis(""))
    assert await store.load("CA1") is None
    st = CallState(call_sid="CA1", from_number="+1", language="es")
    st.messages.append({"role": "user", "content": "hola"})
    await store.save(st)
    loaded = await store.load("CA1")
    assert loaded == st
    assert 0 < await store.r.ttl("call:CA1") <= 4 * 3600  # expiry is set
    await store.delete("CA1")
    assert await store.load("CA1") is None
