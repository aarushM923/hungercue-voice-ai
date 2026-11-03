import wave
from pathlib import Path

from hungercue.audio.recorder import CallRecorder

LOUD = bytes([0x80]) * 160  # constant non-zero mu-law
SIL = b"\xff" * 160


def test_agent_audio_is_placed_at_playback_time(tmp_path: Path) -> None:
    r = CallRecorder()
    r.add_inbound(SIL * 5)  # 100 ms of caller timeline
    r.add_outbound(LOUD)  # arrives now -> plays at sample 800
    r.add_outbound(LOUD)  # queued right after the first chunk
    st = r.to_stereo()
    assert st[:800, 1].max() == 0 and st[800:1120, 1].min() > 0
    assert r.duration_s == 1120 / 8000
    with wave.open(str(r.save(tmp_path / "x" / "c.wav"))) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (2, 8000, 1120)


def test_clear_drops_unplayed_agent_audio() -> None:
    r = CallRecorder()
    r.add_outbound(LOUD * 10)  # 200 ms queued from t=0
    r.add_inbound(SIL * 5)  # only 100 ms has played when the caller barges in
    r.clear()
    st = r.to_stereo()
    assert len(st) == 800 and st[:, 1].min() > 0  # exactly the part that played
    r.add_outbound(LOUD)
    assert r.outbound[-1][0] == 800  # new audio starts at "now"


def test_empty_recorder() -> None:
    assert CallRecorder().to_stereo().shape == (0, 2)
