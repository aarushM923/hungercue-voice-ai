"""Tests for mu-law encoding/decoding."""

import numpy as np

from hungercue.audio.mulaw import mulaw_to_pcm16, pcm16_to_mulaw, silence_mulaw


def test_decode_known_values() -> None:
    # 0xFF and 0x7F are the two encodings of zero; 0x00 is the most negative value.
    out = mulaw_to_pcm16(bytes([0xFF, 0x7F, 0x00, 0x80]))
    assert out.tolist() == [0, 0, -32124, 32124]


def test_every_byte_round_trips() -> None:
    # Decoding then re-encoding any mu-law byte must give the same byte back
    # (except 0x7F, "negative zero", which re-encodes as positive zero 0xFF).
    all_bytes = bytes(range(256))
    again = pcm16_to_mulaw(mulaw_to_pcm16(all_bytes))
    diffs = [b for b, a in zip(all_bytes, again) if b != a]
    assert diffs == [0x7F]


def test_pcm_round_trip_error_is_small() -> None:
    # Mu-law is lossy, but error should be a few percent of the signal at most.
    t = np.arange(8000) / 8000
    pcm = (10000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    back = mulaw_to_pcm16(pcm16_to_mulaw(pcm)).astype(np.int32)
    assert np.max(np.abs(back - pcm)) < 0.04 * 10000 + 16


def test_extremes_do_not_overflow() -> None:
    pcm = np.array([-32768, 32767], dtype=np.int16)
    back = mulaw_to_pcm16(pcm16_to_mulaw(pcm))
    assert back[0] < -30000 and back[1] > 30000


def test_silence_length() -> None:
    assert silence_mulaw(20) == b"\xff" * 160  # 20 ms at 8 kHz = 160 samples
