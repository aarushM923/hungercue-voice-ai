"""VAD tests: the real Silero model on synthetic speech, plus segmenter logic."""

import wave
from pathlib import Path

import numpy as np

from hungercue.audio.mulaw import pcm16_to_mulaw
from hungercue.audio.vad import CHUNK, SileroVAD, VadSegmenter

FIXTURE = Path(__file__).parent / "fixtures" / "speech_8k.wav"


def load_pcm() -> np.ndarray:
    with wave.open(str(FIXTURE)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def feed_in_twilio_frames(seg: VadSegmenter, pcm: np.ndarray) -> list:
    """Push audio exactly as Twilio delivers it: 160-byte mu-law frames."""
    mulaw = pcm16_to_mulaw(pcm)
    events = []
    for i in range(0, len(mulaw), 160):
        events += seg.push_mulaw(mulaw[i : i + 160])
    return events


def test_model_silence_is_low_probability() -> None:
    vad = SileroVAD()
    probs = [vad(np.zeros(CHUNK, dtype=np.float32)) for _ in range(10)]
    assert max(probs) < 0.2


def test_detects_one_utterance_in_real_speech() -> None:
    speech = load_pcm()
    silence = np.zeros(8000, dtype=np.int16)  # 1 s
    audio = np.concatenate([silence, speech, silence])
    events = feed_in_twilio_frames(VadSegmenter(SileroVAD()), audio)
    kinds = [e.kind for e in events]
    assert kinds == ["speech_start", "speech_end"], kinds
    start, end = events[0].stream_ms, events[1].stream_ms
    # Speech begins after the 1 s of leading silence and ends before the trailing silence.
    assert 900 <= start <= 1400
    assert 1000 + 2000 <= end <= 1000 + 3300


def test_segmenter_hysteresis_with_fake_probabilities() -> None:
    probs = iter([0.9, 0.9, 0.9, 0.4, 0.9] + [0.1] * 10)
    seg = VadSegmenter(lambda _chunk: next(probs), min_speech_ms=96, min_silence_ms=224)
    events = seg.push_float(np.zeros(CHUNK * 15, dtype=np.float32))
    # 0.4 is below start but above end threshold: it must NOT end the utterance.
    assert [e.kind for e in events] == ["speech_start", "speech_end"]
    assert events[0].stream_ms == 0  # start is back-dated to the first voiced chunk
    assert events[1].stream_ms == 5 * 32  # end = last voiced chunk, not detection time


def test_short_noise_is_ignored() -> None:
    probs = iter([0.9, 0.1, 0.9, 0.1, 0.1, 0.1])
    seg = VadSegmenter(lambda _c: next(probs), min_speech_ms=96)
    assert seg.push_float(np.zeros(CHUNK * 6, dtype=np.float32)) == []
