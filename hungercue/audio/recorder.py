"""Records a call as a stereo WAV: caller on the left channel, agent on the right.

Caller audio arrives in real time (one 20 ms frame every 20 ms), but agent
audio arrives from TTS in bursts much faster than real time and is queued by
Twilio for playback. To line the two up, we place each agent chunk at the
moment it will actually *play*: right after the previous agent audio, but
never earlier than "now" on the caller timeline.

On barge-in we send Twilio a `clear`, so queued agent audio never plays;
`clear()` here drops it from the recording too, so the recording matches what
the caller really heard.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from hungercue.audio.mulaw import SAMPLE_RATE, mulaw_to_pcm16


class CallRecorder:
    def __init__(self) -> None:
        self.inbound: list[np.ndarray] = []
        self.in_samples = 0  # caller timeline position (samples)
        self.outbound: list[tuple[int, np.ndarray]] = []  # (start sample, audio)
        self.out_cursor = 0  # where the next agent chunk will start playing

    def add_inbound(self, mulaw: bytes) -> None:
        pcm = mulaw_to_pcm16(mulaw)
        self.inbound.append(pcm)
        self.in_samples += len(pcm)

    def add_outbound(self, mulaw: bytes) -> None:
        pcm = mulaw_to_pcm16(mulaw)
        start = max(self.out_cursor, self.in_samples)
        self.outbound.append((start, pcm))
        self.out_cursor = start + len(pcm)

    def clear(self) -> None:
        """Barge-in: drop agent audio that was queued but hasn't played yet."""
        now = self.in_samples
        kept: list[tuple[int, np.ndarray]] = []
        for start, pcm in self.outbound:
            if start + len(pcm) <= now:
                kept.append((start, pcm))
            elif start < now:
                kept.append((start, pcm[: now - start]))  # the part that already played
        self.outbound = kept
        self.out_cursor = now

    def to_stereo(self) -> np.ndarray:
        caller = np.concatenate(self.inbound) if self.inbound else np.zeros(0, dtype=np.int16)
        length = max(len(caller), self.out_cursor)
        stereo = np.zeros((length, 2), dtype=np.int16)
        stereo[: len(caller), 0] = caller
        for start, pcm in self.outbound:
            stereo[start : start + len(pcm), 1] = pcm
        return stereo

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(self.to_stereo().tobytes())  # numpy interleaves L/R per row
        return path

    @property
    def duration_s(self) -> float:
        return max(self.in_samples, self.out_cursor) / SAMPLE_RATE
