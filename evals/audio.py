"""Making test audio realistic: background noise, mumbling, phone framing.

All synthetic (numpy), so nothing to download and every run is reproducible:
- kitchen: broadband hiss plus random clatter (short, loud bursts)
- street:  low rumble (traffic) plus occasional horn-like tones
- cafe:    "babble", several overlapping slowly-modulated noise bands, the
           texture of many people talking at once
Noise is scaled to a target signal-to-noise ratio (SNR) in decibels:
SNR 20 dB = speech is 10x louder (in amplitude) than the noise; 10 dB = ~3x.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from hungercue.audio.mulaw import SAMPLE_RATE, mulaw_to_pcm16, pcm16_to_mulaw

FRAME = 160  # 20 ms at 8 kHz: one Twilio media frame


def _lowpass(x: np.ndarray, alpha: float) -> np.ndarray:
    """One-pole low-pass filter: y[i] = y[i-1] + alpha * (x[i] - y[i-1]).

    Smaller alpha = more muffled. lfilter runs this recurrence in C; written as
    a filter it's numerator [alpha] over denominator [1, alpha - 1].
    """
    return lfilter([alpha], [1.0, alpha - 1.0], x)


def make_noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` samples of float noise (roughly unit loudness) of the given kind."""
    t = np.arange(n) / SAMPLE_RATE
    if kind == "kitchen":
        x = rng.normal(0, 1, n) * 0.6
        for _ in range(max(1, n // SAMPLE_RATE)):  # ~1 clatter per second
            at, length = rng.integers(0, max(1, n - 400)), int(rng.integers(80, 400))
            x[at:at + length] += rng.normal(0, 3, length) * np.exp(-np.arange(length) / 60)
    elif kind == "street":
        x = _lowpass(rng.normal(0, 1, n), 0.02) * 6
        if n > SAMPLE_RATE:
            at = int(rng.integers(0, n - SAMPLE_RATE // 2))
            x[at:at + SAMPLE_RATE // 2] += 0.8 * np.sin(2 * np.pi * 420 * t[: SAMPLE_RATE // 2])
    elif kind == "cafe":
        x = np.zeros(n)
        for _ in range(6):  # six "voices" of shaped noise, each fading in and out
            band = _lowpass(rng.normal(0, 1, n), float(rng.uniform(0.05, 0.25)))
            env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.3, 1.5) * t + rng.uniform(0, 6))
            x += band * env
    else:
        return np.zeros(n)
    return x / (np.sqrt(np.mean(x**2)) + 1e-9)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def degrade(pcm: np.ndarray, noise: str, snr_db: float, mumble: bool, rng: np.random.Generator) -> np.ndarray:
    """Apply the scenario's conditions to clean caller speech (int16 in, int16 out)."""
    x = pcm.astype(np.float64)
    if mumble:
        x = _lowpass(x, 0.35) * 0.35  # muffled and about 9 dB quieter
    if noise != "none" and len(x):
        speech_rms = rms(x) or 1.0
        x = x + make_noise(noise, len(x), rng) * speech_rms / (10 ** (snr_db / 20))
    return np.clip(x, -32768, 32767).astype(np.int16)


def background_frame(noise: str, snr_db: float, rng: np.random.Generator, speech_level: float = 3000.0) -> bytes:
    """20 ms of room tone between utterances (callers are never in perfect silence)."""
    if noise == "none":
        x = rng.normal(0, 30, FRAME)  # faint line hiss
    else:
        x = make_noise(noise, FRAME, rng) * speech_level / (10 ** (snr_db / 20))
    return pcm16_to_mulaw(np.clip(x, -32768, 32767).astype(np.int16))


def mulaw_frames(mulaw: bytes) -> list[bytes]:
    """Split audio into Twilio-sized frames, padding the last with silence."""
    frames = [mulaw[i:i + FRAME] for i in range(0, len(mulaw), FRAME)]
    if frames and len(frames[-1]) < FRAME:
        frames[-1] = frames[-1] + b"\xff" * (FRAME - len(frames[-1]))
    return frames


def to_pcm(mulaw: bytes) -> np.ndarray:
    return mulaw_to_pcm16(mulaw)
