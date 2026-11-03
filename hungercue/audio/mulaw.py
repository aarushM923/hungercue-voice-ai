"""G.711 mu-law <-> 16-bit PCM conversion.

Phone networks send audio as 8-bit "mu-law" samples at 8,000 samples per second.
Mu-law is a logarithmic compression: quiet sounds get more precision than loud
ones, which matches how human hearing works and squeezes 14-bit audio into 8 bits.

Twilio Media Streams send and expect base64-encoded mu-law. Our VAD and audio
files want linear 16-bit PCM, so we convert in both directions here. We use
numpy (vectorised, no Python loops) instead of the stdlib `audioop` module,
which was deprecated and removed in Python 3.13.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 8000
_BIAS = 0x84  # 132: added before encoding so the logarithm segments line up
_CLIP = 32635  # largest magnitude that survives the bias without overflow


def _build_decode_table() -> np.ndarray:
    """Precompute the PCM value for each of the 256 possible mu-law bytes."""
    table = np.zeros(256, dtype=np.int16)
    for byte in range(256):
        u = ~byte & 0xFF  # mu-law bytes are stored bit-inverted
        sign = u & 0x80
        exponent = (u >> 4) & 0x07  # which logarithmic "segment"
        mantissa = u & 0x0F  # position inside the segment
        magnitude = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
        table[byte] = -magnitude if sign else magnitude
    return table


_DECODE_TABLE = _build_decode_table()


def mulaw_to_pcm16(data: bytes) -> np.ndarray:
    """Decode mu-law bytes into an int16 numpy array (one value per sample)."""
    return _DECODE_TABLE[np.frombuffer(data, dtype=np.uint8)]


def pcm16_to_mulaw(samples: np.ndarray) -> bytes:
    """Encode int16 PCM samples into mu-law bytes."""
    pcm = samples.astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0)
    magnitude = np.minimum(np.abs(pcm), _CLIP) + _BIAS
    # The exponent is the position of the highest set bit above bit 7,
    # i.e. floor(log2(magnitude)) - 7, clamped to the 8 segments 0..7.
    exponent = np.clip(np.floor(np.log2(magnitude)).astype(np.int32) - 7, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    encoded = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return encoded.astype(np.uint8).tobytes()


def pcm16_to_float(samples: np.ndarray) -> np.ndarray:
    """Scale int16 samples to float32 in [-1, 1], the format the VAD model expects."""
    return samples.astype(np.float32) / 32768.0


def silence_mulaw(duration_ms: int) -> bytes:
    """Mu-law silence (0xFF encodes PCM zero) of the given length."""
    return b"\xff" * (SAMPLE_RATE * duration_ms // 1000)
