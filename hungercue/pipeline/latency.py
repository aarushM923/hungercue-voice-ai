"""Per-turn latency bookkeeping.

Voice-to-voice latency = time from the caller finishing speaking to the first
byte of the agent's reply leaving our server. We break it into stages so we
can show where the time goes:

  caller stops (VAD)          t_speech_end
  last final transcript       t_transcript      stt_ms       = t_transcript - t_speech_end
  we decide "their turn ended" t_trigger         endpoint_ms  = t_trigger - t_transcript
  LLM's first text token      t_first_token     llm_first_token_ms = t_first_token - t_trigger
  TTS first audio byte        t_first_audio     tts_first_byte_ms  = t_first_audio - t_tts_request
                                                v2v_ms = t_first_audio - t_speech_end

Network time between Twilio and the caller can't be seen from the server; the
eval harness measures end-to-end from the "caller" side.

All times are from the event loop's monotonic clock (seconds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _ms(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(max(0.0, (b - a) * 1000), 1)


@dataclass
class TurnTimer:
    turn_index: int
    t_speech_end: float | None = None
    t_transcript: float | None = None
    t_trigger: float | None = None
    t_first_token: float | None = None
    t_tts_request: float | None = None
    t_first_audio: float | None = None
    interrupted: bool = False

    def breakdown(self) -> dict[str, Any]:
        # If VAD missed the end of speech, fall back to the transcript time so
        # v2v is still reported (slightly optimistic, flagged by stt_ms=None).
        start = self.t_speech_end if self.t_speech_end is not None else self.t_transcript
        return {
            "stt_ms": _ms(self.t_speech_end, self.t_transcript),
            "endpoint_ms": _ms(self.t_transcript, self.t_trigger),
            "llm_first_token_ms": _ms(self.t_trigger, self.t_first_token),
            "tts_first_byte_ms": _ms(self.t_tts_request, self.t_first_audio),
            "v2v_ms": _ms(start, self.t_first_audio),
            "interrupted": self.interrupted,
        }


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (p50 = median). None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    # Nearest-rank: the smallest value with at least pct% of values at or below it.
    rank = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[rank]
