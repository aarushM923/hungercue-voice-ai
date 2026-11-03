"""Cost-per-call estimate.

List prices (USD) at the time of writing. They change, so treat the output as
an estimate and check it against real invoices. One place to update them.
"""

from __future__ import annotations

from dataclasses import dataclass

TWILIO_VOICE_PER_MIN = 0.0085  # inbound local call
TWILIO_STREAM_PER_MIN = 0.004  # Media Streams add-on
DEEPGRAM_STT_PER_MIN = 0.0092  # Nova-3 multilingual streaming, pay as you go
DEEPGRAM_TTS_PER_1K_CHARS = 0.030  # Aura-2
TWILIO_SMS_PER_SEGMENT = 0.0083


@dataclass
class CallCost:
    minutes: float
    tts_chars: int
    llm_usd: float
    sms_segments: int = 0
    include_telephony: bool = True  # the eval harness skips Twilio (no phone involved)

    @property
    def total(self) -> float:
        telephony = (TWILIO_VOICE_PER_MIN + TWILIO_STREAM_PER_MIN) * self.minutes if self.include_telephony else 0.0
        return round(
            telephony
            + DEEPGRAM_STT_PER_MIN * self.minutes
            + DEEPGRAM_TTS_PER_1K_CHARS * self.tts_chars / 1000
            + self.llm_usd
            + TWILIO_SMS_PER_SEGMENT * self.sms_segments,
            5,
        )
