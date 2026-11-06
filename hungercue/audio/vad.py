"""Voice activity detection (VAD) with Silero.

Two jobs in the call pipeline:
1. Barge-in: notice within ~100 ms that the caller started talking while the
   agent is speaking, so we can stop talking (like a person would).
2. Endpointing: notice that the caller *stopped* talking, which starts the
   voice-to-voice latency clock and helps decide when to reply.

The Silero model (MIT licence, hungercue/audio/models/silero_vad.onnx) takes
256-sample chunks at 8 kHz (32 ms) and returns P(speech). It is a small
recurrent network, so we carry its hidden `state` from one chunk to the next,
plus the last 32 samples ("context") so chunk boundaries don't confuse it.
We run it with onnxruntime instead of PyTorch to keep the install small.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hungercue.audio.mulaw import mulaw_to_pcm16, pcm16_to_float

MODEL_PATH = Path(__file__).parent / "models" / "silero_vad.onnx"
SAMPLE_RATE = 8000
CHUNK = 256  # samples per model call at 8 kHz (32 ms)
CONTEXT = 32  # samples of the previous chunk prepended to each call
CHUNK_MS = CHUNK * 1000 // SAMPLE_RATE  # 32


def load_session(model_path: Path = MODEL_PATH) -> Any:
    import onnxruntime as ort  # imported here so tests that fake the VAD don't load it

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1  # tiny model: threads add overhead, not speed
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])


class SileroVAD:
    """Stateful speech-probability model. One instance per call (state is per stream)."""

    def __init__(self, model_path: Path = MODEL_PATH, session: Any = None) -> None:
        # Loading the model takes ~50 ms, so the server loads it once and passes the
        # shared `session` in; only the small per-call state lives on this object.
        self.session = session or load_session(model_path)
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        """P(speech) for exactly CHUNK float32 samples in [-1, 1]."""
        x = np.concatenate([self.context, chunk.reshape(1, -1).astype(np.float32)], axis=1)
        out, self.state = self.session.run(
            None, {"input": x, "state": self.state, "sr": np.array(SAMPLE_RATE, dtype=np.int64)}
        )
        self.context = x[:, -CONTEXT:]
        return float(out[0][0])


@dataclass
class VadEvent:
    kind: str  # "speech_start" | "speech_pause" | "speech_end"
    stream_ms: int  # position in the audio stream where it happened


class VadSegmenter:
    """Turns per-chunk probabilities into start/end events with hysteresis.

    - Speech starts after `min_speech_ms` of chunks above `start_threshold`
      (ignores clicks and short noises).
    - Speech ends after `min_silence_ms` of chunks below `end_threshold`
      (ignores the natural gaps between words).
    Using a lower end threshold than start threshold (hysteresis) stops the
    state from flickering when the probability hovers around one value.
    """

    def __init__(
        self,
        prob_fn: Callable[[np.ndarray], float],
        start_threshold: float = 0.5,
        end_threshold: float = 0.35,
        min_speech_ms: int = 96,
        min_silence_ms: int = 224,
        pause_ms: int = 96,
    ) -> None:
        self.prob_fn = prob_fn
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.min_speech_chunks = max(1, min_speech_ms // CHUNK_MS)
        self.min_silence_chunks = max(1, min_silence_ms // CHUNK_MS)
        # A "pause" is a shorter silence than an end of turn. It fires first, so the
        # call can ask Deepgram to finalise early while we wait to confirm the end.
        self.pause_chunks = max(1, min(pause_ms // CHUNK_MS, self.min_silence_chunks))
        self.buffer = np.zeros(0, dtype=np.float32)
        self.in_speech = False
        self.voiced_run = 0
        self.silent_run = 0
        self.samples_seen = 0  # total samples processed, to compute stream positions
        self.last_voiced_ms = 0
        self.last_prob = 0.0

    def push_mulaw(self, payload: bytes) -> list[VadEvent]:
        """Feed one Twilio frame (160 mu-law bytes = 20 ms); return any events."""
        return self.push_float(pcm16_to_float(mulaw_to_pcm16(payload)))

    def push_float(self, samples: np.ndarray) -> list[VadEvent]:
        # Twilio frames (160 samples) don't line up with model chunks (256),
        # so we buffer and process whole chunks as they become available.
        self.buffer = np.concatenate([self.buffer, samples])
        events: list[VadEvent] = []
        while len(self.buffer) >= CHUNK:
            chunk, self.buffer = self.buffer[:CHUNK], self.buffer[CHUNK:]
            self.samples_seen += CHUNK
            now_ms = self.samples_seen * 1000 // SAMPLE_RATE
            p = self.last_prob = self.prob_fn(chunk)
            if not self.in_speech:
                self.voiced_run = self.voiced_run + 1 if p >= self.start_threshold else 0
                if self.voiced_run >= self.min_speech_chunks:
                    self.in_speech, self.silent_run = True, 0
                    self.last_voiced_ms = now_ms
                    start_ms = now_ms - self.voiced_run * CHUNK_MS
                    events.append(VadEvent("speech_start", start_ms))
            else:
                if p < self.end_threshold:
                    self.silent_run += 1
                    if self.silent_run == self.pause_chunks:  # exactly once per silent stretch
                        events.append(VadEvent("speech_pause", self.last_voiced_ms))
                    if self.silent_run >= self.min_silence_chunks:
                        self.in_speech, self.voiced_run = False, 0
                        # The caller actually stopped at the last voiced chunk, not now.
                        events.append(VadEvent("speech_end", self.last_voiced_ms))
                else:
                    self.silent_run = 0
                    self.last_voiced_ms = now_ms
        return events
