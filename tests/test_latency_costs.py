from hungercue.pipeline.costs import CallCost
from hungercue.pipeline.latency import TurnTimer, percentile


def test_breakdown() -> None:
    t = TurnTimer(1, t_speech_end=10.0, t_transcript=10.1, t_trigger=10.12, t_first_token=10.42,
                  t_tts_request=10.45, t_first_audio=10.6)
    b = t.breakdown()
    assert b == {"stt_ms": 100.0, "endpoint_ms": 20.0, "llm_first_token_ms": 300.0,
                 "tts_first_byte_ms": 150.0, "v2v_ms": 600.0, "interrupted": False}


def test_breakdown_without_vad_end_uses_transcript_time() -> None:
    b = TurnTimer(1, t_transcript=5.0, t_trigger=5.0, t_first_audio=5.5).breakdown()
    assert b["v2v_ms"] == 500.0 and b["stt_ms"] is None and b["llm_first_token_ms"] is None


def test_percentile_nearest_rank() -> None:
    vals = [float(v) for v in range(1, 101)]
    assert percentile(vals, 50) == 50 and percentile(vals, 95) == 95 and percentile(vals, 100) == 100
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0 and percentile([], 50) is None


def test_cost() -> None:
    c = CallCost(minutes=2, tts_chars=1000, llm_usd=0.002)
    assert abs(c.total - (0.0125 * 2 + 0.0092 * 2 + 0.03 + 0.002)) < 1e-6
    assert CallCost(2, 1000, 0.002, include_telephony=False).total < c.total
