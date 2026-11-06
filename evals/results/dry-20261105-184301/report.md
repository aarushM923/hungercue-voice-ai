# Eval dry run

3 simulated calls through the real pipeline (Deepgram STT and TTS, GPT-4.1-mini), real-time audio.

| Metric | Result |
|---|---|
| Caller classification accuracy | 100.0% |
| Slot accuracy | 89.5% (17/19 fields) |
| Task completion rate | 100.0% |
| Voice-to-voice latency p50 / p95 (caller side) | 3274 ms / 5293 ms |
| Voice-to-voice latency p50 / p95 (server side) | 3146 ms / 5415 ms |
| Barge-in success rate | 100.0% of 1 (median reaction 541 ms) |
| Cost per call (agent, incl. Twilio estimate) | $0.0553 |
| Calls with errors | 0 |

## Where the time goes (median per stage)

| Stage | Median |
|---|---|
| stt | 1602 ms |
| endpoint | 537 ms |
| llm first token | 644 ms |
| tts first byte | 163 ms |

## Accuracy by condition

| Slice | Calls | Slot accuracy | Completion |
|---|---|---|---|
| clean audio | 2 | 83.3% | 100.0% |
| background noise | 1 | 100.0% | 100.0% |
| mumbling | 0 | n/a | n/a |
| interruptions | 1 | 71.4% | 100.0% |
| topic change | 1 | 71.4% | 100.0% |
| spanish | 1 | 71.4% | 100.0% |
| restaurant | 1 | 100.0% | 100.0% |
| shelter | 1 | 71.4% | 100.0% |
| volunteer | 1 | 100.0% | 100.0% |

## Accuracy by field

| Field | Accuracy |
|---|---|
| restaurant.address | 100.0% |
| restaurant.business_name | 100.0% |
| restaurant.items | 100.0% |
| restaurant.pickup_date | 100.0% |
| restaurant.pickup_end | 100.0% |
| restaurant.pickup_start | 100.0% |
| restaurant.quantity | 100.0% |
| shelter.address | 0.0% |
| shelter.capacity | 100.0% |
| shelter.dietary_restrictions | 100.0% |
| shelter.name | 100.0% |
| shelter.needs | 0.0% |
| shelter.receiving_end | 100.0% |
| shelter.receiving_start | 100.0% |
| volunteer.availability_days | 100.0% |
| volunteer.availability_end | 100.0% |
| volunteer.availability_start | 100.0% |
| volunteer.has_car | 100.0% |
| volunteer.neighborhoods | 100.0% |

## Missed fields

| Call | Field | Expected | Captured |
|---|---|---|---|
| S004 | needs | ['prepared meals'] | [] |
| S004 | address | 212 West 112th Street |  |

Eval overhead (simulated caller LLM and voice): $0.0124 per call; whole run $0.2.
