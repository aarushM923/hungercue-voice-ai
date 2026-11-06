# Eval dry run

3 simulated calls through the real pipeline (Deepgram STT and TTS, GPT-4.1-mini), real-time audio.

| Metric | Result |
|---|---|
| Caller classification accuracy | 100.0% |
| Slot accuracy | 52.6% (10/19 fields) |
| Task completion rate | 66.7% |
| Voice-to-voice latency p50 / p95 (caller side) | 730 ms / 1962 ms |
| Voice-to-voice latency p50 / p95 (server side) | 760 ms / 1884 ms |
| Barge-in success rate | 100.0% of 1 (median reaction 521 ms) |
| Cost per call (agent, incl. Twilio estimate) | $0.0536 |
| Calls with errors | 0 |

## Where the time goes (median per stage)

| Stage | Median |
|---|---|
| stt | 168 ms |
| endpoint | 56 ms |
| llm first token | 406 ms |
| tts first byte | 74 ms |

## Accuracy by condition

| Slice | Calls | Slot accuracy | Completion |
|---|---|---|---|
| clean audio | 2 | 33.3% | 50.0% |
| background noise | 1 | 85.7% | 100.0% |
| mumbling | 0 | n/a | n/a |
| interruptions | 1 | 0.0% | 0.0% |
| topic change | 1 | 0.0% | 0.0% |
| spanish | 1 | 0.0% | 0.0% |
| restaurant | 1 | 85.7% | 100.0% |
| shelter | 1 | 0.0% | 0.0% |
| volunteer | 1 | 80.0% | 100.0% |

## Accuracy by field

| Field | Accuracy |
|---|---|
| restaurant.address | 0.0% |
| restaurant.business_name | 100.0% |
| restaurant.items | 100.0% |
| restaurant.pickup_date | 100.0% |
| restaurant.pickup_end | 100.0% |
| restaurant.pickup_start | 100.0% |
| restaurant.quantity | 100.0% |
| shelter.address | 0.0% |
| shelter.capacity | 0.0% |
| shelter.dietary_restrictions | 0.0% |
| shelter.name | 0.0% |
| shelter.needs | 0.0% |
| shelter.receiving_end | 0.0% |
| shelter.receiving_start | 0.0% |
| volunteer.availability_days | 100.0% |
| volunteer.availability_end | 100.0% |
| volunteer.availability_start | 100.0% |
| volunteer.has_car | 100.0% |
| volunteer.neighborhoods | 0.0% |

## Missed fields

| Call | Field | Expected | Captured |
|---|---|---|---|
| S001 | address | 2210 Broadway | twenty two ten Broadway |
| S003 | neighborhoods | ['Park Slope'] | ['Parksloe'] |
| S004 | capacity | 80 | None |
| S004 | receiving_start | 12:00 | None |
| S004 | receiving_end | 20:00 | None |
| S004 | needs | ['prepared meals'] | [] |
| S004 | dietary_restrictions | [] | [] |
| S004 | name | Bridge Women's Center | None |
| S004 | address | 212 West 112th Street | None |

Eval overhead (simulated caller LLM and voice): $0.0138 per call; whole run $0.2.
