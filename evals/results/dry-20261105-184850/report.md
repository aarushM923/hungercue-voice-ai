# Eval dry run

3 simulated calls through the real pipeline (Deepgram STT and TTS, GPT-4.1-mini), real-time audio.

| Metric | Result |
|---|---|
| Caller classification accuracy | 66.7% |
| Slot accuracy | 57.9% (11/19 fields) |
| Task completion rate | 66.7% |
| Voice-to-voice latency p50 / p95 (caller side) | 1064 ms / 2202 ms |
| Voice-to-voice latency p50 / p95 (server side) | 975 ms / 2365 ms |
| Barge-in success rate | n/a of 0 (median reaction n/a) |
| Cost per call (agent, incl. Twilio estimate) | $0.0352 |
| Calls with errors | 1 |

## Where the time goes (median per stage)

| Stage | Median |
|---|---|
| stt | 323 ms |
| endpoint | 0 ms |
| llm first token | 435 ms |
| tts first byte | 168 ms |

## Accuracy by condition

| Slice | Calls | Slot accuracy | Completion |
|---|---|---|---|
| clean audio | 2 | 41.7% | 50.0% |
| background noise | 1 | 85.7% | 100.0% |
| mumbling | 0 | n/a | n/a |
| interruptions | 1 | 0.0% | 0.0% |
| topic change | 1 | 0.0% | 0.0% |
| spanish | 1 | 0.0% | 0.0% |
| restaurant | 1 | 85.7% | 100.0% |
| shelter | 1 | 0.0% | 0.0% |
| volunteer | 1 | 100.0% | 100.0% |

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
| volunteer.neighborhoods | 100.0% |

## Missed fields

| Call | Field | Expected | Captured |
|---|---|---|---|
| S001 | address | 2210 Broadway | twenty two ten Broadway |
| S004 | capacity | 80 | None |
| S004 | receiving_start | 12:00 | None |
| S004 | receiving_end | 20:00 | None |
| S004 | needs | ['prepared meals'] | [] |
| S004 | dietary_restrictions | [] | [] |
| S004 | name | Bridge Women's Center | None |
| S004 | address | 212 West 112th Street | None |

## Errors

- S004: RemoteProtocolError: Server disconnected without sending a response.

Eval overhead (simulated caller LLM and voice): $0.0068 per call; whole run $0.13.
