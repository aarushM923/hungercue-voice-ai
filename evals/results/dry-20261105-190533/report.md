# Eval dry run

3 simulated calls through the real pipeline (Deepgram STT and TTS, GPT-4.1-mini), real-time audio.

| Metric | Result |
|---|---|
| Caller classification accuracy | 0.0% |
| Slot accuracy | 0.0% (0/19 fields) |
| Task completion rate | 0.0% |
| Voice-to-voice latency p50 / p95 (caller side) | n/a / n/a |
| Voice-to-voice latency p50 / p95 (server side) | n/a / n/a |
| Barge-in success rate | n/a of 0 (median reaction n/a) |
| Cost per call (agent, incl. Twilio estimate) | $0.0206 |
| Calls with errors | 3 |

## Where the time goes (median per stage)

| Stage | Median |
|---|---|
| stt | n/a |
| endpoint | n/a |
| llm first token | n/a |
| tts first byte | n/a |

## Accuracy by condition

| Slice | Calls | Slot accuracy | Completion |
|---|---|---|---|
| clean audio | 2 | 0.0% | 0.0% |
| background noise | 1 | 0.0% | 0.0% |
| mumbling | 0 | n/a | n/a |
| interruptions | 1 | 0.0% | 0.0% |
| topic change | 1 | 0.0% | 0.0% |
| spanish | 1 | 0.0% | 0.0% |
| restaurant | 1 | 0.0% | 0.0% |
| shelter | 1 | 0.0% | 0.0% |
| volunteer | 1 | 0.0% | 0.0% |

## Accuracy by field

| Field | Accuracy |
|---|---|
| restaurant.address | 0.0% |
| restaurant.business_name | 0.0% |
| restaurant.items | 0.0% |
| restaurant.pickup_date | 0.0% |
| restaurant.pickup_end | 0.0% |
| restaurant.pickup_start | 0.0% |
| restaurant.quantity | 0.0% |
| shelter.address | 0.0% |
| shelter.capacity | 0.0% |
| shelter.dietary_restrictions | 0.0% |
| shelter.name | 0.0% |
| shelter.needs | 0.0% |
| shelter.receiving_end | 0.0% |
| shelter.receiving_start | 0.0% |
| volunteer.availability_days | 0.0% |
| volunteer.availability_end | 0.0% |
| volunteer.availability_start | 0.0% |
| volunteer.has_car | 0.0% |
| volunteer.neighborhoods | 0.0% |

## Missed fields

| Call | Field | Expected | Captured |
|---|---|---|---|
| S001 | items | [['pastr', 'pastel', 'pastri'], ['muffin']] | None |
| S001 | quantity | 2 | None |
| S001 | pickup_date | 2026-11-06 | None |
| S001 | pickup_start | 18:00 | None |
| S001 | pickup_end | 20:00 | None |
| S001 | address | 2210 Broadway | None |
| S001 | business_name | Sunrise Bagels |  |
| S003 | availability_days | ['mon', 'tue'] | [] |
| S003 | availability_start | 10:00 | [] |
| S003 | availability_end | 16:00 | [] |
| S003 | neighborhoods | ['Park Slope'] | [] |
| S003 | has_car | False | None |
| S004 | capacity | 80 | None |
| S004 | receiving_start | 12:00 | None |
| S004 | receiving_end | 20:00 | None |
| S004 | needs | ['prepared meals'] | [] |
| S004 | dietary_restrictions | [] | [] |
| S004 | name | Bridge Women's Center | None |
| S004 | address | 212 West 112th Street | None |

## Errors

- S001: ReadError: 
- S003: ReadTimeout: 
- S004: ReadTimeout: 

Eval overhead (simulated caller LLM and voice): $0.0029 per call; whole run $0.07.
