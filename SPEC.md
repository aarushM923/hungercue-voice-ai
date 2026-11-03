Project 2: HungerCue Voice v2 — a real-time phone agent that runs the whole pickup
Headline result to aim for: "HungerCue's phone agent answers in under 800 ms, handles interruptions like a person, and turns a 2-minute call into a scheduled pickup with a matched shelter and volunteer, with 95%+ of details captured correctly across a 100-call test suite."
Where it is today vs. v2
Today the agent takes one spoken message and classifies the caller as a restaurant, shelter or volunteer. V2 holds a full natural conversation, collects every detail, books the pickup, matches it to a shelter and volunteer, and confirms by text, with measured speed and accuracy.
Architecture (the call path)
1. Caller dials the HungerCue number. Twilio answers and opens a bidirectional Media Stream: live audio over a WebSocket.
2. Async server (FastAPI with WebSockets, replacing Flask) receives 8 kHz audio chunks.
3. Voice activity detection (Silero VAD) decides when the caller has stopped talking.
4. Streaming speech-to-text (Deepgram, or faster-whisper self-hosted) produces the transcript while the caller is still speaking.
5. LLM with tool calling (GPT-4.1-mini, streaming) decides what to say and which tools to call.
6. Streaming text-to-speech (ElevenLabs, Cartesia or Deepgram Aura) starts speaking the first words before the full reply is written.
7. Audio goes back to Twilio on the same WebSocket.
8. Barge-in: if VAD hears the caller speak while the agent is talking, the server sends Twilio a clear message, cancels the TTS and LLM stream, and listens.
State lives in Redis (per-call session) and Postgres (restaurants, shelters, volunteers, pickups).
Conversation flows (LLM tools)
• Restaurant: log_donation(items, quantity, pickup_window, address, notes). It repeats details back to confirm, then creates the pickup.
• Shelter: update_shelter(capacity, needs, dietary_restrictions, receiving_hours).
• Volunteer: update_volunteer(availability, neighborhoods, has_car).
• Returning callers: look up the phone number, e.g. "Hi, is this another pickup from the bagel shop on 112th?"
• Escape hatch: transfer_to_human() or take a voicemail when the agent is unsure or the caller asks.
• Bonus: Spanish support, since many NYC restaurant staff prefer it; detect language on the first turn.
Matching engine
When a donation is logged, score every available shelter s and volunteer v and assign the best pair. Here dist is travel time, overlap is the hours shared by the pickup window and the shelter's receiving hours, and fit rewards matching needs and dietary rules.
\text{score}(s, v) = w_1\,\text{fit}(d, s) + w_2\,\text{overlap}(d, s) - w_3\,\text{dist}(d, v) - w_4\,\text{dist}(v, s)
Start greedy (best pair per donation). For a stretch goal, batch a day's donations and solve the assignment with the Hungarian algorithm (scipy.optimize.linear_sum_assignment).
Outbound follow-up
• SMS confirmation to the restaurant with the pickup time and volunteer name
• Automated call or SMS to the matched volunteer: "Reply YES to accept"; if no reply in 15 minutes, offer it to the next-best volunteer
• Reminder text 30 minutes before pickup
Latency budget
Voice-to-voice latency is the time from the caller finishing speaking to the first audio byte the caller hears. Targets: under 800 ms median and under 1.5 s at the 95th percentile.
T_{\text{v2v}} = T_{\text{endpoint}} + T_{\text{STT}} + T_{\text{LLM first token}} + T_{\text{TTS first byte}} + T_{\text{network}}
Rough budget: endpointing 200 ms, STT 100 ms, LLM first token 300 ms, TTS first byte 150 ms, network 50 ms. Log every component on every turn so you can show where time goes.
Evaluation suite
1. 100 test calls: an LLM plays restaurant, shelter and volunteer personas through TTS into your pipeline, with background noise, accents, mumbling, interruptions and topic changes. Add real recorded calls where callers consent.
2. Metrics per run:
    ◦ caller classification accuracy
    ◦ slot accuracy: share of fields (items, quantity, time, address) captured exactly right
    ◦ task completion rate: calls that end with a valid pickup booked
    ◦ latency p50 and p95
    ◦ barge-in success rate
    ◦ cost per call in dollars
3. CI: run a 20-call smoke suite on every pull request; fail the build if slot accuracy or latency regresses.
Ops dashboard
A small web page for HungerCue coordinators: today's pickups, match status, call transcripts with audio, and a latency chart.
Timeline (about 7 weeks part-time)
Weeks
Milestone
1
Twilio Media Streams plus FastAPI; echo audio back; measure baseline latency
2
Streaming STT, LLM and TTS loop; first full conversation
3
Barge-in, endpointing tuning, latency logging
4
Tools, Postgres schema, returning-caller lookup
5
Matching engine plus SMS confirmations
6
100-call eval suite, CI, Spanish
7
Dashboard, deploy (Fly.io or Render), demo video, write-up
Deliverables
• The live HungerCue number running v2 for real restaurants
• Public repo with architecture diagram, eval results and latency breakdown
• A 2-minute demo video: a real call with an interruption, then the SMS confirmations arriving
• Resume bullet, filled in with your measured numbers: "Rebuilt HungerCue's phone agent as a real-time streaming system (Twilio Media Streams, FastAPI, streaming STT/TTS) with sub-800 ms median response, interruption handling and automated shelter-volunteer matching; 9X% slot accuracy across a 100-call eval suite."
Privacy
Play a short recording notice at the start of every call, keep transcripts in access-controlled storage, and delete raw audio after 30 days unless a caller agrees otherwise.