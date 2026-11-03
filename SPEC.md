Claude
You can do that. Have Claude build each project straight through in one long session, and only stop when it truly needs you. Using the same repo is fine too: the old code just gets replaced.

A few things will still pause it, because only you can do them:

Keys and accounts: Twilio, OpenAI and Deepgram keys for HungerCue, and AWS access for HackGuard.
Spending money: HackGuard steps 2–6 need the GPU on AWS, and the experiments take hours of computing no matter how fast the code gets written.
Real-world testing: someone has to actually call the HungerCue number.
Everything else, all the code, tests and docs, Claude can write in one go.

HackGuard (already running)
When Step 1 finishes, send this in the HackGuard Claude tab:

Update CLAUDE.md: replace the rule “Build one milestone at a time...” with “Build the entire spec in one continuous session. Only stop when you need something from me: API keys, AWS access, approval to spend money, or a decision only I can make.” Then continue straight through every remaining step of SPEC.md. When you reach the parts that need an AWS GPU, write all the code and scripts first, then tell me exactly what to set up so we can run it.

HungerCue (full rebuild in the same repo)
In the HungerCue setup, use this CLAUDE.md instead of the earlier one:

# Project rules
- This repo is being completely rebuilt to match SPEC.md. The old Flask code is outdated; replace it entirely rather than patching it. Keep only what is still useful (e.g. Google Sheets logging, if it fits the new design).
- This is a LIVE system that real restaurants and shelters call. Work on the v2 branch. Never change the live Twilio number's settings or deploy without asking me first.
- Build the entire spec in one continuous session. Only stop when you need something from me: API keys, accounts, approval to spend money (calls, SMS, paid APIs), or a decision only I can make.
- Python 3.11, type hints, pytest tests for every module. Run all tests before saying you're done.
- Use a virtual environment in .venv. Secrets live in .env, never committed. Free downloads and package installs are fine.
- Commit after each major piece works, with clear messages.
- Never add Claude as a co-author or mention Claude in commit messages or pull requests.
- I am new to programming tools. Explain each command before running it, and explain non-obvious code in comments. I need to be able to explain every part in an interview.
Then, in plan mode, paste:

Read SPEC.md, CLAUDE.md, and the existing code. This is a full rebuild: replace the old Flask app with the complete v2 system in SPEC.md (real-time streaming, tool calling, matching engine, SMS, eval suite, dashboard). Keep Google Sheets logging since our coordinators use it. Remove the committed .venv folder from git tracking (without deleting it locally), make sure .gitignore covers .venv, .env and google-credentials.json, and check git history for any leaked API keys. Then give me one plan for building the entire spec start to finish, and a list of every account and API key I’ll need.

Send me the plan before you say go.

Two practical notes:

Switch Claude Code to Bedrock first (/setup-bedrock, then pin Sonnet). Two projects built in one long session will blow through Pro’s 5-hour limit fast; on Bedrock it just draws from your $10K.
“In one go” is still a long session, likely several hours per project. You don’t need to watch every second, but check back to approve prompts and answer its questions.
Pasted text.txt
Pasted text.txt
TXT
now tell me again what to do step by tep for everyhting

Step 1 of HackGuard is done, and it’s solid: 1,200 tasks, 60 passing tests, and Claude caught and fixed two real bugs on its own. Here’s everything from here, in order.

Part A: HackGuard, continue in one go
1. Upload Step 1 to GitHub. It’s committed but not pushed. In the zsh tab:

cd ~/projects/HackGuard
git push
2. Send this in the HackGuard Claude tab. It switches Claude to one-go mode and fixes a GPU mistake in the spec:

Update CLAUDE.md: replace the rule “Build one milestone at a time...” with “Build the entire spec in one continuous session. Only stop when you need something from me: API keys, AWS access, approval to spend money, or a decision only I can make.” Then build Steps 2–6 straight through. Three requirements: (1) Use AWS g5.2xlarge (one A10G GPU, 24 GB), not A100/H100. My GPU quota is 8 vCPUs, which fits exactly one g5.2xlarge, and a 7B model fits in 24 GB. Fix that line in SPEC.md too. (2) Before using any GPU, test the whole pipeline end to end on my laptop with a tiny model (Qwen2.5-Coder-0.5B-Instruct) on about 20 tasks, so all bugs are found for free. (3) Run model-written code inside Docker on EC2, as you suggested. When everything is ready for the real run, give me exact step-by-step instructions to set up AWS access and launch the instance, and add an auto-shutdown so the GPU can’t run up costs while idle.

3. When Claude asks for AWS access, you’ll likely do this in the zsh tab:

brew install awscli
aws login
aws sts get-caller-identity
aws login opens your browser to sign in; the last line confirms it worked. Wait for Claude to ask, though, since it may give slightly different steps.

4. Check your GPU quota in the AWS console: Service Quotas → Quota request history. It needs to say Approved before the real run.

Part B: HungerCue, set up and start
5. Press Cmd + T for a new tab, then one line at a time:

cd ~/projects
git clone https://github.com/aarushM923/hungercue-voice-ai.git hungercue
cd hungercue
git checkout -b v2
touch SPEC.md CLAUDE.md
6. Copy the “Project 2” section from the Project Specs doc, then:

open -e SPEC.md
In TextEdit, press Cmd + V, Cmd + S, Cmd + W.

7. Copy the one-go HungerCue CLAUDE.md from my earlier message (the one that starts “This repo is being completely rebuilt...”), then:

open -e CLAUDE.md
In TextEdit, press Cmd + V, Cmd + S, Cmd + W.

8. Check and commit:

wc -c SPEC.md CLAUDE.md
git add SPEC.md CLAUDE.md
git commit -m "Add v2 spec and project rules"
claude
Both numbers should be well over 100. Trust the folder (it should say .../projects/hungercue).

9. Press Shift + Tab until plan mode on, then paste the HungerCue full-rebuild prompt from my earlier message (the one starting “Read SPEC.md, CLAUDE.md, and the existing code. This is a full rebuild...”).

10. Send me the plan before saying go.

Part C: Switch to your AWS credits (recommended)
Both projects running in one go will hit Pro’s limit fast. In either Claude tab:

Bedrock console → API keys → create a key.
Type /setup-bedrock → API key → paste → us-east-1 → pin Sonnet.
Type /status, which should say Amazon Bedrock.
Note on auto mode: your HackGuard tab shows “auto mode on,” so Claude runs commands without asking you. That’s fine for writing code, and your CLAUDE.md still requires it to ask before spending money. Just glance at what it’s doing now and then, especially once AWS is connected.

Update CLAUDE.md: replace the rule “Build one milestone at a time...” with “Build the entire spec in one continuous session. Only stop when you need something from me: API keys, AWS access, approval to spend money, or a decision only I can make.” Then build Steps 2–6 straight through. Three requirements: (1) Use AWS g5.2xlarge (one A10G GPU, 24 GB), not A100/H100. My GPU quota is 8 vCPUs, which fits exactly one g5.2xlarge, and a 7B model fits in 24 GB. Fix that line in SPEC.md too. (2) Before using any GPU, test the whole pipeline end to end on my laptop with a tiny model (Qwen2.5-Coder-0.5B-Instruct) on about 20 tasks, so all bugs are found for free. (3) Run model-written code inside Docker on EC2, as you suggested. When everything is ready for the real run, give me exact step-by-step instructions to set up AWS access and launch the instance, and add an auto-shutdown so the GPU can’t run up costs while idle.

is there anything else i should say??

That covers the main points. I’d add five more things, mostly to save money and avoid extra accounts. Paste this right after it, as one message:

Also: (4) Push to GitHub after every commit, not just commit. (5) For the LLM-judge baseline, use Claude on Amazon Bedrock, since my AWS credits cover it. (6) Skip Weights & Biases and log results to local CSV/JSON files instead, so I don’t need another account. (7) For the second model in the transfer experiment, pick one that fits in 24 GB (e.g. Llama-3.1-8B-Instruct), and tell me early if it needs a Hugging Face account or license approval, since that can take time. (8) Before launching any EC2 instance, give me a cost estimate for the full run, and keep total AWS spend under $300. When everything is done, explain the results to me in plain English: what we found, what the numbers mean, and what I’d say about them in an interview.

Why each one matters:

Push: your work gets backed up on GitHub and shows up on your profile.
Claude on Bedrock for the judge: it’s paid by your credits, so you don’t need an OpenAI key.
No W&B: one less account to set up.
Second model: some models, like Llama, require approval on Hugging Face, which can take a day. Better to find out early.
$300 cap plus estimate: a hard spending limit, so nothing surprises you.
Plain-English results: this becomes your interview prep and your real resume numbers.
idc about the cost tho

Then drop the cost cap. I’d still keep the auto-shutdown, though. It costs you nothing and stops a forgotten GPU from quietly eating your credits, at about $730 a month per idle machine.

Revised add-on:

Also: (4) Push to GitHub after every commit, not just commit. (5) For the LLM-judge baseline, use Claude on Amazon Bedrock, since my AWS credits cover it. (6) Skip Weights & Biases and log results to local CSV/JSON files instead, so I don’t need another account. (7) For the second model in the transfer experiment, pick one that fits in 24 GB (e.g. Llama-3.1-8B-Instruct), and tell me early if it needs a Hugging Face account or license approval. When everything is done, explain the results to me in plain English: what we found, what the numbers mean, and what I’d say about them in an interview.




Project 2: HungerCue Voice v2 — a real-time phone agent that runs the whole pickup Headline result to aim for: "HungerCue's phone agent answers in under 800 ms, handles interruptions like a person, and turns a 2-minute call into a scheduled pickup with a matched shelter and volunteer, with 95%+ of d

pasted




Claude is AI and can make mistakes.
Pasted content
5.52 KB •69 lines
•
Formatting may be inconsistent from source

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