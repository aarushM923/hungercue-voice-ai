# HungerCue Voice AI (GPT-driven Clarifications)

This redesign turns the IVR into a conversational agent:
1) Caller says *one full message*.
2) We use GPT to extract **role, name, phone, notes**.
3) If anything is missing/ambiguous, GPT writes a **clarifying question**.
4) We loop until all fields are filled, then log to Google Sheets.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure your environment
cp .env.example .env
# Edit .env: PUBLIC_BASE_URL, OPENAI_API_KEY, and Google Sheets vars

# Put your Google service account file:
# google-credentials.json  (and share your Sheet with its client_email)

python3 app.py
```

Expose your local server to Twilio via ngrok/Cloudflare:
```bash
ngrok http 5000
# set PUBLIC_BASE_URL to the ngrok URL, no trailing slash
```

In the Twilio Console → Phone Numbers → Voice → A Call Comes In:
- **Webhook**: `POST {PUBLIC_BASE_URL}/voice`

## Endpoints
- `POST /voice` — Greets and asks for a full message (press # when done).
- `POST /process_full_message` — Extracts fields using GPT; if missing, asks a GPT-written clarifying question.
- `POST /clarify` — Accepts one more utterance, merges it, and repeats until complete.

## Google Sheets
- If `ROLE_TABS=1`, create tabs named: `Volunteers`, `Restaurants`, `Shelters`.
- Otherwise, everything logs to the first sheet.
- Provide either `GOOGLE_SHEET_ID` **or** `GOOGLE_SHEET_NAME`.
- Ensure your service account `client_email` has edit access to the Sheet.

## Env Vars
- `PUBLIC_BASE_URL`, `OPENAI_API_KEY`, `GOOGLE_CREDENTIALS_JSON`
- `GOOGLE_SHEET_ID` or `GOOGLE_SHEET_NAME`
- `ROLE_TABS` (1 or 0)

## Notes
- We pass small bits of state between steps via querystring. Keep answers concise.
- Phone autodetection fallback uses regex if GPT misses it.
