import os, datetime, re
from urllib.parse import urlencode
from flask import request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
import gspread
from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound

# ---- Config ----
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "google-credentials.json")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "HungerCue Calls")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
ROLE_TABS = os.getenv("ROLE_TABS", "1") == "1"

# ---- Sheets init ----
def _init_sheets():
    try:
        gc = gspread.service_account(filename=GOOGLE_CREDENTIALS_JSON)
        sh = gc.open_by_key(GOOGLE_SHEET_ID) if GOOGLE_SHEET_ID else gc.open(GOOGLE_SHEET_NAME)
        return sh
    except (SpreadsheetNotFound, APIError) as e:
        print("❌ Google Sheets init error:", e)
        raise
    except Exception as e:
        print("❌ Google Sheets generic error:", e)
        raise

SHEET = None
WORKSHEETS = {}
TAB_BY_ROLE = {"Volunteer": "Volunteers", "Restaurant": "Restaurants", "Shelter": "Shelters"}

def ensure_sheets_ready():
    global SHEET, WORKSHEETS
    if SHEET:
        return
    SHEET = _init_sheets()
    if ROLE_TABS:
        # build worksheet cache, but don't crash if a tab is missing — create it with headers
        headers = ["Type", "Name", "Phone", "Notes", "Timestamp"]
        for role, title in TAB_BY_ROLE.items():
            try:
                ws = SHEET.worksheet(title)
            except WorksheetNotFound:
                ws = SHEET.add_worksheet(title=title, rows=1000, cols=10)
                ws.append_row(headers)
            WORKSHEETS[title] = ws

def log_to_sheets(role: str, name: str, phone: str, notes: str):
    ensure_sheets_ready()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [role, name, phone, notes, ts]
    if ROLE_TABS:
        tab = TAB_BY_ROLE.get(role, "Volunteers")
        ws = WORKSHEETS.get(tab) or SHEET.worksheet(tab)
        ws.append_row(row)
    else:
        SHEET.sheet1.append_row(row)
    print(f"📝 Logged: {role} | {name} | {phone} | {notes[:48]}...")

# ---- Twilio helpers ----
GATHER_TIMEOUT_SEC = 20
SPEECH_TIMEOUT = "auto"
LANG = "en-US"

def next_action(endpoint: str, **params) -> str:
    qs = urlencode({k: v for k, v in params.items() if v is not None})
    base = PUBLIC_BASE_URL or request.url_root.rstrip("/")
    return f"{base}/{endpoint}?{qs}" if qs else f"{base}/{endpoint}"

def g(key, default=""):
    return request.form.get(key) or request.args.get(key) or default

def gather_speech(prompt: str, action_url: str) -> VoiceResponse:
    vr = VoiceResponse()
    gather = Gather(
        input="speech dtmf",
        action=action_url,
        method="POST",
        speech_timeout=SPEECH_TIMEOUT,
        timeout=GATHER_TIMEOUT_SEC,
        language=LANG,
        finish_on_key="#",
        barge_in=True,
        enhanced=True,
        action_on_empty_result=True,
        speech_model="phone_call",
        hints="volunteer, restaurant, shelter, donate, food, receive, need, address, phone, pickup"
    )
    gather.say(prompt + " You can press pound when finished.")
    vr.append(gather)
    return vr

def internal_error_response(message_tts: str):
    vr = VoiceResponse()
    vr.say(message_tts)
    return Response(str(vr), mimetype="text/xml")

# ---- Light phone extraction fallback ----
PHONE_RE = re.compile(r"(?:\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})")

def maybe_find_phone(text: str) -> str:
    if not text:
        return ""
    m = PHONE_RE.search(text)
    return m.group(1) if m else ""
