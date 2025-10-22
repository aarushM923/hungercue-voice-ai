import os
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from utils import next_action, gather_speech, internal_error_response, g, log_to_sheets, maybe_find_phone
from nlu import extract_structured, clarifying_question, missing_fields
from dotenv import load_dotenv
from pathlib import Path

# Load .env explicitly
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

app = Flask(__name__)


@app.route("/healthz", methods=["GET"])
def healthz():
    return {"ok": True}


@app.route("/voice", methods=["POST"])
def voice():
    prompt = (
        "Welcome to HungerCue. Please describe your message in one go — "
        "you can include your name, phone number, and whether you’re a restaurant, shelter, or volunteer. "
        "Press pound when you’re done."
    )
    vr = gather_speech(prompt, next_action("process_full_message"))
    return Response(str(vr), mimetype="text/xml")


@app.route("/process_full_message", methods=["POST", "GET"])
def process_full_message():
    speech = request.values.get("SpeechResult", "") or ""
    data = extract_structured(speech)

    # If caller said "use my caller ID", or GPT flagged CALLER_ID, use Twilio's From
    if data.get("phone") == "CALLER_ID" or ("use my caller id" in speech.lower()):
        data["phone"] = request.values.get("From", "") or data.get("phone", "")

    # Fallback: regex phone from the transcript if GPT missed it
    if not data.get("phone"):
        fb = maybe_find_phone(speech)
        if fb:
            data["phone"] = fb

    miss = missing_fields(data)

    # If nothing is missing, log + finish
    if not miss:
        try:
            log_to_sheets(
                (data.get("role") or "volunteer").title(),
                data.get("name") or "Unknown",
                data.get("phone") or "",
                data.get("notes") or speech,
            )
        except Exception as e:
            print("❌ save error:", e)
            return internal_error_response("Sorry, we had trouble saving your information.")
        vr = VoiceResponse()
        vr.say(f"Thank you {data.get('name','')}. Your {data.get('role','volunteer')} information has been saved. Goodbye!")
        return Response(str(vr), mimetype="text/xml")

    # Otherwise ask ONE clarifying question and pass current state forward
    ctx = {"known": data, "original": speech, "history": []}
    q = clarifying_question(ctx)

    # Track how many times we asked for name/phone to prevent loops
    asked_name_count = 1 if "name" in miss else 0
    asked_phone_count = 1 if "phone" in miss else 0

    vr = gather_speech(
        q,
        next_action(
            "clarify",
            role=data.get("role", ""),
            name=data.get("name", ""),
            phone=data.get("phone", ""),
            notes=data.get("notes", ""),
            original=speech,
            asked_name_count=asked_name_count,
            asked_phone_count=asked_phone_count,
        ),
    )
    return Response(str(vr), mimetype="text/xml")


def _only_digits(s: str) -> str:
    return "".join([c for c in (s or "") if c.isdigit()])


@app.route("/clarify", methods=["POST", "GET"])
def clarify():
    # Pull prior state carried on the URL
    role = g("role", "")
    name = g("name", "")
    phone = g("phone", "")
    notes = g("notes", "")
    original = g("original", "")

    asked_name_count = int(g("asked_name_count", "0") or 0)
    asked_phone_count = int(g("asked_phone_count", "0") or 0)

    # New utterance from caller
    utter = request.values.get("SpeechResult", "") or ""
    upd = extract_structured(utter)

    # --- MERGE NEW ANSWERS WITH PRIOR STATE ---

    # Role
    role = upd.get("role") or role

    # Name (avoid infinite loop: after 1 ask, accept Unknown)
    name = upd.get("name") or name
    if not name and asked_name_count >= 1:
        name = "Unknown"

    # Phone:
    # 1) If caller said "use my caller ID" or GPT flagged CALLER_ID, take Twilio From
    if upd.get("phone") == "CALLER_ID" or ("use my caller id" in utter.lower()):
        phone = request.values.get("From", "") or phone

    # 2) Otherwise, take any digits GPT extracted (or regex fallback)
    if not phone and upd.get("phone"):
        phone = _only_digits(upd.get("phone"))
    if not phone:
        fb = maybe_find_phone(utter)
        if fb:
            phone = _only_digits(fb)

    # Notes: keep additive
    notes = (notes + " " + (upd.get("notes") or utter)).strip()

    data = {"role": role, "name": name, "phone": phone, "notes": notes}

    # Decide what's still missing
    miss = missing_fields(data)

    # --- Loop breakers ---
    if "name" in miss and asked_name_count >= 1:
        data["name"] = "Unknown"
        miss.remove("name")

    # --- Phone loop breaker ---
    if "phone" in miss:
        # If they said "use my caller ID", trust it once
        if "caller id" in utter.lower():
            data["phone"] = _only_digits(request.values.get("From", ""))
            miss.remove("phone")
        # If they spoke any digits (7+), accept it
        elif any(ch.isdigit() for ch in utter) or maybe_find_phone(utter):
            spoken = _only_digits(utter)
            if spoken and len(spoken) >= 7:
                data["phone"] = spoken
                miss.remove("phone")
        # If still empty after already one ask, fallback to caller ID automatically
        elif asked_phone_count >= 1:
            cid = _only_digits(request.values.get("From", ""))
            if cid:
                data["phone"] = cid
                miss.remove("phone")

    # If complete -> save
    if not miss:
        try:
            log_to_sheets(
                (data.get("role") or "volunteer").title(),
                data.get("name") or "Unknown",
                data.get("phone") or "",
                data.get("notes") or original or utter,
            )
        except Exception as e:
            print("save error:", e)
            return internal_error_response("Sorry, we had trouble saving your information.")
        vr = VoiceResponse()
        vr.say(f"Thank you {data['name']}. Your {data['role']} information has been recorded. Goodbye from HungerCue.")
        return Response(str(vr), mimetype="text/xml")

    # Otherwise ask the next best question; increment counters
    next_field = "role" if "role" in miss else ("name" if "name" in miss else "phone")
    if next_field == "name":
        asked_name_count += 1
    elif next_field == "phone":
        asked_phone_count += 1

    ctx = {"known": data, "original": original, "history": [f"Caller said: {utter[:160]}"]}
    q = clarifying_question(ctx)

    vr = gather_speech(
        q,
        next_action(
            "clarify",
            role=data.get("role", ""),
            name=data.get("name", ""),
            phone=data.get("phone", ""),
            notes=data.get("notes", ""),
            original=original or "initial message omitted",
            asked_name_count=asked_name_count,
            asked_phone_count=asked_phone_count,
        ),
    )
    return Response(str(vr), mimetype="text/xml")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print("✅ HungerCue Voice AI (GPT Clarify) starting on port", port)
    app.run(port=port, debug=True)
