import os, json, re, traceback
from typing import Dict, Any, List
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# --- OpenAI setup ---
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    NEW_SDK = True
except ImportError:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai
    NEW_SDK = False


# -----------------------------
# Helpers
# -----------------------------
def spoken_to_digits(text: str) -> str:
    """Convert spoken numbers like 'seven three two' → '732'."""
    words = {
        "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
    }
    tokens = text.lower().split()
    digits = "".join(words.get(tok, "") for tok in tokens)
    return digits if len(digits) >= 7 else ""


# -----------------------------
# Extraction Logic
# -----------------------------
def extract_structured(transcript: str) -> Dict[str, str]:
    """Extracts name, role, phone, and notes from the caller's message."""
    if not transcript:
        return {"role": "", "name": "", "phone": "", "notes": ""}

    # Try to detect if caller said to use their caller ID
    if "use my caller id" in transcript.lower():
        return {"role": "", "name": "", "phone": "CALLER_ID", "notes": transcript.strip()}

    # Regex fallback for written or spoken numbers
    phone_match = re.search(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', transcript)
    phone = phone_match.group(0) if phone_match else spoken_to_digits(transcript)

    prompt = f"""
You are HungerCue's AI assistant. Extract caller information from this spoken message.

Return JSON with:
- "name": caller’s name (if missing, leave blank)
- "role": one of ["restaurant", "shelter", "volunteer"]
- "phone": phone number (digits only, or blank)
- "notes": donation or message details

If role is implied, infer it.
If the phone number is not spoken, leave it blank — we'll use caller ID.
Input: "{transcript}"
"""

    try:
        if NEW_SDK:
            completion = client.responses.create(
                model="gpt-4o-mini",
                input=[{"role": "user", "content": prompt}],
            )
            parsed = completion.output[0].content[0].text.strip()
        else:
            resp = client.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            parsed = resp.choices[0].message.content.strip()

        try:
            data = json.loads(parsed)
        except Exception:
            json_match = re.search(r"\{.*\}", parsed, re.DOTALL)
            data = json.loads(json_match.group(0)) if json_match else {}

    except Exception as e:
        print("❌ extract_structured error:", e)
        traceback.print_exc()
        data = {}

    result = {
        "name": data.get("name", "").strip(),
        "role": data.get("role", "").lower().strip(),
        "phone": data.get("phone", phone).strip(),
        "notes": data.get("notes", transcript.strip()),
    }

    # Normalize role
    if "rest" in result["role"]:
        result["role"] = "restaurant"
    elif "shel" in result["role"]:
        result["role"] = "shelter"
    elif "vol" in result["role"]:
        result["role"] = "volunteer"

    # Infer role heuristically
    if not result["role"]:
        t = transcript.lower()
        if any(w in t for w in ["donate", "bagel", "panera", "starbucks", "dunkin"]):
            result["role"] = "restaurant"
        elif any(w in t for w in ["shelter", "home", "center"]):
            result["role"] = "shelter"
        elif any(w in t for w in ["help", "volunteer", "collect"]):
            result["role"] = "volunteer"

    print("✅ Extracted structured data:", result)
    return result


# -----------------------------
# Missing Fields Logic
# -----------------------------
def missing_fields(data: Dict[str, str]) -> List[str]:
    """Identify which fields need clarification."""
    missing = []
    if not data.get("role"):
        missing.append("role")
    if not data.get("name"):
        missing.append("name")
    if not data.get("phone"):
        missing.append("phone")
    return missing


# -----------------------------
# Clarifying Question Logic
# -----------------------------
def clarifying_question(context: Dict[str, Any]) -> str:
    """
    Ask one clarifying question based on missing data.
    Handles phone and name loop prevention.
    """
    known = context.get("known", {})
    history = " ".join(context.get("history", []))
    missing = missing_fields(known)

    # Loop prevention: if we already asked for the name twice
    if "name" in missing and history.lower().count("name") > 1:
        known["name"] = "Unknown"
        missing.remove("name")

    # Phone handling
    if "phone" in missing:
        # If caller said "use my caller ID" previously, accept that
        if "caller id" in history.lower():
            known["phone"] = "CALLER_ID"
            missing.remove("phone")
        else:
            return "I didn’t catch your phone number. Please say your number slowly, or say 'use my caller ID'."

    if "role" in missing:
        return "Are you a restaurant, shelter, or volunteer?"

    if "name" in missing:
        return "May I have your name, please?"

    return "Could you please repeat that?"
