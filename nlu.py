import os, json, traceback
from typing import Dict, Any, List
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_INSTRUCTIONS = """
You are the HungerCue Voice AI NLU.
Extract a JSON object with:
- role: one of "Volunteer", "Restaurant", "Shelter"
- name: caller's name if present
- phone: any phone number mentioned (digits or formatted)
- notes: all other details (address, items, quantity, time)
Return ONLY valid JSON.
"""

def extract_structured(transcript: str) -> Dict[str, str]:
    if not transcript:
        return {"role": "", "name": "", "phone": "", "notes": ""}
    try:
        completion = client.responses.create(
            model="gpt-4.1-mini",
            input=[{"role": "user", "content": BASE_INSTRUCTIONS + "\nCaller said:\n" + transcript}],
        )
        text = completion.output[0].content[0].text.strip()
        return json.loads(text)
    except Exception as e:
        print("❌ extract_structured error:", e)
        traceback.print_exc()
        return {"role": "", "name": "", "phone": "", "notes": ""}


def missing_fields(data: Dict[str, str]) -> List[str]:
    return [k for k in ("role", "name", "phone") if not data.get(k)]


def clarifying_question(context: Dict[str, Any]) -> str:
    prompt = f"""
Given known fields: {context.get('known')}
Ask ONE question to clarify missing info.
Rules:
- If role missing → ask if they're a restaurant, shelter, or volunteer.
- If phone missing → ask if you can use caller ID or ask for digits.
- If name missing → ask for name once.
Return only the question text.
"""
    try:
        res = client.responses.create(
            model="gpt-4.1-mini",
            input=[{"role": "user", "content": prompt}],
        )
        return res.output[0].content[0].text.strip()
    except Exception as e:
        print("❌ clarifying_question error:", e)
        return "Could you please clarify your name or role?"
