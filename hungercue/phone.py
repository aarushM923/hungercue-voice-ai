"""Phone number helpers.

All numbers are stored in E.164 format (+1 then 10 digits for US numbers).
Twilio sends caller IDs as E.164 already, but people *say* numbers in many
shapes ("212 555 0100", "(212) 555-0100"), so we normalise before saving or
looking up. Otherwise a returning caller would not be recognised.
"""

from __future__ import annotations

import re

# Carried over from v1 utils.maybe_find_phone: optional +1, then 3-3-4 digits
# with optional brackets, dots, dashes or spaces between groups.
PHONE_RE = re.compile(r"(?:\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})")


def normalize_phone(raw: str) -> str:
    """Return +1XXXXXXXXXX for a US number, '+<digits>' for others, '' if unusable."""
    digits = "".join(c for c in (raw or "") if c.isdigit())
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    if raw.strip().startswith("+") and 8 <= len(digits) <= 15:
        return "+" + digits  # international number, already has a country code
    return ""


def find_phone(text: str) -> str:
    """Find the first phone number in free text and normalise it ('' if none)."""
    m = PHONE_RE.search(text or "")
    return normalize_phone(m.group(1)) if m else ""
