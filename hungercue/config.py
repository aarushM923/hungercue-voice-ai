"""Central configuration.

Every setting is read from environment variables (or the .env file) by
pydantic-settings. Field names map to upper-case env vars, e.g.
`openai_api_key` <- OPENAI_API_KEY. Keeping all settings in one typed class
means a typo in a setting name fails loudly at startup instead of silently
returning None deep inside a phone call.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Server ---
    public_base_url: str = ""  # e.g. https://hungercue.fly.dev (no trailing slash)
    port: int = 8000

    # --- Twilio ---
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = "+18889583920"  # HungerCue's toll-free number (calls + SMS)
    # Texting from a toll-free number requires Twilio toll-free verification. Until
    # it's approved, SMS stays off: messages are logged as "pending verification"
    # in the database (and visible on the dashboard) but not sent.
    sms_enabled: bool = False
    validate_twilio_signature: bool = True  # reject webhook requests not signed by Twilio
    coordinator_phone: str = ""  # where transfer_to_human() forwards the caller

    # --- AI providers ---
    openai_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    deepgram_api_key: str = ""
    stt_model: str = "nova-3"
    tts_voice_en: str = "aura-2-thalia-en"
    tts_voice_es: str = "aura-2-celeste-es"

    # --- Storage ---
    # SQLite for local dev and tests; set to postgresql+asyncpg://... in production.
    database_url: str = "sqlite+aiosqlite:///./hungercue.db"
    # Empty means "use an in-memory fake Redis" (fine for dev and tests, not for production).
    redis_url: str = ""
    recordings_dir: str = "recordings"
    audio_retention_days: int = 30

    # --- Google Sheets (coordinator log, carried over from v1) ---
    google_credentials_json: str = "google-credentials.json"
    google_sheet_id: str = ""
    google_sheet_name: str = "HungerCue Calls"
    role_tabs: bool = True
    sheets_enabled: bool = True

    # --- Dashboard ---
    dashboard_user: str = "coordinator"
    dashboard_password: str = ""  # empty disables the dashboard entirely (safe default)

    # --- Call behaviour ---
    echo_mode: bool = False  # milestone-1 mode: play the caller's audio straight back
    # Endpointing: how long a pause counts as "the caller finished". Shorter = faster
    # replies but more risk of cutting people off mid-thought. Tuned with the eval suite.
    stt_endpointing_ms: int = 200
    vad_min_silence_ms: int = 224
    vad_min_speech_ms: int = 96  # speech needed before a barge-in counts (ignores coughs, clicks)
    timezone: str = "America/New_York"

    # --- Matching weights: score = w1*fit + w2*overlap - w3*dist(d,v) - w4*dist(v,s) ---
    match_w_fit: float = 3.0
    match_w_overlap: float = 1.0  # per hour of overlap
    match_w_dist_dv: float = 0.05  # per minute of volunteer travel to the restaurant
    match_w_dist_vs: float = 0.05  # per minute of travel from restaurant to shelter

    # --- Follow-up timing ---
    volunteer_reply_timeout_min: int = 15
    reminder_lead_min: int = 30


@lru_cache
def get_settings() -> Settings:
    """Return one shared Settings object (lru_cache makes it a cheap singleton)."""
    return Settings()
