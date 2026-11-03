"""FastAPI application: the HTTP and WebSocket entry points.

Run locally with:  .venv/bin/uvicorn hungercue.main:app --reload

Why FastAPI instead of Flask: a phone call is a long-lived WebSocket where we
must listen, transcribe, think and speak at the same time. FastAPI runs on
asyncio, so one process can juggle many of those concurrent tasks without a
thread per call.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from hungercue.config import Settings, get_settings
from hungercue.telephony import twilio_stream as ts

log = logging.getLogger("hungercue")

RECORDING_NOTICE = (
    "Thanks for calling HungerCue. This call is recorded to help coordinate food pickups. "
    "Para español, simplemente hable en español."
)


def ws_url_for(settings: Settings, request: Request) -> str:
    """The wss:// URL Twilio should open for the media stream."""
    base = settings.public_base_url or str(request.base_url).rstrip("/")
    # Twilio requires a secure WebSocket, so http(s) becomes ws(s).
    return base.replace("https://", "wss://").replace("http://", "ws://") + "/media"


async def verify_twilio(request: Request, settings: Settings) -> dict[str, str]:
    """Read the form body and reject the request if Twilio's signature is wrong."""
    form = {k: str(v) for k, v in (await request.form()).items()}
    if settings.validate_twilio_signature:
        # Behind a proxy (Fly, cloudflared) request.url may say http://, so we
        # rebuild the URL Twilio actually called from our public base URL.
        url = str(request.url)
        if settings.public_base_url:
            url = settings.public_base_url + request.url.path
            if request.url.query:
                url += "?" + request.url.query
        sig = request.headers.get("X-Twilio-Signature", "")
        if not ts.is_valid_twilio_request(settings.twilio_auth_token, url, form, sig):
            raise HTTPException(status_code=403, detail="invalid Twilio signature")
    return form


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests pass their own Settings; production uses .env."""
    settings = settings or get_settings()
    app = FastAPI(title="HungerCue v2")
    app.state.settings = settings

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/voice")
    async def voice(request: Request) -> Response:
        """Twilio calls this when the phone rings. We answer with streaming TwiML."""
        form = await verify_twilio(request, settings)
        twiml = ts.stream_twiml(ws_url_for(settings, request), form.get("From", ""), RECORDING_NOTICE)
        return Response(content=twiml, media_type="text/xml")

    @app.websocket("/media")
    async def media(websocket: WebSocket) -> None:
        """Twilio's live audio stream for one call."""
        await websocket.accept()
        try:
            await echo_loop(websocket)
        except WebSocketDisconnect:
            log.info("media stream disconnected")

    return app


async def echo_loop(websocket: WebSocket) -> None:
    """Milestone-1 behaviour: send every inbound audio chunk straight back.

    This proves the round trip works and measures the transport-only baseline
    latency before any AI is involved.
    """
    stream_sid = ""
    while True:
        event = ts.parse_message(await websocket.receive_text())
        if isinstance(event, ts.StartEvent):
            stream_sid = event.stream_sid
            log.info("call started: %s", event.call_sid)
        elif isinstance(event, ts.MediaEvent) and stream_sid:
            await websocket.send_text(ts.media_message(stream_sid, event.payload))
        elif isinstance(event, ts.StopEvent):
            return


app = create_app()
