# Production image for HungerCue v2 (used by Fly.io; not deployed yet).
FROM python:3.11-slim

# Don't write .pyc files; print logs immediately (important for watching live calls).
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hungercue ./hungercue

# Run as a normal user, not root.
RUN useradd --create-home app && mkdir -p /data/recordings && chown -R app /data /app
USER app

EXPOSE 8080
# --proxy-headers: trust Fly's proxy so URLs are seen as https (Twilio signature checks depend on it).
CMD ["uvicorn", "hungercue.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
