# ============================================================
#  Prometix by Atimos AI — Flask Backend  v3.0
#
#  Run:  python app.py
#  Requires: pip install flask flask-cors requests werkzeug
#
#  AI backend : Multi-model AI system
#  User store : SQLite  — persistent across restarts / deploys
#
#  Endpoints:
#    GET  /health              — backend ping
#    POST /auth/register       — create account (hashed password)
#    POST /auth/login          — authenticate + issue session token
#    POST /auth/logout         — invalidate session token
#    POST /generate            — rewrite raw idea → prompt (protected)
#    POST /user/consent        — save training consent flag (protected)
#    POST /user/search         — log search query (protected)
#    POST /training/pair       — store training pair (protected)
# ============================================================

import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("prometix")
import uuid
import datetime
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from flask import Flask, request, jsonify, redirect
from time import time
from collections import defaultdict
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth
import requests
import json
import psycopg2
# google genai removed — Gemini routed through Pollinations
import re
from functools import wraps

# ── Prompt dataset (Hybrid RAG) ─────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "prompt_dataset.json")
try:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        PROMPT_DATA = json.load(f)
except Exception:
    PROMPT_DATA = []
import smtplib
from email.mime.text import MIMEText

# ── Image request helper ─────────────────────────────────────
def is_image_request(text: str) -> bool:
    t = text.lower()
    keywords = ["image", "photo", "picture", "generate image", "create image", "art", "illustration", "logo"]
    return any(k in t for k in keywords)

# ── Audio request helper ────────────────────────────────────
def is_audio_request(text: str) -> bool:
    t = text.lower()

    keywords = [
        "audio",
        "voice",
        "speech",
        "tts",
        "text to speech",
        "music",
        "song",
        "sound effect",
        "podcast",
        "narration",
        "generate audio",
        "create audio"
    ]

    return any(k in t for k in keywords)

# ── Pollinations image generator ─────────────────────────────
def generate_image(prompt: str, model: str = "gptimage") -> str:
    """
    Generate image using Pollinations API and return base64.
    """

    import urllib.parse
    import base64

    encoded = urllib.parse.quote(prompt)

    # ── Pollinations API Key ─────────────────────────
    pollinations_key = os.environ.get("POLLINATIONS_API_KEY")

    headers = {}

    if pollinations_key:
        headers["Authorization"] = f"Bearer {pollinations_key}"

    # ── Model Routing ────────────────────────────────
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?model={model}&width=1024&height=1024"
        f"&enhance=true&safe=true&nologo=true"
    )

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=40
        )

        if response.status_code == 200:
            from PIL import Image, ImageDraw, ImageFont
            from io import BytesIO

            img = Image.open(BytesIO(response.content)).convert("RGBA")

            # ── Logo Watermark (Prometix) ─────────────────────
            from PIL import ImageEnhance

            try:
                logo_path = os.path.join(os.path.dirname(__file__), "watermark.png")
                logo = Image.open(logo_path).convert("RGBA")

                base_width = int(img.width * 0.15)
                ratio = base_width / logo.width
                new_size = (base_width, int(logo.height * ratio))
                logo = logo.resize(new_size, Image.LANCZOS)

                alpha = logo.split()[3]
                alpha = ImageEnhance.Brightness(alpha).enhance(0.5)
                logo.putalpha(alpha)

                margin = 20
                position = (margin, img.height - logo.height - margin)

                img.paste(logo, position, logo)

            except Exception as e:
                logger.warning(f"Watermark error: {e}")

            buffered = BytesIO()
            img.save(buffered, format="PNG")

            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        logger.warning(
            f"Pollinations image request failed | status={response.status_code}"
        )

    except Exception:
        logger.exception("Image generation failed")

    return None

# ── Pollinations audio generator ───────────────────────────
def generate_audio(prompt: str, model: str = "elevenlabs"):
    """
    Generate audio/music using Pollinations audio models.
    Returns audio URL if successful.
    """

    headers = {
        "Content-Type": "application/json"
    }

    if POLLINATIONS_API_KEY:
        headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"

    payload = {
        "model": model,
        "prompt": prompt
    }

    try:
        response = requests.post(
            "https://audio.pollinations.ai/generate",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            logger.warning(
                f"Pollinations audio failed | model={model} | status={response.status_code}"
            )
            return None

        result = response.json()

        audio_url = (
            result.get("audio")
            or result.get("url")
            or result.get("audio_url")
        )

        if not audio_url:
            logger.warning(
                f"Pollinations audio returned empty URL | model={model}"
            )
            return None

        return audio_url

    except Exception:
        logger.exception("Audio generation failed")
        return None


def get_client_key(email=None):
    ip = request.remote_addr or "unknown"
    ua = request.headers.get("User-Agent", "unknown")
    accept = request.headers.get("Accept", "unknown")
    lang = request.headers.get("Accept-Language", "unknown")

    # persistent device cookie
    device_id = request.cookies.get("device_id")

    base = f"{ip}:{ua}:{accept}:{lang}"

    # include email if logged in (stronger binding)
    if email:
        base += f":{email}"

    import hashlib
    fingerprint = hashlib.sha256(base.encode()).hexdigest()

    return device_id or fingerprint

# ── VPN / Proxy Detection ─────────────────────────────
def is_vpn_or_proxy():
    """
    Lightweight abuse protection.
    Blocks malformed forwarded IP chains and suspicious proxy headers.
    Keeps Render/Netlify/CDN traffic compatible.
    """

    ip = request.remote_addr or ""

    # Allow localhost/private development
    private_prefixes = (
        "127.",
        "10.",
        "172.",
        "192.168"
    )

    if ip.startswith(private_prefixes):
        return False

    forwarded = request.headers.get("X-Forwarded-For", "")

    # Suspiciously large forwarded chain
    if len(forwarded) > 200:
        return True

    suspicious_headers = [
        "Via",
        "X-Proxy-ID",
        "X-Forwarded-Host"
    ]

    for header in suspicious_headers:
        value = request.headers.get(header)

        if value and len(value) > 120:
            return True

    # Malformed IPv4 detection
    if ip and ip.count(".") != 3:
        return True

    return False

# ── In-memory rate limiting ─────────────────────────────

rate_limit_store = defaultdict(list)
last_rate_cleanup = 0
RATE_LIMIT_CLEANUP_INTERVAL = 60 * 30  # every 30 minutes

RATE_LIMITS = {
    "guest": {
        "window": 60 * 60,  # 1 hour
        "limit": 5          # 5 requests per hour for guests
    },
    "auth": {
        "window": 60 * 60 * 5,  # 5 hours
        "limit": 15             # 15 requests per 5 hours for logged-in users
    },
    "image": {
        "window": 60 * 10,
        "limit": 10
    },
    "password_reset": {
        "window": 60 * 15,
        "limit": 5
    }
}



MAX_PROMPT_LENGTH = 8000
SESSION_EXPIRY_DAYS = 7

# ── User data validation limits ─────────────────────────────
MAX_NAME_LENGTH = 80
MAX_EMAIL_LENGTH = 254
PASSWORD_MIN_LENGTH = 8

# ── Basic moderation filters ──────────────────────────
BLOCKED_PATTERNS = [
    "child porn",
    "cp content",
    "extremist manifesto",
    "terrorist propaganda",
    "make a bomb",
    "credit card dump",
    "steal passwords",
    "malware builder",
    "ransomware code"
]


# ─────────────────────────────────────────────────────────────
def sanitize_text(value: str, max_length: int = 5000) -> str:
    if not isinstance(value, str):
        return ""

    value = value.strip()

    # Remove dangerous control characters
    value = re.sub(r"[\x00-\x1f\x7f]", "", value)

    return value[:max_length]


def is_valid_email(email: str) -> bool:
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return bool(re.fullmatch(pattern, email or ""))
# ── Flask App Initialization and CORS ────────────────────────
app = Flask(__name__)

# ── Request size protection ───────────────────────────
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

SECRET_KEY = os.environ.get("SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY environment variable is required")

app.secret_key = SECRET_KEY
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=datetime.timedelta(days=7)
)

frontend_origin = os.environ.get("FRONTEND_URL")

if not frontend_origin:
    raise RuntimeError("FRONTEND_URL environment variable is required")

# Support optional test origin (e.g. random Netlify preview URL)
frontend_origin_test = os.environ.get("FRONTEND_URL_TEST", "")

allowed_origins = [o for o in [frontend_origin, frontend_origin_test] if o]

CORS(
    app,
    resources={
        r"/*": {
            "origins": allowed_origins
        }
    },
    supports_credentials=True
)

# ── Google OAuth setup ───────────────────────────────────────
oauth = OAuth(app)

google = oauth.register(
    name='google',
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
    api_base_url='https://openidconnect.googleapis.com/v1/',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

# ── Config ────────────────────────────────────────────────────
# ── Config ────────────────────────────────────────────────────
# ── Config ────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# DeepSeek routed through Pollinations — no separate key needed
POLLINATIONS_API_KEY = os.environ.get("POLLINATIONS_API_KEY")
# Internal AI routing models
_GROQ_MODELS    = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
]
REQUEST_TIMEOUT = int(os.environ.get("PROMETIX_TIMEOUT", "30"))

# SMTP/Email configuration
SMTP_SERVER = "smtp.zoho.in"
SMTP_PORT = 587
EMAIL_ADDRESS = "admin@atimosai.com"
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

# ── Required environment validation ───────────────────
 # ── Database Routing ──────────────────────────────────
# Supabase → authentication + users
# Neon → AI chats + memory + generations

SUPABASE_DATABASE_URL = os.environ.get("DATABASE_URL")
NEON_DATABASE_URL = os.environ.get("NEON_DATABASE_URL")

REQUIRED_ENV_VARS = {
    "SUPABASE_DATABASE_URL": SUPABASE_DATABASE_URL,
    "NEON_DATABASE_URL": NEON_DATABASE_URL,
    "SECRET_KEY": os.environ.get("SECRET_KEY"),
    "EMAIL_PASSWORD": EMAIL_PASSWORD,
    "GOOGLE_CLIENT_ID": os.environ.get("GOOGLE_CLIENT_ID"),
    "GOOGLE_CLIENT_SECRET": os.environ.get("GOOGLE_CLIENT_SECRET")
}

missing_env = [
    key for key, value in REQUIRED_ENV_VARS.items()
    if not value
]

if missing_env:
    missing_text = ", ".join(missing_env)
    logger.critical(f"Missing required environment variables: {missing_text}")
    raise RuntimeError(
        f"Missing required environment variables: {missing_text}"
    )
# Gemini is now fully routed through Pollinations (gemini-fast model)

def send_email(to_email, subject, message):
    def _send():
        try:
            msg = MIMEText(message)
            msg["Subject"] = subject
            msg["From"] = EMAIL_ADDRESS
            msg["To"] = to_email

            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            logger.info(f"Email sent successfully to {to_email}")
        except Exception as e:
            logger.warning(
                f"Email delivery failed | recipient={to_email} | error={e}"
            )

    email_thread = threading.Thread(target=_send, daemon=True)
    email_thread.start()


# ── PostgreSQL setup ──────────────────────────────────────────────
# Supabase connection → auth/users/sessions
# Neon connection → chats/history/generations

def _get_auth_conn():
    if not SUPABASE_DATABASE_URL:
        raise Exception("SUPABASE DATABASE_URL environment variable not set")

    return psycopg2.connect(
        SUPABASE_DATABASE_URL,
        sslmode="require",
        connect_timeout=10
    )


def _get_chat_conn():
    if not NEON_DATABASE_URL:
        raise Exception("NEON_DATABASE_URL environment variable not set")

    return psycopg2.connect(
        NEON_DATABASE_URL,
        sslmode="require",
        connect_timeout=10
    )


@contextmanager
def auth_db():
    conn = _get_auth_conn()

    try:
        cur = conn.cursor()
        yield cur
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


@contextmanager
def chat_db():
    conn = _get_chat_conn()

    try:
        cur = conn.cursor()
        yield cur
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

# ── Token helpers ─────────────────────────────────────────────
def _issue_token(email: str) -> str:
    token = uuid.uuid4().hex
    ts    = datetime.datetime.utcnow().isoformat()
    expires_at = (
        datetime.datetime.utcnow() +
        datetime.timedelta(days=SESSION_EXPIRY_DAYS)
    ).isoformat()
    with auth_db() as cur:
        cur.execute(
            """
            INSERT INTO sessions (token, email, created_at, expires_at)
            VALUES (%s, %s, %s, %s)
            """,
            (token, email, ts, expires_at)
        )
    return token

def _validate_token():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None, (jsonify({"error": "Session expired. Please sign in again.", "code": "TOKEN_MISSING"}), 401)
    token = auth_header[len("Bearer "):].strip()

    # Basic token validation
    if len(token) < 20:
        return None, (
            jsonify({
                "error": "Invalid session token.",
                "code": "TOKEN_INVALID"
            }),
            401
        )
    with auth_db() as cur:
        cur.execute(
            "SELECT email, created_at, expires_at FROM sessions WHERE token = %s",
            (token,)
        )
        row = cur.fetchone()
    if not row:
        return None, (jsonify({"error": "Session expired. Please sign in again.", "code": "TOKEN_INVALID"}), 401)

    try:
        created_at = row[1]
        expires_at = row[2]

        if created_at:
            created_dt = datetime.datetime.fromisoformat(str(created_at))

            if (datetime.datetime.utcnow() - created_dt).days >= 30:
                with auth_db() as cur:
                    cur.execute(
                        "DELETE FROM sessions WHERE token = %s",
                        (token,)
                    )

                return None, (
                    jsonify({
                        "error": "Session expired. Please sign in again.",
                        "code": "SESSION_EXPIRED"
                    }),
                    401
                )

        if expires_at:
            expiry_dt = datetime.datetime.fromisoformat(str(expires_at))

            if datetime.datetime.utcnow() > expiry_dt:
                with auth_db() as cur:
                    cur.execute(
                        "DELETE FROM sessions WHERE token = %s",
                        (token,)
                    )

                return None, (
                    jsonify({
                        "error": "Session expired. Please sign in again.",
                        "code": "SESSION_EXPIRED"
                    }),
                    401
                )

    except Exception as e:
        logger.warning(f"Session expiry validation failed: {e}")

    # ── Rolling session refresh ─────────────────────────
    try:
        new_expiry = (
            datetime.datetime.utcnow() +
            datetime.timedelta(days=SESSION_EXPIRY_DAYS)
        ).isoformat()

        with auth_db() as cur:
            cur.execute(
                "UPDATE sessions SET expires_at = %s WHERE token = %s",
                (new_expiry, token)
            )

    except Exception as e:
        logger.warning(f"Session refresh failed: {e}")
    return row[0], None

# ── Admin required decorator ─────────────────────────────
def admin_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        email, err = _validate_token()
        if err:
            return err
        with auth_db() as cur:
            cur.execute("SELECT role FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
        if not user or user[0] != "admin":
            logger.warning(
                f"Unauthorized admin access attempt | email={email} | ip={request.remote_addr}"
            )
            return jsonify({
                "error": "Forbidden",
                "code": "ADMIN_FORBIDDEN"
            }), 403
        logger.info(
            f"Admin access granted | email={email} | path={request.path}"
        )
        return f(*args, **kwargs)
    return wrapper

def _invalidate_token(token: str):
    with auth_db() as cur:
        cur.execute("DELETE FROM sessions WHERE token = %s", (token,))

def _public_profile(row) -> dict:
    return {
        "email": row[0],
        "name": row[1],
        "avatar": row[3] or row[1][0].upper(),
        "provider": row[4],
        "consent": bool(row[5]),
        "created_at": row[7],
        "is_admin": row[9] == "admin" if len(row) > 9 else False
    }

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are Prometix, an expert prompt engineer built by Atimos AI.

YOUR ONLY JOB:
Rewrite the user's input into a clean, high-quality, structured prompt that another AI can execute perfectly.

STRICT OUTPUT RULES:
- Output ONLY the final prompt text.
- DO NOT include any introduction, explanation, or commentary.
- DO NOT write phrases like "Here is your prompt", "Rewritten Prompt", or anything similar.
- DO NOT include labels, headings, or prefixes.
- DO NOT use bullet points or numbered lists.
- Start directly with the prompt content.
- End cleanly without extra lines or symbols.

PROMPT QUALITY RULES:
- Preserve the user's intent exactly.
- Improve clarity, specificity, and usefulness.
- Add relevant context, constraints, tone, and format instructions.
- Ensure the prompt is immediately usable in AI tools.
- Keep it concise (1-4 sentences unless absolutely needed).

The response must be ready to copy-paste directly into any AI tool.
"""

def detect_intent(text):
    text = text.lower()
    if any(w in text for w in ["code", "python", "function", "api", "program", "script"]):
        return "coding"
    elif any(w in text for w in ["reel", "youtube", "video", "content", "caption", "thumbnail"]):
        return "content"
    elif any(w in text for w in ["explain", "what is", "meaning", "define", "why", "how"]):
        return "explanation"
    elif any(w in text for w in ["story", "creative", "poem", "script", "write"]):
        return "creative"
    return "general"

def get_prompt_style(intent, user_message):
    length = len(user_message.split())
    styles = {
        "coding":      "Use instruction-based prompting. Be precise, technical, solution-focused.",
        "content":     "Use role-based creative prompting. Focus on engagement, hooks, and audience psychology.",
        "explanation": "Use clear, step-by-step explanation style. Break concepts into simple parts.",
        "creative":    "Use imaginative prompting. Encourage storytelling, originality, and emotional depth.",
        "general":     "Use structured and balanced prompting. Ensure clarity, context, and usefulness.",
    }
    base   = styles.get(intent, styles["general"])
    detail = "Keep it simple and direct." if length < 6 else "Expand with better detail and structure."
    return f"{base} {detail}"

# ── Simple retrieval (Hybrid RAG) ───────────────────────────
def retrieve_context(user_input):
    user_input = user_input.lower()
    results = []

    for item in PROMPT_DATA:
        inp = item.get("input", "").lower()
        if any(word in inp for word in user_input.split()):
            results.append(item.get("output", ""))

    return "\n\n".join(results[:2])

def _call_single_model(model: str, messages: list) -> str:
    """Call one Groq model. Returns the response text or raises on error."""
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={
            "model":       model,
            "messages":    messages,
            "temperature": 0.4,
            "max_tokens":  512,
        },
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code}: {response.text}")
    result = response.json()
    if "choices" not in result or not result["choices"]:
        raise Exception("No choices in response")
    text = result["choices"][0]["message"]["content"].strip()
    if not text:
        raise Exception("Empty content in response")
    return text


def call_groq(user_message: str) -> str:
    """
    Race both models simultaneously. Return the first successful response.
    If the winner fails or returns empty, fall back to the other model's result.
    If both fail, raise the last error.
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY missing — using Gemini-only pipeline")
        return user_message

    intent       = detect_intent(user_message)
    style        = get_prompt_style(intent, user_message)
    length       = len(user_message.split())
    context = retrieve_context(user_message)

    user_content = (
        f"STYLE: {style}\n"
        f"COMPLEXITY: {'simple' if length < 6 else 'advanced'}\n\n"
        f"Raw idea:\n{user_message}\n\n"
        f"Relevant Prompt Engineering Context:\n{context}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    last_error = None
    with ThreadPoolExecutor(max_workers=len(_GROQ_MODELS)) as executor:
        futures = {
            executor.submit(_call_single_model, model, messages): model
            for model in _GROQ_MODELS
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                # Cancel remaining futures (best-effort)
                for f in futures:
                    if f is not future:
                        f.cancel()
                return result
            except Exception as e:
                last_error = e
                continue  # Try next model to finish

    raise Exception(f"All models failed. Last error: {last_error}")


# ── Gemini via Pollinations ──────────────────────────────────
# Real Gemini API replaced by Pollinations (gemini-fast = Gemini 2.5 Flash Lite).
# model_choice kept for frontend compatibility; all tiers route through Pollinations.
def call_gemini(prompt: str, model_choice: str = "2.5-flash"):
    """
    Routes all Gemini-mode requests through Pollinations.
    Primary: gemini-fast. Fallback chain: gpt-5.5 → mistral.
    """
    gemini_chain = ["gemini-fast", "gpt-5.5", "mistral"]

    headers = {"Content-Type": "application/json"}
    if POLLINATIONS_API_KEY:
        headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"

    attempted_models = []
    last_error = None

    for idx, model_name in enumerate(gemini_chain):
        attempted_models.append(model_name)
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "private": True,
            "seed": -1
        }
        try:
            response = requests.post(
                "https://text.pollinations.ai/openai",
                headers=headers,
                json=payload,
                timeout=50
            )
            if response.status_code != 200:
                logger.warning(
                    f"Pollinations gemini-chain failed | model={model_name} | status={response.status_code}"
                )
                last_error = f"HTTP {response.status_code}"
                continue

            result = response.json()
            text = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not text:
                logger.warning(f"Pollinations gemini-chain empty response | model={model_name}")
                last_error = "empty response"
                continue

            return {
                "text": text,
                "usedFallback": idx > 0,
                "fallbackModel": model_name if idx > 0 else None,
                "attemptedModels": attempted_models
            }

        except Exception as e:
            logger.warning(
                f"Pollinations gemini-chain exception | model={model_name} | error={e}"
            )
            last_error = str(e)
            continue

    logger.error(
        f"All Pollinations gemini-chain models failed | last_error={last_error}"
    )
    return {
        "text": "AI response temporarily unavailable. Please try again.",
        "usedFallback": True,
        "fallbackModel": None,
        "attemptedModels": attempted_models
    }


# ── Pollinations Text Function ──────────────────────────────
def call_pollinations_text(prompt: str, provider: str = "gpt"):
    """
    Multi-model Pollinations text routing with fallback chain.
    Uses correct model identifiers per Pollinations API v2.
    """

    # Correct model names per Pollinations API
    # Exact model IDs from Pollinations dashboard (May 2026)
    provider_map = {
        "claude":   "claude",        # Claude Sonnet 4.6
        "gpt":      "gpt-5.5",       # GPT-5.5
        "gemini":   "gemini-fast",   # Gemini 2.5 Flash Lite
        "deepseek": "deepseek",      # DeepSeek V4 Flash
        "qwen":     "qwen-coder-large", # Qwen3 Coder Next
        "grok":     "grok-large",    # Grok 4.20 Reasoning
        "mistral":  "mistral",       # Mistral Small 3.1
    }

    # Fallback chain: if selected model fails, try these in order
    fallback_chain = [
        provider_map.get(provider, "gpt-5.5"),
        "gpt-5.5",
        "mistral",
        "claude",
    ]
    # Remove duplicates while preserving order
    seen = set()
    fallback_chain = [x for x in fallback_chain if not (x in seen or seen.add(x))]

    headers = {"Content-Type": "application/json"}
    if POLLINATIONS_API_KEY:
        headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"

    last_error = None
    for selected_model in fallback_chain:
        payload = {
            "model": selected_model,
            "messages": [{"role": "user", "content": prompt}],
            "private": True,
            "seed": -1
        }

        try:
            response = requests.post(
                "https://text.pollinations.ai/openai",
                headers=headers,
                json=payload,
                timeout=50
            )

            if response.status_code != 200:
                logger.warning(
                    f"Pollinations text failed | model={selected_model} | status={response.status_code} | body={response.text[:200]}"
                )
                last_error = f"HTTP {response.status_code}"
                continue

            result = response.json()
            text = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if not text:
                logger.warning(f"Pollinations empty response | model={selected_model}")
                last_error = "empty response"
                continue

            # Success
            return {
                "text": text,
                "provider": provider,
                "model": selected_model,
                "success": True
            }

        except Exception as e:
            logger.warning(f"Pollinations exception | model={selected_model} | error={e}")
            last_error = str(e)
            continue

    # All models in chain failed
    logger.error(f"All Pollinations models failed | provider={provider} | last_error={last_error}")
    return {
        "text": "AI response temporarily unavailable. Please try Gemini mode instead.",
        "provider": provider,
        "model": fallback_chain[0],
        "success": False
    }

# ── Health ────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "product": "Prometix by Atimos AI"})

# ── Root Route (for uptime + browser access) ──────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "service": "Prometix Backend",
        "message": "Backend is live and working"
    })

# ── Generate ─────────────────────────────────────────────────
@app.route("/generate-x7k9A2", methods=["POST"])
def generate():
    email, err = _validate_token()
    # Allow guest users (no login required)
    if err:
        email = None

    client_key = get_client_key(email)
    now = time()

    # ── Periodic rate-limit cleanup ────────────────────
    global last_rate_cleanup

    if now - last_rate_cleanup > RATE_LIMIT_CLEANUP_INTERVAL:
        try:
            stale_keys = []

            for key, timestamps in rate_limit_store.items():
                fresh = [t for t in timestamps if now - t < 60 * 60 * 24]

                if fresh:
                    rate_limit_store[key] = fresh
                else:
                    stale_keys.append(key)

            for key in stale_keys:
                rate_limit_store.pop(key, None)

            last_rate_cleanup = now
            logger.info("Rate-limit store cleanup completed")

        except Exception as e:
            logger.warning(f"Rate-limit cleanup failed: {e}")

    # Ensure device_id cookie is always set (anti-incognito persistence)
    device_cookie = request.cookies.get("device_id")
    if not device_cookie:
        device_cookie = client_key

    # ── VPN / Proxy Block ─────────────────────────────
    if is_vpn_or_proxy():
        resp = jsonify({
            "error": "VPN or proxy detected. Please turn it off to use Prometix.",
            "code": "VPN_BLOCKED"
        })
        resp.set_cookie("device_id", device_cookie, max_age=60*60*24*30, httponly=True, samesite="Lax")
        return resp, 403

    # ── Guest usage limits ─────────────────────────────
    if not email:
        window = RATE_LIMITS["guest"]["window"]
        limit = RATE_LIMITS["guest"]["limit"]

        if client_key not in rate_limit_store:
            rate_limit_store[client_key] = []

        # remove old timestamps
        rate_limit_store[client_key] = [
            t for t in rate_limit_store[client_key] if now - t < window
        ]

        if len(rate_limit_store[client_key]) >= limit:
            resp = jsonify({
                "error": "Free preview completed. Login to continue using Prometix AI.",
                "code": "GUEST_LIMIT"
            })
            resp.set_cookie("device_id", device_cookie, max_age=60*60*24*30, httponly=True, samesite="Lax")
            return resp, 429

        rate_limit_store[client_key].append(now)

    # ── General request rate limiting ─────────────────────
    auth_limit = RATE_LIMITS["auth"]
    auth_rate_key = f"auth:{client_key}"

    if auth_rate_key not in rate_limit_store:
        rate_limit_store[auth_rate_key] = []

    rate_limit_store[auth_rate_key] = [
        t for t in rate_limit_store[auth_rate_key]
        if now - t < auth_limit["window"]
    ]

    if len(rate_limit_store[auth_rate_key]) >= auth_limit["limit"]:
        resp = jsonify({
            "error": "Too many requests. Please slow down.",
            "code": "RATE_LIMITED"
        })
        resp.set_cookie(
            "device_id",
            device_cookie,
            max_age=60*60*24*30,
            httponly=True,
            samesite="Lax"
        )
        return resp, 429

    rate_limit_store[auth_rate_key].append(now)

    data = request.get_json(silent=True)

    # ── Request type validation ────────────────────────
    if request.content_type and "application/json" not in request.content_type:
        return jsonify({
            "error": "Unsupported content type.",
            "code": "INVALID_CONTENT_TYPE"
        }), 415

    if not data:
        return jsonify({"error": "Invalid JSON in request body."}), 400

    user_message = sanitize_text(
        data.get("message") or "",
        MAX_PROMPT_LENGTH
    )

    # Normalize whitespace spam
    user_message = " ".join(user_message.split())
    image_mode = is_image_request(user_message)
    audio_mode = is_audio_request(user_message)
    model_choice = (data.get("model") or "2.5-flash").lower()
    mode = (data.get("mode") or "prompt").lower()

    pollinations_model = (
        data.get("pollinations_model") or "gptimage"
    ).lower()

    pollinations_audio_model = (
        data.get("pollinations_audio_model") or "elevenlabs-v3"
    ).lower()

    pollinations_provider = (
        data.get("pollinations_provider") or "gpt"
    ).lower()

    # ── Allowed AI mode validation ─────────────────────
    allowed_modes = {"prompt", "gemini", "pollinations"}

    if mode not in allowed_modes:
        return jsonify({
            "error": "Invalid AI mode selected.",
            "code": "INVALID_MODE"
        }), 400

    # ── Allowed audio models ───────────────────────────
    allowed_audio_models = {
        "elevenlabs-v3",
        "elevenlabs-v2",
        "elevenlabs-music"
    }

    if pollinations_audio_model not in allowed_audio_models:
        pollinations_audio_model = "elevenlabs-v3"

    # ── Allowed Pollinations providers ──────────────────
    allowed_pollinations_providers = {
        "claude",
        "gpt",
        "gemini",
        "deepseek",
        "qwen",
        "grok"
    }

    if pollinations_provider not in allowed_pollinations_providers:
        pollinations_provider = "gpt"

    # ── Allowed Gemini model validation ───────────────
    allowed_models = {
        "2.5-pro",
        "2.5-flash",
        "1.5-pro",
        "1.5-flash"
    }

    if model_choice not in allowed_models:
        model_choice = "2.5-flash"
    # ── Restrict premium AI features for guests ─────────────────────────
    if not email and mode in ["gemini", "pollinations"]:
        return jsonify({
            "error": "Login required to use AI features.",
            "code": "LOGIN_REQUIRED"
        }), 403
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    # ── Basic moderation filter ───────────────────────
    lowered_message = user_message.lower()

    for blocked in BLOCKED_PATTERNS:
        if blocked in lowered_message:
            logger.warning(
                f"Blocked unsafe prompt | ip={request.remote_addr} | pattern={blocked}"
            )

            return jsonify({
                "error": "This request violates usage policies.",
                "code": "PROMPT_BLOCKED"
            }), 403

    # Repeated-character spam protection
    if len(set(user_message)) <= 2 and len(user_message) > 30:
        return jsonify({
            "error": "Spam-like input detected.",
            "code": "SPAM_DETECTED"
        }), 400

    # ── Prompt length protection ─────────────────────────
    if len(user_message) > MAX_PROMPT_LENGTH:
        return jsonify({
            "error": "Prompt too large. Please shorten your message.",
            "code": "PROMPT_TOO_LARGE"
        }), 413

    # Gemini Pro limit: 5 requests per 5 hours per user
    if model_choice == "2.5-pro":
        window_seconds = 5 * 60 * 60
        now_ts = int(time())

        if email:
            with chat_db() as cur:
                cur.execute(
                    "SELECT ts FROM generations WHERE email = %s ORDER BY ts DESC LIMIT 20",
                    (email,)
                )
                rows = cur.fetchall()
        else:
            rows = []

        recent = []
        for r in rows:
            try:
                t = int(datetime.datetime.fromisoformat(r[0]).timestamp())
                if now_ts - t <= window_seconds:
                    recent.append(t)
            except:
                continue

        if len(recent) >= 5:
            return jsonify({
                "error": "Pro limit reached (5 requests in 5 hours). Switch to Fast mode.",
                "code": "PRO_LIMIT"
            }), 429

    try:
        # Step 1: Always create improved prompt
        improved_prompt = call_groq(user_message)

        # Step 2: If user wants only prompt
        if mode == "prompt":
            # Save generation history
            if email:
                try:
                    ts = datetime.datetime.utcnow().isoformat()

                    with chat_db() as cur:
                        cur.execute(
                            """
                            INSERT INTO generations (email, input, output, ts)
                            VALUES (%s, %s, %s, %s)
                            """,
                            (email, user_message, improved_prompt, ts)
                        )
                except Exception as e:
                    logger.warning(f"Failed to save generation history: {e}")
            resp = jsonify({
                "type": "prompt",
                "response": improved_prompt,
                "done": True
            })
            resp.set_cookie("device_id", device_cookie, max_age=60*60*24*30, httponly=True, samesite="Lax")
            return resp

        # Step 3A: Pollinations AI Mode
        if mode == "pollinations":

            # ── Audio Mode ─────────────────────────────
            if audio_mode:

                # Correct audio model IDs from Pollinations dashboard
                audio_model_map = {
                    "elevenlabs-v3": "elevenlabs",   # ElevenLabs v3 TTS
                    "elevenlabs-v2": "scribe",        # ElevenLabs Scribe v2
                    "elevenlabs-music": "elevenmusic" # ElevenLabs Music
                }
                mapped_audio = audio_model_map.get(pollinations_audio_model, "elevenlabs")
                audio_fallbacks = [
                    mapped_audio,
                    "elevenlabs",
                    "scribe",
                ]

                audio_url = None
                used_audio_model = None

                for fallback_model in audio_fallbacks:
                    audio_url = generate_audio(
                        improved_prompt,
                        fallback_model
                    )

                    if audio_url:
                        used_audio_model = fallback_model
                        break

                if not audio_url:
                    return jsonify({
                        "error": "Audio generation temporarily unavailable.",
                        "code": "AUDIO_GENERATION_FAILED"
                    }), 502

                if email:
                    try:
                        ts = datetime.datetime.utcnow().isoformat()

                        with chat_db() as cur:
                            cur.execute(
                                """
                                INSERT INTO generations (email, input, output, ts)
                                VALUES (%s, %s, %s, %s)
                                """,
                                (
                                    email,
                                    user_message,
                                    "[AUDIO GENERATED]",
                                    ts
                                )
                            )
                    except Exception as e:
                        logger.warning(f"Failed to save audio history: {e}")

                resp = jsonify({
                    "type": "audio",
                    "provider": "elevenlabs",
                    "model": used_audio_model,
                    "audio": audio_url,
                    "done": True
                })

                resp.set_cookie(
                    "device_id",
                    device_cookie,
                    max_age=60*60*24*30,
                    httponly=True,
                    samesite="Lax"
                )

                return resp

            # ── Pollinations image rate limit ─────────────────
            if image_mode:
                image_limit = RATE_LIMITS["image"]
                image_key = f"pollinations-image:{client_key}"

                if image_key not in rate_limit_store:
                    rate_limit_store[image_key] = []

                rate_limit_store[image_key] = [
                    t for t in rate_limit_store[image_key]
                    if now - t < image_limit["window"]
                ]

                if len(rate_limit_store[image_key]) >= image_limit["limit"]:
                    return jsonify({
                        "error": "Image generation limit reached. Please try again later.",
                        "code": "IMAGE_RATE_LIMIT"
                    }), 429

                rate_limit_store[image_key].append(now)

            # ── Image Mode ─────────────────────────────
            if image_mode:
                # Correct image model IDs from Pollinations dashboard
                image_model_map = {
                    "nanobanana":  "nanobanana",
                    "gptimage":    "gptimage-large",  # GPT Image 1.5
                    "novacanvas":  "nova-canvas",
                    "flux":        "flux"
                }
                mapped_img = image_model_map.get(pollinations_model, "flux")
                image_fallbacks = [
                    mapped_img,
                    "flux",
                    "nanobanana",
                    "gptimage-large",
                ]

                image_data = None
                used_image_model = None

                for fallback_model in image_fallbacks:
                    image_data = generate_image(
                        improved_prompt,
                        fallback_model
                    )

                    if image_data:
                        used_image_model = fallback_model
                        break

                if not image_data:
                    return jsonify({
                        "error": "Image generation temporarily unavailable.",
                        "code": "IMAGE_GENERATION_FAILED"
                    }), 502

                resp = jsonify({
                    "type": "image",
                    "provider": "pollinations",
                    "model": used_image_model,
                    "image": image_data,
                    "done": True
                })

                resp.set_cookie(
                    "device_id",
                    device_cookie,
                    max_age=60*60*24*30,
                    httponly=True,
                    samesite="Lax"
                )

                return resp

            # ── Text Mode ──────────────────────────────
            ai_result = call_pollinations_text(
                improved_prompt,
                pollinations_provider
            )

            if email:
                try:
                    ts = datetime.datetime.utcnow().isoformat()

                    with chat_db() as cur:
                        cur.execute(
                            """
                            INSERT INTO generations (email, input, output, ts)
                            VALUES (%s, %s, %s, %s)
                            """,
                            (
                                email,
                                user_message,
                                ai_result["text"],
                                ts
                            )
                        )
                except Exception as e:
                    logger.warning(f"Failed to save Pollinations history: {e}")

            resp = jsonify({
                "type": "text",
                "provider": ai_result["provider"],
                "model": ai_result["model"],
                "response": ai_result["text"],
                "done": True
            })

            resp.set_cookie(
                "device_id",
                device_cookie,
                max_age=60*60*24*30,
                httponly=True,
                samesite="Lax"
            )

            return resp

        # Step 3: If user selected Gemini
        if mode == "gemini":
            # ── Image generation rate limit ─────────────────
            if image_mode:
                image_limit = RATE_LIMITS["image"]
                image_key = f"image:{client_key}"

                if image_key not in rate_limit_store:
                    rate_limit_store[image_key] = []

                rate_limit_store[image_key] = [
                    t for t in rate_limit_store[image_key]
                    if now - t < image_limit["window"]
                ]

                if len(rate_limit_store[image_key]) >= image_limit["limit"]:
                    return jsonify({
                        "error": "Image generation limit reached. Please try again later.",
                        "code": "IMAGE_RATE_LIMIT"
                    }), 429

                rate_limit_store[image_key].append(now)
                image_model_map2 = {
                    "nanobanana":  "nanobanana",
                    "gptimage":    "gptimage-large",
                    "novacanvas":  "nova-canvas",
                    "flux":        "flux"
                }
                mapped_img2 = image_model_map2.get(pollinations_model, "flux")
                image_data = generate_image(
                    improved_prompt,
                    mapped_img2
                )

                if not image_data:
                    return jsonify({
                        "error": "Image generation temporarily unavailable.",
                        "code": "IMAGE_GENERATION_FAILED"
                    }), 502

                # Save image generation history
                if email:
                    try:
                        ts = datetime.datetime.utcnow().isoformat()

                        with chat_db() as cur:
                            cur.execute(
                                """
                                INSERT INTO generations (email, input, output, ts)
                                VALUES (%s, %s, %s, %s)
                                """,
                                (
                                    email,
                                    user_message,
                                    "[IMAGE GENERATED]",
                                    ts
                                )
                            )
                    except Exception as e:
                        logger.warning(f"Failed to save image history: {e}")

                resp = jsonify({
                    "type": "image",
                    "image": image_data,
                    "done": True
                })
                resp.set_cookie("device_id", device_cookie, max_age=60*60*24*30, httponly=True, samesite="Lax")
                return resp
            else:
                gemini_result = call_gemini(improved_prompt, model_choice)

                # Save Gemini response history
                if email:
                    try:
                        ts = datetime.datetime.utcnow().isoformat()

                        with chat_db() as cur:
                            cur.execute(
                                """
                                INSERT INTO generations (email, input, output, ts)
                                VALUES (%s, %s, %s, %s)
                                """,
                                (
                                    email,
                                    user_message,
                                    gemini_result["text"],
                                    ts
                                )
                            )
                    except Exception as e:
                        logger.warning(f"Failed to save Gemini history: {e}")

                resp = jsonify({
                    "type": "text",
                    "response": gemini_result["text"],
                    "usedFallback": gemini_result.get("usedFallback", False),
                    "fallbackModel": gemini_result.get("fallbackModel"),
                    "done": True
                })
                resp.set_cookie("device_id", device_cookie, max_age=60*60*24*30, httponly=True, samesite="Lax")
                return resp
    except Exception as e:
        logger.exception("Generate endpoint failure")

        return jsonify({
            "error": "AI service temporarily unavailable.",
            "code": "AI_BACKEND_ERROR"
        }), 502

    # Fallback if mode is not recognized
    return jsonify({"error": "Invalid mode or configuration."}), 400

# ── Auth: register ────────────────────────────────────────────
@app.route("/auth/register", methods=["POST"])
def auth_register():
    body = request.get_json(silent=True) or {}
    name = sanitize_text(body.get("name") or "", MAX_NAME_LENGTH)
    email = sanitize_text(body.get("email") or "", MAX_EMAIL_LENGTH).lower()
    password = (body.get("password") or "").strip()

    if not name:
        return jsonify({"error": "Name is required."}), 400
    if not email or not is_valid_email(email):
        return jsonify({"error": "A valid email address is required."}), 400

    if len(password) < PASSWORD_MIN_LENGTH:
        return jsonify({
            "error": f"Password must be at least {PASSWORD_MIN_LENGTH} characters."
        }), 400

    ts = datetime.datetime.utcnow().isoformat()
    try:
        with auth_db() as cur:
            cur.execute("SELECT email FROM users WHERE email = %s", (email,))
            existing = cur.fetchone()
            if existing:
                return jsonify({"error": "An account with this email already exists."}), 409
            cur.execute(
                """INSERT INTO users (email, name, password, avatar, provider, consent, login_count, created_at, last_seen)
                   VALUES (%s, %s, %s, %s, 'email', %s, %s, %s, %s)""",
                (email, name, generate_password_hash(password), name[0].upper(), False, 0, ts, ts)
            )
    except Exception as e:
        logger.exception("User registration failed")

        return jsonify({
            "error": "Registration temporarily unavailable.",
            "code": "REGISTER_ERROR"
        }), 500

    logger.info(f"REGISTER | {email} | {ts}")
    send_email(
        "admin@atimosai.com",
        "New User Registered",
        f"Name: {name}\nEmail: {email}\nTime: {ts}"
    )
    token = _issue_token(email)
    with auth_db() as cur:
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

    return jsonify({"status": "ok", "token": token, "user": _public_profile(user)}), 201

# ── Auth: login ───────────────────────────────────────────────
@app.route("/auth/login", methods=["POST"])
def auth_login():
    body = request.get_json(silent=True) or {}
    email = sanitize_text(body.get("email") or "", MAX_EMAIL_LENGTH).lower()
    password = (body.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    # ── Login abuse protection ─────────────────────────
    client_key = get_client_key(email)
    now = time()

    login_limit_key = f"login:{client_key}"

    if login_limit_key not in rate_limit_store:
        rate_limit_store[login_limit_key] = []

    rate_limit_store[login_limit_key] = [
        t for t in rate_limit_store[login_limit_key]
        if now - t < 60
    ]

    if len(rate_limit_store[login_limit_key]) >= 10:
        return jsonify({
            "error": "Too many login attempts. Please wait a minute.",
            "code": "LOGIN_RATE_LIMIT"
        }), 429

    rate_limit_store[login_limit_key].append(now)

    with auth_db() as cur:
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

    if not user:
        return jsonify({"error": "Invalid email or password."}), 401

    stored = user[2]
    if stored.startswith("pbkdf2:") or stored.startswith("scrypt:"):
        password_ok = check_password_hash(stored, password)
    else:
        # Legacy plain-text — accept and upgrade to hash
        password_ok = (stored == password)
        if password_ok:
            with auth_db() as cur2:
                cur2.execute("UPDATE users SET password = %s WHERE email = %s",
                             (generate_password_hash(password), email))

    if not password_ok:
        return jsonify({"error": "Invalid email or password."}), 401

    ts = datetime.datetime.utcnow().isoformat()
    with auth_db() as cur:
        cur.execute(
            "UPDATE users SET last_seen = %s, login_count = login_count + 1 WHERE email = %s",
            (ts, email)
        )
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

    token = _issue_token(email)
    logger.info(f"LOGIN | {email} | {ts}")
    # Send login notification email
    send_email(
        "admin@atimosai.com",
        "User Logged In",
        f"User: {email}\nTime: {ts}\nMethod: Email/Password"
    )

    return jsonify({"status": "ok", "token": token, "user": _public_profile(user)})

# ── Auth: logout ──────────────────────────────────────────────
@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):].strip()
        _invalidate_token(token)
    return jsonify({"status": "ok"})

# ── Auth: forgot password ─────────────────────────────────────
import random

@app.route("/auth/forgot-password", methods=["POST"])
def forgot_password():
    data = request.get_json(silent=True) or {}
    email = sanitize_text(data.get("email") or "", MAX_EMAIL_LENGTH).lower()

    if not email:
        return jsonify({"error": "Email required"}), 400

    # ── Password reset abuse protection ────────────────
    client_key = get_client_key(email)
    now = time()

    reset_limit = RATE_LIMITS["password_reset"]
    reset_key = f"reset:{client_key}"

    if reset_key not in rate_limit_store:
        rate_limit_store[reset_key] = []

    rate_limit_store[reset_key] = [
        t for t in rate_limit_store[reset_key]
        if now - t < reset_limit["window"]
    ]

    if len(rate_limit_store[reset_key]) >= reset_limit["limit"]:
        return jsonify({
            "error": "Too many reset requests. Please try again later.",
            "code": "RESET_RATE_LIMIT"
        }), 429

    rate_limit_store[reset_key].append(now)
    with auth_db() as cur:
        cur.execute("SELECT email FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
    if not user:
        return jsonify({"error": "No account found with this email"}), 404

    otp = str(random.randint(100000, 999999))
    expires = (datetime.datetime.utcnow() + datetime.timedelta(minutes=10)).isoformat()

    with auth_db() as cur:
        cur.execute("DELETE FROM password_reset WHERE email = %s", (email,))
        cur.execute(
            "INSERT INTO password_reset (email, otp, expires_at) VALUES (%s, %s, %s)",
            (email, otp, expires)
        )

    send_email(
        email,
        "Password Reset OTP",
        f"Your OTP is: {otp}\nValid for 10 minutes"
    )

    return jsonify({"status": "otp_sent"})


@app.route("/auth/send-reset-link", methods=["POST"])
def send_reset_link():
    data = request.get_json(silent=True) or {}
    email = sanitize_text(data.get("email") or "", MAX_EMAIL_LENGTH).lower()

    if not email:
        return jsonify({"error": "Email required"}), 400

    # ── Reset link abuse protection ────────────────────
    client_key = get_client_key(email)
    now = time()

    reset_limit = RATE_LIMITS["password_reset"]
    reset_key = f"reset-link:{client_key}"

    if reset_key not in rate_limit_store:
        rate_limit_store[reset_key] = []

    rate_limit_store[reset_key] = [
        t for t in rate_limit_store[reset_key]
        if now - t < reset_limit["window"]
    ]

    if len(rate_limit_store[reset_key]) >= reset_limit["limit"]:
        return jsonify({
            "error": "Too many reset requests. Please try again later.",
            "code": "RESET_RATE_LIMIT"
        }), 429

    rate_limit_store[reset_key].append(now)

    with auth_db() as cur:
        cur.execute("SELECT email FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

    if not user:
        return jsonify({"error": "No account found with this email"}), 404

    import random
    otp = str(random.randint(100000, 999999))
    expires = (datetime.datetime.utcnow() + datetime.timedelta(minutes=10)).isoformat()

    with auth_db() as cur:
        cur.execute("DELETE FROM password_reset WHERE email = %s", (email,))
        cur.execute(
            "INSERT INTO password_reset (email, otp, expires_at) VALUES (%s, %s, %s)",
            (email, otp, expires)
        )

    send_email(
        email,
        "Prometix Password Reset",
        f"Your OTP is: {otp}\nValid for 10 minutes"
    )

    return jsonify({"message": "Reset link (OTP) sent to your email"})

# ── Auth: reset password ─────────────────────────────────────
@app.route("/auth/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json(silent=True) or {}
    email = sanitize_text(data.get("email") or "", MAX_EMAIL_LENGTH).lower()
    otp = sanitize_text(data.get("otp") or "", 12)
    new_password = (data.get("new_password") or "").strip()

    if not email or not otp or not new_password:
        return jsonify({"error": "Missing fields"}), 400
    if len(new_password) < PASSWORD_MIN_LENGTH:
        return jsonify({
            "error": f"Password must be at least {PASSWORD_MIN_LENGTH} characters"
        }), 400

    with auth_db() as cur:
        cur.execute(
            "SELECT * FROM password_reset WHERE email = %s AND otp = %s",
            (email, otp)
        )
        record = cur.fetchone()

    if not record:
        return jsonify({"error": "Invalid OTP"}), 400

    if datetime.datetime.utcnow().isoformat() > record[2]:
        return jsonify({"error": "OTP expired"}), 400

    with auth_db() as cur:
        cur.execute(
            "UPDATE users SET password = %s WHERE email = %s",
            (generate_password_hash(new_password), email)
        )
        cur.execute("DELETE FROM password_reset WHERE email = %s", (email,))

    return jsonify({"status": "password_updated"})

# ── User: consent ─────────────────────────────────────────────
@app.route("/user/consent", methods=["POST"])
def user_consent():
    email, err = _validate_token()
    if err:
        return err
    body    = request.get_json(silent=True) or {}
    consent = 1 if body.get("consent") else 0
    with auth_db() as cur:
        cur.execute("UPDATE users SET consent = %s WHERE email = %s", (consent, email))
    logger.info(f"CONSENT | {email} | {'YES' if consent else 'NO'}")
    return jsonify({"status": "ok"})

# ── User: search log ─────────────────────────────────────────
@app.route("/user/search", methods=["POST"])
def user_search():
    email, err = _validate_token()
    if err:
        return err
    body  = request.get_json(silent=True) or {}
    query = sanitize_text(body.get("query") or "", 1000)
    if not query:
        return jsonify({"status": "ignored"})

    if len(query) > 1000:
        return jsonify({
            "error": "Search query too long.",
            "code": "SEARCH_TOO_LONG"
        }), 413

    ts = datetime.datetime.utcnow().isoformat()
    with chat_db() as cur:
        cur.execute("INSERT INTO searches (email, query, ts) VALUES (%s, %s, %s)", (email, query, ts))
    return jsonify({"status": "ok"})

# ── User: feedback ───────────────────────────────────────────

@app.route("/user/feedback", methods=["POST"])
def user_feedback():
    email, err = _validate_token()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    message = sanitize_text(data.get("message") or "", 2000)
    rating = int(data.get("rating") or 0)

    # Clamp invalid ratings
    if rating < 0:
        rating = 0
    if rating > 5:
        rating = 5

    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    if len(message) > 2000:
        return jsonify({
            "error": "Feedback message too long.",
            "code": "FEEDBACK_TOO_LONG"
        }), 413

    ts = datetime.datetime.utcnow().isoformat()

    with chat_db() as cur:
        cur.execute(
            "INSERT INTO feedback (email, message, rating, ts) VALUES (%s, %s, %s, %s)",
            (email, message, rating, ts)
        )

    send_email(
        "admin@atimosai.com",
        "New Feedback Received",
        f"User: {email}\nRating: {rating}\nMessage: {message}\nTime: {ts}"
    )

    return jsonify({"status": "ok"})

# ── User: history ────────────────────────────────────────────
@app.route("/user/history", methods=["GET"])
def user_history():
    email, err = _validate_token()
    if err:
        return err

    with chat_db() as cur:
        cur.execute(
            "SELECT input, output, ts FROM generations WHERE email = %s ORDER BY id DESC LIMIT 10",
            (email,)
        )
        rows = cur.fetchall()

    return jsonify([{"input": r[0], "output": r[1], "ts": r[2]} for r in rows])

# ── Training: pair ────────────────────────────────────────────
@app.route("/training/pair", methods=["POST"])
def training_pair():
    email, err = _validate_token()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    inp = sanitize_text(body.get("input") or "", 4000)
    out = sanitize_text(body.get("output") or "", 12000)
    if not inp or not out:
        return jsonify({"status": "ignored"})

    if len(inp) > 4000 or len(out) > 12000:
        return jsonify({
            "error": "Training data too large.",
            "code": "TRAINING_TOO_LARGE"
        }), 413

    with auth_db() as cur:
        cur.execute("SELECT consent FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
    if not user or not user[0]:
        return jsonify({"status": "no_consent"})
    ts = datetime.datetime.utcnow().isoformat()
    with chat_db() as cur:
        cur.execute(
            "INSERT INTO training_pairs (email, input, output, ts) VALUES (%s, %s, %s, %s)",
            (email, inp, out, ts)
        )
    return jsonify({"status": "stored"})

# ── Google OAuth ─────────────────────────────────────────────
@app.route("/auth/google")
def google_login():
    redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI")

    if not redirect_uri:
        logger.error("GOOGLE_REDIRECT_URI is missing")
        return jsonify({
            "error": "OAuth configuration unavailable.",
            "code": "OAUTH_CONFIG_ERROR"
        }), 500
    return google.authorize_redirect(redirect_uri)

@app.route("/google-callback")
def google_callback():
    logger.info("Google OAuth callback triggered")
    try:
        token = google.authorize_access_token()
        user_info = google.get("userinfo", token=token).json()

        email = user_info.get("email")
        name = user_info.get("name")

        if not email or not name:
            return jsonify({
                "error": "Google authentication failed.",
                "code": "GOOGLE_USERDATA_MISSING"
            }), 400

        ts = datetime.datetime.utcnow().isoformat()

        try:
            with auth_db() as cur:
                cur.execute("SELECT * FROM users WHERE email = %s", (email,))
                user = cur.fetchone()

                if not user:
                    cur.execute(
                        """INSERT INTO users (email, name, password, avatar, provider, consent, login_count, created_at, last_seen)
                           VALUES (%s, %s, '', %s, 'google', %s, %s, %s, %s)""",
                        (email, name, name[0].upper(), False, 1, ts, ts)
                    )
                else:
                    cur.execute(
                        "UPDATE users SET last_seen = %s, login_count = login_count + 1 WHERE email = %s",
                        (ts, email)
                    )
        except Exception as e:
            logger.exception("Database error during Google login")

            return jsonify({
                "error": "Account login failed.",
                "code": "GOOGLE_DB_ERROR"
            }), 500

        session_token = _issue_token(email)
        # Send login notification email for Google login
        try:
            send_email(
                "admin@atimosai.com",
                "User Logged In (Google)",
                f"User: {email}\nName: {name}\nTime: {ts}\nMethod: Google OAuth"
            )
        except Exception as e:
            logger.warning(f"Google login email error: {e}")

        frontend_url = os.environ.get("FRONTEND_URL", "https://atimosai.com")
        return redirect(f"{frontend_url}/login.html?token={session_token}&name={name}&email={email}")

    except Exception as e:
        logger.exception("Google OAuth callback failure")

        return jsonify({
            "error": "Google authentication failed.",
            "code": "GOOGLE_AUTH_ERROR"
        }), 500


# ── Security Headers ─────────────────────────────────────────
@app.after_request
def secure_headers(response):
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(self), geolocation=(), payment=()"
    )
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' https: data: blob:; "
        "img-src 'self' https: data: blob:; "
        "script-src 'self' 'unsafe-inline' https:; "
        "style-src 'self' 'unsafe-inline' https:; "
        "font-src 'self' https: data:; "
        "connect-src 'self' https:; "
        "frame-ancestors 'none';"
    )
    # Prevent caching of sensitive endpoints
    if request.path.startswith("/auth") or request.path.startswith("/generate"):
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, private"
        )
        response.headers["Pragma"] = "no-cache"
    return response

# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Prometix backend starting")
    logger.info(f"AI backend initialized with {len(_GROQ_MODELS)} Groq models")
    logger.info("Database: Supabase Auth + Neon AI Storage")
    logger.info("Security systems initialized")
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Running on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
