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
import google.generativeai as genai
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

# ── Pollinations image generator ─────────────────────────────
def generate_image(prompt: str) -> str:
    """
    Generate image and return base64 so frontend can render directly
    """
    import urllib.parse
    import base64

    encoded = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}"

    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            from PIL import Image, ImageDraw, ImageFont
            from io import BytesIO

            # Load image
            img = Image.open(BytesIO(response.content)).convert("RGBA")

            # ── Logo Watermark (Prometix) ─────────────────────
            from PIL import ImageEnhance

            try:
                logo_path = os.path.join(os.path.dirname(__file__), "watermark.png")
                logo = Image.open(logo_path).convert("RGBA")

                # Resize logo based on image size (responsive)
                base_width = int(img.width * 0.15)
                ratio = base_width / logo.width
                new_size = (base_width, int(logo.height * ratio))
                logo = logo.resize(new_size, Image.LANCZOS)

                # Reduce opacity
                alpha = logo.split()[3]
                alpha = ImageEnhance.Brightness(alpha).enhance(0.5)
                logo.putalpha(alpha)

                # Position (bottom-left, safe from Gemini watermark)
                margin = 20
                position = (margin, img.height - logo.height - margin)

                # Paste logo
                img.paste(logo, position, logo)

            except Exception as e:
                logger.warning(f"Watermark error: {e}")

            # Convert back to bytes
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.exception("Image generation failed")

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

# ── Gemini Model Health Tracking ─────────────────────────────
model_health = {}
MODEL_FAILURE_COOLDOWN = 60 * 5  # 5 minutes

RATE_LIMITS = {
    "guest": {
        "window": 60 * 60,
        "limit": 12
    },
    "auth": {
        "window": 60,
        "limit": 40
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

CORS(
    app,
    resources={
        r"/*": {
            "origins": [frontend_origin]
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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
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
REQUIRED_ENV_VARS = {
    "DATABASE_URL": os.environ.get("DATABASE_URL"),
    "SECRET_KEY": os.environ.get("SECRET_KEY"),
    "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY"),
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
# ── Gemini Setup ─────────────────────────────────────────────
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

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
def _get_conn():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise Exception("DATABASE_URL environment variable not set")
    return psycopg2.connect(
        db_url,
        sslmode="require",
        connect_timeout=10
    )

@contextmanager
def db():
    conn = _get_conn()
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
    with db() as cur:
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
    with db() as cur:
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
                with db() as cur:
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
                with db() as cur:
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

    return row[0], None

# ── Admin required decorator ─────────────────────────────
def admin_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        email, err = _validate_token()
        if err:
            return err
        with db() as cur:
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
    with db() as cur:
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


# ── Gemini Function ──────────────────────────────────────────
# ── Gemini Function ──────────────────────────────────────────
def call_gemini(prompt: str, model_choice: str = "2.5-flash"):
    # ── Dynamic Fallback Chains ─────────────────────────
    fallback_chains = {
        "2.5-pro": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ],

        "2.5-flash": [
            "gemini-2.5-flash",
            "gemini-1.5-flash"
        ],

        "1.5-pro": [
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ],

        "1.5-flash": [
            "gemini-1.5-flash"
        ]
    }

    model_chain = fallback_chains.get(
        model_choice,
        ["gemini-1.5-flash"]
    )

    # ── Smart health-based routing ─────────────────────
    now_ts = time()

    filtered_chain = []

    for model in model_chain:
        failed_until = model_health.get(model, 0)

        if now_ts >= failed_until:
            filtered_chain.append(model)

    # fallback if all models temporarily unhealthy
    if not filtered_chain:
        filtered_chain = model_chain

    model_chain = filtered_chain

    used_fallback = False
    attempted_models = []

    for idx, model_name in enumerate(model_chain):
        try:
            attempted_models.append(model_name)

            # Retry each Gemini model once before fallback
            for attempt in range(2):
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    break
                except Exception as retry_error:
                    if attempt == 1:
                        raise retry_error

                    logger.warning(
                        f"Retrying Gemini model ({model_name}) after temporary failure"
                    )

            if not response:
                raise Exception("Empty Gemini response")
            if response and hasattr(response, "text") and response.text:
                # Mark healthy again after successful response
                model_health[model_name] = 0
                return {
                    "text": response.text.strip(),
                    "usedFallback": idx > 0,
                    "fallbackModel": model_name if idx > 0 else None,
                    "attemptedModels": attempted_models
                }

        except Exception as e:
            logger.warning(f"Gemini model failed ({model_name}): {e}")

            # Mark temporarily unhealthy
            model_health[model_name] = time() + MODEL_FAILURE_COOLDOWN

            used_fallback = True
            continue

    return {
        "text": "All Gemini models failed.",
        "usedFallback": used_fallback,
        "fallbackModel": None,
        "attemptedModels": attempted_models
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
                "error": "Guest limit reached (12 prompts/hour). Please login to continue.",
                "code": "GUEST_LIMIT"
            })
            resp.set_cookie("device_id", device_cookie, max_age=60*60*24*30, httponly=True, samesite="Lax")
            return resp, 429

        rate_limit_store[client_key].append(now)

    # ── General request rate limiting ─────────────────────
    auth_limit = RATE_LIMITS["auth"]

    if client_key not in rate_limit_store:
        rate_limit_store[client_key] = []

    rate_limit_store[client_key] = [
        t for t in rate_limit_store[client_key]
        if now - t < auth_limit["window"]
    ]

    if len(rate_limit_store[client_key]) >= auth_limit["limit"]:
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

    rate_limit_store[client_key].append(now)

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
    model_choice = (data.get("model") or "2.5-flash").lower()
    mode = (data.get("mode") or "prompt").lower()
    # ── Restrict AI features for guests ─────────────────────────
    if not email and mode == "gemini":
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
            with db() as cur:
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

                    with db() as cur:
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
                image_data = generate_image(improved_prompt)

                if not image_data:
                    return jsonify({
                        "error": "Image generation temporarily unavailable.",
                        "code": "IMAGE_GENERATION_FAILED"
                    }), 502

                # Save image generation history
                if email:
                    try:
                        ts = datetime.datetime.utcnow().isoformat()

                        with db() as cur:
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

                        with db() as cur:
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
        with db() as cur:
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
    with db() as cur:
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

    with db() as cur:
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
            with db() as cur2:
                cur2.execute("UPDATE users SET password = %s WHERE email = %s",
                             (generate_password_hash(password), email))

    if not password_ok:
        return jsonify({"error": "Invalid email or password."}), 401

    ts = datetime.datetime.utcnow().isoformat()
    with db() as cur:
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
    with db() as cur:
        cur.execute("SELECT email FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
    if not user:
        return jsonify({"error": "No account found with this email"}), 404

    otp = str(random.randint(100000, 999999))
    expires = (datetime.datetime.utcnow() + datetime.timedelta(minutes=10)).isoformat()

    with db() as cur:
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

    with db() as cur:
        cur.execute("SELECT email FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

    if not user:
        return jsonify({"error": "No account found with this email"}), 404

    import random
    otp = str(random.randint(100000, 999999))
    expires = (datetime.datetime.utcnow() + datetime.timedelta(minutes=10)).isoformat()

    with db() as cur:
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

    with db() as cur:
        cur.execute(
            "SELECT * FROM password_reset WHERE email = %s AND otp = %s",
            (email, otp)
        )
        record = cur.fetchone()

    if not record:
        return jsonify({"error": "Invalid OTP"}), 400

    if datetime.datetime.utcnow().isoformat() > record[2]:
        return jsonify({"error": "OTP expired"}), 400

    with db() as cur:
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
    with db() as cur:
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
    with db() as cur:
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

    with db() as cur:
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

    with db() as cur:
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

    with db() as cur:
        cur.execute("SELECT consent FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
    if not user or not user[0]:
        return jsonify({"status": "no_consent"})
    ts = datetime.datetime.utcnow().isoformat()
    with db() as cur:
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
            with db() as cur:
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
    logger.info("Database: Supabase PostgreSQL")
    logger.info("Security systems initialized")
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Running on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
