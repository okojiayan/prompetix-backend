# ============================================================
#  Prometix by Atimos AI — Flask Backend  v3.0
#
#  Run:  python app.py
#  Requires: pip install flask flask-cors requests werkzeug
#
#  AI backend : Groq API (mixtral-8x7b-32768)
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
import uuid
import sqlite3
import datetime
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, redirect
from time import time
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth
import requests
import smtplib
from email.mime.text import MIMEText

rate_limit_store = {}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret")
CORS(app, resources={r"/*": {"origins": "*"}})
rate_limit_store = {}

# ── Google OAuth setup ───────────────────────────────────────
oauth = OAuth(app)

google = oauth.register(
    name='google',
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
    api_base_url='https://openidconnect.googleapis.com/v1/',
    client_kwargs={'scope': 'openid email profile'},
)

# ── Config ────────────────────────────────────────────────────
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
# Two competing models — fastest valid response wins; other is fallback
_GROQ_MODELS    = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
]
REQUEST_TIMEOUT = int(os.environ.get("PROMETIX_TIMEOUT", "30"))

SMTP_SERVER = "smtp.zoho.in"
SMTP_PORT = 587
EMAIL_ADDRESS = "admin@atimosai.com"
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

def send_email(to_email, subject, message):
    try:
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to_email

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print("Email error:", e)

# ── SQLite setup ──────────────────────────────────────────────
# On Render: set PROMETIX_DB to a path on a mounted persistent disk,
# e.g. /var/data/prometix.db — otherwise it lives beside app.py.
DB_PATH = os.environ.get("PROMETIX_DB", os.path.join(os.path.dirname(__file__), "prometix.db"))

def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

@contextmanager
def db():
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                email       TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                password    TEXT NOT NULL,
                avatar      TEXT NOT NULL DEFAULT '',
                provider    TEXT NOT NULL DEFAULT 'email',
                consent     INTEGER NOT NULL DEFAULT 0,
                login_count INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL,
                last_seen   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token       TEXT PRIMARY KEY,
                email       TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (email) REFERENCES users(email) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS searches (
                id    INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                query TEXT NOT NULL,
                ts    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS training_pairs (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                email  TEXT NOT NULL,
                input  TEXT NOT NULL,
                output TEXT NOT NULL,
                ts     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                message TEXT,
                rating INTEGER,
                ts TEXT
            );

            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                input TEXT,
                output TEXT,
                ts TEXT
            );

            CREATE TABLE IF NOT EXISTS password_reset (
                email TEXT,
                otp TEXT,
                expires_at TEXT
            );
        """)

init_db()

# ── Token helpers ─────────────────────────────────────────────
def _issue_token(email: str) -> str:
    token = str(uuid.uuid4())
    ts    = datetime.datetime.utcnow().isoformat()
    with db() as conn:
        conn.execute(
            "INSERT INTO sessions (token, email, created_at) VALUES (?, ?, ?)",
            (token, email, ts)
        )
    return token

def _validate_token():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None, (jsonify({"error": "Session expired. Please sign in again.", "code": "TOKEN_MISSING"}), 401)
    token = auth_header[len("Bearer "):].strip()
    with db() as conn:
        row = conn.execute("SELECT email FROM sessions WHERE token = ?", (token,)).fetchone()
    if not row:
        return None, (jsonify({"error": "Session expired. Please sign in again.", "code": "TOKEN_INVALID"}), 401)
    return row["email"], None

def _invalidate_token(token: str):
    with db() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))

def _public_profile(row) -> dict:
    return {
        "email":      row["email"],
        "name":       row["name"],
        "avatar":     row["avatar"] or row["name"][0].upper(),
        "provider":   row["provider"],
        "consent":    bool(row["consent"]),
        "created_at": row["created_at"],
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
        raise Exception("GROQ_API_KEY environment variable is not set.")

    intent       = detect_intent(user_message)
    style        = get_prompt_style(intent, user_message)
    length       = len(user_message.split())
    user_content = (
        f"STYLE: {style}\n"
        f"COMPLEXITY: {'simple' if length < 6 else 'advanced'}\n\n"
        f"Raw idea:\n{user_message}"
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

# ── Health ────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "product": "Prometix by Atimos AI"})

# ── Generate ─────────────────────────────────────────────────
@app.route("/generate", methods=["POST"])
def generate():
    email, err = _validate_token()
    if err:
        return err

    user_ip = request.remote_addr
    now = time()

    if user_ip in rate_limit_store:
        if now - rate_limit_store[user_ip] < 2:
            return jsonify({"error": "Too many requests"}), 429

    rate_limit_store[user_ip] = now

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON in request body."}), 400

    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    try:
        ai_text = call_groq(user_message)
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    if not ai_text:
        return jsonify({"error": "Backend returned an empty response. Try rephrasing."}), 502

    with db() as conn:
        conn.execute(
            "INSERT INTO generations (email, input, output, ts) VALUES (?, ?, ?, ?)",
            (email, user_message, ai_text, datetime.datetime.utcnow().isoformat())
        )
    return jsonify({"response": ai_text, "done": True})

# ── Auth: register ────────────────────────────────────────────
@app.route("/auth/register", methods=["POST"])
def auth_register():
    body     = request.get_json(silent=True) or {}
    name     = (body.get("name")     or "").strip()
    email    = (body.get("email")    or "").strip().lower()
    password = (body.get("password") or "").strip()

    if not name:
        return jsonify({"error": "Name is required."}), 400
    if not email or "@" not in email:
        return jsonify({"error": "A valid email address is required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    ts = datetime.datetime.utcnow().isoformat()
    try:
        with db() as conn:
            existing = conn.execute("SELECT email FROM users WHERE email = ?", (email,)).fetchone()
            if existing:
                return jsonify({"error": "An account with this email already exists."}), 409
            conn.execute(
                """INSERT INTO users (email, name, password, avatar, provider, consent, login_count, created_at, last_seen)
                   VALUES (?, ?, ?, ?, 'email', 0, 0, ?, ?)""",
                (email, name, generate_password_hash(password), name[0].upper(), ts, ts)
            )
    except Exception as e:
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500

    print(f"  [REGISTER] {email}  —  {ts}")
    send_email(
        "admin@atimosai.com",
        "New User Registered",
        f"Name: {name}\nEmail: {email}\nTime: {ts}"
    )
    token = _issue_token(email)
    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    return jsonify({"status": "ok", "token": token, "user": _public_profile(user)}), 201

# ── Auth: login ───────────────────────────────────────────────
@app.route("/auth/login", methods=["POST"])
def auth_login():
    body     = request.get_json(silent=True) or {}
    email    = (body.get("email")    or "").strip().lower()
    password = (body.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if not user:
        return jsonify({"error": "Invalid email or password."}), 401

    stored = user["password"]
    if stored.startswith("pbkdf2:") or stored.startswith("scrypt:"):
        password_ok = check_password_hash(stored, password)
    else:
        # Legacy plain-text — accept and upgrade to hash
        password_ok = (stored == password)
        if password_ok:
            with db() as conn:
                conn.execute("UPDATE users SET password = ? WHERE email = ?",
                             (generate_password_hash(password), email))

    if not password_ok:
        return jsonify({"error": "Invalid email or password."}), 401

    ts = datetime.datetime.utcnow().isoformat()
    with db() as conn:
        conn.execute(
            "UPDATE users SET last_seen = ?, login_count = login_count + 1 WHERE email = ?",
            (ts, email)
        )
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    token = _issue_token(email)
    print(f"  [LOGIN]  {email}  —  {ts}")

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
    email = (data.get("email") or "").strip().lower()

    if not email:
        return jsonify({"error": "Email required"}), 400

    otp = str(random.randint(100000, 999999))
    expires = (datetime.datetime.utcnow() + datetime.timedelta(minutes=10)).isoformat()

    with db() as conn:
        conn.execute("DELETE FROM password_reset WHERE email = ?", (email,))
        conn.execute(
            "INSERT INTO password_reset (email, otp, expires_at) VALUES (?, ?, ?)",
            (email, otp, expires)
        )

    send_email(
        email,
        "Password Reset OTP",
        f"Your OTP is: {otp}\nValid for 10 minutes"
    )

    return jsonify({"status": "otp_sent"})


# ── Auth: reset password ─────────────────────────────────────
@app.route("/auth/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    otp = (data.get("otp") or "").strip()
    new_password = (data.get("new_password") or "").strip()

    if not email or not otp or not new_password:
        return jsonify({"error": "Missing fields"}), 400

    with db() as conn:
        record = conn.execute(
            "SELECT * FROM password_reset WHERE email = ? AND otp = ?",
            (email, otp)
        ).fetchone()

    if not record:
        return jsonify({"error": "Invalid OTP"}), 400

    if datetime.datetime.utcnow().isoformat() > record["expires_at"]:
        return jsonify({"error": "OTP expired"}), 400

    with db() as conn:
        conn.execute(
            "UPDATE users SET password = ? WHERE email = ?",
            (generate_password_hash(new_password), email)
        )
        conn.execute("DELETE FROM password_reset WHERE email = ?", (email,))

    return jsonify({"status": "password_updated"})

# ── User: consent ─────────────────────────────────────────────
@app.route("/user/consent", methods=["POST"])
def user_consent():
    email, err = _validate_token()
    if err:
        return err
    body    = request.get_json(silent=True) or {}
    consent = 1 if body.get("consent") else 0
    with db() as conn:
        conn.execute("UPDATE users SET consent = ? WHERE email = ?", (consent, email))
    print(f"  [CONSENT] {email}  ->  {'YES' if consent else 'NO'}")
    return jsonify({"status": "ok"})

# ── User: search log ─────────────────────────────────────────
@app.route("/user/search", methods=["POST"])
def user_search():
    email, err = _validate_token()
    if err:
        return err
    body  = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"status": "ignored"})
    ts = datetime.datetime.utcnow().isoformat()
    with db() as conn:
        conn.execute("INSERT INTO searches (email, query, ts) VALUES (?, ?, ?)", (email, query, ts))
    return jsonify({"status": "ok"})

# ── User: feedback ───────────────────────────────────────────

@app.route("/user/feedback", methods=["POST"])
def user_feedback():
    email, err = _validate_token()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    rating = int(data.get("rating") or 0)

    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    ts = datetime.datetime.utcnow().isoformat()

    with db() as conn:
        conn.execute(
            "INSERT INTO feedback (email, message, rating, ts) VALUES (?, ?, ?, ?)",
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

    with db() as conn:
        rows = conn.execute(
            "SELECT input, output, ts FROM generations WHERE email = ? ORDER BY id DESC LIMIT 10",
            (email,)
        ).fetchall()

    return jsonify([dict(r) for r in rows])

# ── Training: pair ────────────────────────────────────────────
@app.route("/training/pair", methods=["POST"])
def training_pair():
    email, err = _validate_token()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    inp  = (body.get("input")  or "").strip()
    out  = (body.get("output") or "").strip()
    if not inp or not out:
        return jsonify({"status": "ignored"})
    with db() as conn:
        user = conn.execute("SELECT consent FROM users WHERE email = ?", (email,)).fetchone()
    if not user or not user["consent"]:
        return jsonify({"status": "no_consent"})
    ts = datetime.datetime.utcnow().isoformat()
    with db() as conn:
        conn.execute(
            "INSERT INTO training_pairs (email, input, output, ts) VALUES (?, ?, ?, ?)",
            (email, inp, out, ts)
        )
    return jsonify({"status": "stored"})

# ── Google OAuth ─────────────────────────────────────────────
@app.route("/auth/google")
def google_login():
    redirect_uri = "https://prometix-backend.onrender.com/google-callback"
    return google.authorize_redirect(redirect_uri)

@app.route("/google-callback")
def google_callback():
    print("Google callback triggered")
    try:
        token = google.authorize_access_token()
        user_info = google.get("userinfo", token=token).json()

        email = user_info.get("email")
        name = user_info.get("name")

        if not email:
            return "Error: No email received from Google", 400

        ts = datetime.datetime.utcnow().isoformat()

        with db() as conn:
            user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

            if not user:
                conn.execute(
                    """INSERT INTO users (email, name, password, avatar, provider, consent, login_count, created_at, last_seen)
                       VALUES (?, ?, '', ?, 'google', 0, 1, ?, ?)""",
                    (email, name, name[0].upper(), ts, ts)
                )
            else:
                conn.execute(
                    "UPDATE users SET last_seen = ?, login_count = login_count + 1 WHERE email = ?",
                    (ts, email)
                )

        session_token = _issue_token(email)

        return redirect(f"https://atimosai.com/login.html?token={session_token}&name={name}&email={email}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Google Auth Error: {str(e)}", 500

# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║   Prometix by Atimos AI  —  v3.0         ║")
    print("  ║   Mode : Prompt Engineering Tool         ║")
    print("  ╚══════════════════════════════════════════╝")
    print(f"\n  AI backend  : Groq API (dual-model, race mode)")
    print(f"  DB path     : {DB_PATH}")
    print(f"  Endpoints   : /auth/register  /auth/login  /auth/logout  /generate")
    print(f"  Security    : bcrypt hashing  UUID tokens  SQLite\n")
    port = int(os.environ.get("PORT", 10000))
    print(f"Running on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
