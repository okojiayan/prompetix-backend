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
import datetime
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, redirect
from time import time
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth
import requests
import json
import psycopg2
import google.generativeai as genai

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
                print("Watermark error:", e)

            # Convert back to bytes
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print("Image fetch error:", e)

    return None

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
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
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
        except Exception as e:
            print("Email error:", e)

    import threading
    threading.Thread(target=_send).start()


# ── PostgreSQL setup ──────────────────────────────────────────────
def _get_conn():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise Exception("DATABASE_URL environment variable not set")
    return psycopg2.connect(
        db_url,
        sslmode="require"
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
    token = str(uuid.uuid4())
    ts    = datetime.datetime.utcnow().isoformat()
    with db() as cur:
        cur.execute(
            "INSERT INTO sessions (token, email, created_at) VALUES (%s, %s, %s)",
            (token, email, ts)
        )
    return token

def _validate_token():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None, (jsonify({"error": "Session expired. Please sign in again.", "code": "TOKEN_MISSING"}), 401)
    token = auth_header[len("Bearer "):].strip()
    with db() as cur:
        cur.execute("SELECT email FROM sessions WHERE token = %s", (token,))
        row = cur.fetchone()
    if not row:
        return None, (jsonify({"error": "Session expired. Please sign in again.", "code": "TOKEN_INVALID"}), 401)
    return row[0], None

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
        raise Exception("GROQ_API_KEY environment variable is not set.")

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
def call_gemini(prompt: str, model_choice: str = "auto") -> str:
    try:
        if model_choice == "3.1-pro":
            model_name = "gemini-3.1-pro"
        elif model_choice == "3.1-flash":
            model_name = "gemini-3.1-flash"
        elif model_choice == "1.5-pro":
            model_name = "gemini-1.5-pro"
        else:
            model_name = "gemini-1.5-flash"

        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)

        if response:
            # Handle text response
            if hasattr(response, "text") and response.text:
                return response.text.strip()

            # Handle possible image output (future-safe)
            if hasattr(response, "candidates"):
                return "Image generated (display in frontend)"

    except Exception as e:
        print("Model failed, fallback:", e)

    # fallback
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    if response and hasattr(response, "text"):
        return response.text.strip()

    return "No response generated"

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
    image_mode = is_image_request(user_message)
    model_choice = (data.get("model") or "auto").lower()
    mode = (data.get("mode") or "prompt").lower()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    # Gemini Pro limit: 5 requests per 5 hours per user
    if model_choice == "3.1-pro":
        window_seconds = 5 * 60 * 60
        now_ts = int(time())
        
        with db() as cur:
            cur.execute(
                "SELECT ts FROM generations WHERE email = %s ORDER BY ts DESC LIMIT 20",
                (email,)
            )
            rows = cur.fetchall()

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
            return jsonify({
                "type": "prompt",
                "response": improved_prompt,
                "done": True
            })

        # Step 3: If user selected Gemini
        if mode == "gemini":
            if image_mode:
                image_data = generate_image(improved_prompt)
                return jsonify({
                    "type": "image",
                    "image": image_data,
                    "done": True
                })
            else:
                final_output = call_gemini(improved_prompt, model_choice)
                return jsonify({
                    "type": "text",
                    "response": final_output,
                    "done": True
                })
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    # Fallback if mode is not recognized
    return jsonify({"error": "Invalid mode or configuration."}), 400

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
        with db() as cur:
            cur.execute("SELECT email FROM users WHERE email = %s", (email,))
            existing = cur.fetchone()
            if existing:
                return jsonify({"error": "An account with this email already exists."}), 409
            cur.execute(
                """INSERT INTO users (email, name, password, avatar, provider, consent, login_count, created_at, last_seen)
                   VALUES (%s, %s, %s, %s, 'email', 0, 0, %s, %s)""",
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
    with db() as cur:
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

    return jsonify({"status": "ok", "token": token, "user": _public_profile(user)}), 201

# ── Auth: login ───────────────────────────────────────────────
@app.route("/auth/login", methods=["POST"])
def auth_login():
    body     = request.get_json(silent=True) or {}
    email    = (body.get("email")    or "").strip().lower()
    password = (body.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

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
    print(f"  [LOGIN]  {email}  —  {ts}")
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
    email = (data.get("email") or "").strip().lower()

    if not email:
        return jsonify({"error": "Email required"}), 400
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


# ── Auth: reset password ─────────────────────────────────────
@app.route("/auth/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    otp = (data.get("otp") or "").strip()
    new_password = (data.get("new_password") or "").strip()

    if not email or not otp or not new_password:
        return jsonify({"error": "Missing fields"}), 400
    if len(new_password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

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
    message = (data.get("message") or "").strip()
    rating = int(data.get("rating") or 0)

    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

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
    inp  = (body.get("input")  or "").strip()
    out  = (body.get("output") or "").strip()
    if not inp or not out:
        return jsonify({"status": "ignored"})
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
    redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "https://prometix-backend.onrender.com/google-callback")
    return google.authorize_redirect(redirect_uri)

@app.route("/google-callback")
def google_callback():
    print("Google callback triggered")
    try:
        token = google.authorize_access_token()
        user_info = google.get("userinfo", token=token).json()

        email = user_info.get("email")
        name = user_info.get("name")

        if not email or not name:
            return "Google authentication failed: missing user data", 400

        ts = datetime.datetime.utcnow().isoformat()

        try:
            with db() as cur:
                cur.execute("SELECT * FROM users WHERE email = %s", (email,))
                user = cur.fetchone()

                if not user:
                    cur.execute(
                        """INSERT INTO users (email, name, password, avatar, provider, consent, login_count, created_at, last_seen)
                           VALUES (%s, %s, '', %s, 'google', 0, 1, %s, %s)""",
                        (email, name, name[0].upper(), ts, ts)
                    )
                else:
                    cur.execute(
                        "UPDATE users SET last_seen = %s, login_count = login_count + 1 WHERE email = %s",
                        (ts, email)
                    )
        except Exception as e:
            return f"Database error during Google login: {str(e)}", 500

        session_token = _issue_token(email)
        # Send login notification email for Google login
        try:
            send_email(
                "admin@atimosai.com",
                "User Logged In (Google)",
                f"User: {email}\nName: {name}\nTime: {ts}\nMethod: Google OAuth"
            )
        except Exception as e:
            print("Google login email error:", e)

        frontend_url = os.environ.get("FRONTEND_URL", "https://atimosai.com")
        return redirect(f"{frontend_url}/login.html?token={session_token}&name={name}&email={email}")

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
    print("  DB          : Supabase PostgreSQL")
    print(f"  Endpoints   : /auth/register  /auth/login  /auth/logout  /generate")
    print(f"  Security    : bcrypt hashing  UUID tokens  PostgreSQL\n")
    port = int(os.environ.get("PORT", 10000))
    print(f"Running on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
