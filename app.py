# ============================================================
#  Prometix by Atimos AI — Flask Backend  v2.2
#
#  Run:  python app.py
#  Requires: pip install flask flask-cors requests werkzeug
#
#  Endpoints:
#    GET  /health              — backend ping
#    GET  /models              — list Ollama models
#    POST /auth/register       — create account (hashed password)
#    POST /auth/login          — authenticate + issue session token
#    POST /auth/logout         — invalidate session token
#    POST /generate            — rewrite raw idea → prompt (protected)
#    POST /user/consent        — save training consent flag
#    POST /user/search         — log search query (protected)
#    POST /training/pair       — store training pair (protected)
# ============================================================

import json
import os
import uuid
import datetime
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Config ────────────────────────────────────────────────────
# Override PROMETIX_BACKEND_URL via environment variable in production.
BACKEND_BASE_URL = os.environ.get("PROMETIX_BACKEND_URL", "http://127.0.0.1:5000")
OLLAMA_URL       = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL    = os.environ.get("PROMETIX_MODEL", "mistral")
REQUEST_TIMEOUT  = int(os.environ.get("PROMETIX_TIMEOUT", "120"))

# ── In-memory token store ─────────────────────────────────────
# Maps token (str) → email (str).
# Tokens are invalidated on logout or server restart.
# In production: replace with Redis or a database-backed store.
_active_tokens: dict[str, str] = {}

# ── Data directories ─────────────────────────────────────────
# All user data is stored in flat JSON files locally.
# In production, replace with a real database.
DATA_DIR      = os.path.join(os.path.dirname(__file__), "data")
USERS_FILE    = os.path.join(DATA_DIR, "users.json")
SEARCHES_FILE = os.path.join(DATA_DIR, "searches.json")
TRAINING_FILE = os.path.join(DATA_DIR, "training_pairs.json")

os.makedirs(DATA_DIR, exist_ok=True)

def _read_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ── Token helpers ─────────────────────────────────────────────
def _issue_token(email: str) -> str:
    """Generate a new UUID session token and store it."""
    token = str(uuid.uuid4())
    _active_tokens[token] = email
    return token

def _validate_token() -> tuple[str | None, object | None]:
    """
    Reads the Authorization header, validates the token.
    Returns (email, None) on success or (None, error_response) on failure.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None, (jsonify({"error": "Session expired. Please sign in again.", "code": "TOKEN_MISSING"}), 401)

    token = auth_header[len("Bearer "):].strip()
    email = _active_tokens.get(token)
    if not email:
        return None, (jsonify({"error": "Session expired. Please sign in again.", "code": "TOKEN_INVALID"}), 401)

    return email, None

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
- Keep it concise (1–4 sentences unless absolutely needed).

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
        "coding":      "Use instruction-based prompting. Be precise, technical, solution-focused. Include constraints and expected output.",
        "content":     "Use role-based creative prompting. Focus on engagement, hooks, and audience psychology.",
        "explanation": "Use clear, step-by-step explanation style. Break concepts into simple parts. Avoid jargon.",
        "creative":    "Use imaginative prompting. Encourage storytelling, originality, and emotional depth.",
        "general":     "Use structured and balanced prompting. Ensure clarity, context, and usefulness.",
    }
    base = styles.get(intent, styles["general"])
    detail = "Keep it simple and direct." if length < 6 else "Expand with better detail and structure."
    return f"{base} {detail}"

# ── Health ────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": DEFAULT_MODEL, "product": "Prometix by Atimos AI"})

# ── Models ───────────────────────────────────────────────────
@app.route("/models", methods=["GET"])
def list_models():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return jsonify({"models": models})
    except Exception:
        return jsonify({"models": [DEFAULT_MODEL]})

# ── Generate ─────────────────────────────────────────────────
@app.route("/generate", methods=["POST"])
def generate():
    # Validate session token
    email, err = _validate_token()
    if err:
        return err

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON in request body."}), 400

    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    model  = (data.get("model") or DEFAULT_MODEL).strip()
    intent = detect_intent(user_message)
    style  = get_prompt_style(intent, user_message)
    length = len(user_message.split())

    full_prompt = f"""[SYSTEM]
{SYSTEM_PROMPT}

INTERNAL STYLE GUIDE:
{style}

LANGUAGE:
- Always output in clean English regardless of input language.
- Understand mixed or regional language before rewriting.

COMPLEXITY:
Generate a {"simple" if length < 6 else "advanced"}-level prompt.

[USER]
{user_message}

[ASSISTANT]
"""

    payload = {
        "model":  model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature":    0.4,
            "top_p":          0.9,
            "repeat_penalty": 1.1,
            "stop":           ["\n\n\n"],
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach the Atimos AI backend. Run: ollama serve"}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": f"Backend timed out after {REQUEST_TIMEOUT}s. Try a smaller model."}), 504
    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"Backend error: {e.response.status_code}"}), 502

    try:
        ollama_data = resp.json()
    except Exception:
        return jsonify({"error": "Backend returned unreadable data."}), 502

    ai_text = (ollama_data.get("response") or "").strip()
    if not ai_text:
        return jsonify({"error": "Backend returned an empty response. Try a different model or rephrase."}), 502

    return jsonify({
        "response": ai_text,
        "model":    ollama_data.get("model", model),
        "done":     ollama_data.get("done", True),
    })

# ── Auth: register ────────────────────────────────────────────
@app.route("/auth/register", methods=["POST"])
def auth_register():
    """
    Registers a new user.
    Accepts: { name, email, password }
    Stores hashed password in data/users.json.
    Returns: public profile + session token on success.
    """
    body     = request.get_json(silent=True) or {}
    name     = (body.get("name")     or "").strip()
    email    = (body.get("email")    or "").strip().lower()
    password = (body.get("password") or "").strip()

    # Basic validation
    if not name:
        return jsonify({"error": "Name is required."}), 400
    if not email or "@" not in email:
        return jsonify({"error": "A valid email address is required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    users = _read_json(USERS_FILE, {})

    if email in users:
        return jsonify({"error": "An account with this email already exists."}), 409

    ts = datetime.datetime.utcnow().isoformat()
    users[email] = {
        "email":      email,
        "name":       name,
        "password":   generate_password_hash(password),  # hashed — never stored plain
        "provider":   "email",
        "avatar":     name[0].upper(),
        "created_at": ts,
        "firstSeen":  ts,
        "lastSeen":   ts,
        "loginCount": 0,
        "consent":    False,
    }

    _write_json(USERS_FILE, users)
    print(f"  [REGISTER] {email}  —  {ts}")

    # Issue session token immediately — user is logged in after registration
    token = _issue_token(email)

    return jsonify({
        "status": "ok",
        "token":  token,
        "user":   _public_profile(users[email]),
    }), 201


# ── Auth: login ───────────────────────────────────────────────
@app.route("/auth/login", methods=["POST"])
def auth_login():
    """
    Authenticates an existing user with hashed password verification.
    Accepts: { email, password }
    Returns: public profile + session token on success.
    """
    body     = request.get_json(silent=True) or {}
    email    = (body.get("email")    or "").strip().lower()
    password = (body.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    users = _read_json(USERS_FILE, {})
    user  = users.get(email)

    # check_password_hash works for hashed passwords;
    # also handles legacy plain-text entries gracefully via fallback below.
    password_ok = False
    if user:
        stored = user.get("password", "")
        if stored.startswith("pbkdf2:") or stored.startswith("scrypt:"):
            # Properly hashed — use werkzeug verifier
            password_ok = check_password_hash(stored, password)
        else:
            # Legacy plain-text entry — accept and upgrade to hash on the fly
            if stored == password:
                password_ok = True
                user["password"] = generate_password_hash(password)

    if not user or not password_ok:
        return jsonify({"error": "Invalid email or password."}), 401

    # Update last seen and login count
    ts = datetime.datetime.utcnow().isoformat()
    user["lastSeen"]   = ts
    user["loginCount"] = user.get("loginCount", 0) + 1
    _write_json(USERS_FILE, users)

    # Issue a fresh session token
    token = _issue_token(email)

    print(f"  [LOGIN]  {email}  (email)  —  {ts}")

    return jsonify({
        "status": "ok",
        "token":  token,
        "user":   _public_profile(user),
    })


# ── Auth: logout ──────────────────────────────────────────────
@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    """Invalidates the session token so it can no longer be used."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):].strip()
        _active_tokens.pop(token, None)
    return jsonify({"status": "ok"})


def _public_profile(user):
    """Returns only the fields safe to send to the frontend."""
    return {
        "email":      user.get("email", ""),
        "name":       user.get("name", ""),
        "avatar":     user.get("avatar", user.get("name", "U")[0].upper()),
        "provider":   user.get("provider", "email"),
        "consent":    user.get("consent", False),
        "created_at": user.get("created_at", ""),
    }


# ── User: consent ─────────────────────────────────────────────
@app.route("/user/consent", methods=["POST"])
def user_consent():
    email, err = _validate_token()
    if err:
        return err

    body    = request.get_json(silent=True) or {}
    consent = bool(body.get("consent"))

    users = _read_json(USERS_FILE, {})
    if email in users:
        users[email]["consent"] = consent
        _write_json(USERS_FILE, users)

    print(f"  [CONSENT] {email}  →  {'YES' if consent else 'NO'}")
    return jsonify({"status": "ok"})

# ── User: search log ─────────────────────────────────────────
@app.route("/user/search", methods=["POST"])
def user_search():
    """
    Logs every search query with the user's email and timestamp.
    Requires valid session token.
    """
    email, err = _validate_token()
    if err:
        return err

    body  = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    ts    = datetime.datetime.utcnow().isoformat()

    if not query:
        return jsonify({"status": "ignored"})

    searches = _read_json(SEARCHES_FILE, [])
    searches.append({"email": email, "query": query, "ts": ts})

    # Keep last 5000 searches to avoid unbounded growth
    if len(searches) > 5000:
        searches = searches[-5000:]

    _write_json(SEARCHES_FILE, searches)
    return jsonify({"status": "ok"})

# ── Training: pair ────────────────────────────────────────────
@app.route("/training/pair", methods=["POST"])
def training_pair():
    """
    Stores a raw→prompt pair from users who have given training consent.
    Requires valid session token.
    """
    email, err = _validate_token()
    if err:
        return err

    body = request.get_json(silent=True) or {}
    inp  = (body.get("input")  or "").strip()
    out  = (body.get("output") or "").strip()
    ts   = datetime.datetime.utcnow().isoformat()

    # Only store if user has consent on record
    users = _read_json(USERS_FILE, {})
    if not users.get(email, {}).get("consent"):
        return jsonify({"status": "no_consent"})

    if not inp or not out:
        return jsonify({"status": "ignored"})

    pairs = _read_json(TRAINING_FILE, [])
    pairs.append({"email": email, "input": inp, "output": out, "ts": ts})

    if len(pairs) > 10000:
        pairs = pairs[-10000:]

    _write_json(TRAINING_FILE, pairs)
    return jsonify({"status": "stored"})

# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║   Prometix by Atimos AI  —  v2.2         ║")
    print("  ║   Mode : Prompt Engineering Tool         ║")
    print("  ╚══════════════════════════════════════════╝")
    print(f"\n  Ollama endpoint : {OLLAMA_URL}")
    print(f"  Default model   : {DEFAULT_MODEL}")
    print(f"  Data directory  : {DATA_DIR}")
    print(f"  Auth endpoints  : POST /auth/register  POST /auth/login  POST /auth/logout")
    print(f"  Security        : bcrypt password hashing · UUID session tokens")
    print(f"  Frontend URL    : open login.html in browser")
    print("\n  Tracking : users.json · searches.json · training_pairs.json\n")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)