#!/usr/bin/env python3
"""Minimal Flask server for chatPF2E — a sessionless web UI over ChatAgent."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

FEEDBACK_LOG_PATH = Path(os.getenv("FEEDBACK_LOG_PATH", "data/chat_feedback.jsonl"))
_feedback_lock = threading.Lock()

app = Flask(__name__, static_folder="gm_agent/static/chatpf2e", static_url_path="")

# In-memory session store: {session_id: {"agent": ChatAgent, "last_seen": float}}
_sessions: dict = {}
_sessions_lock = threading.Lock()
SESSION_TTL_SECONDS = 1800  # 30 minutes idle expiry


def _reap_sessions() -> None:
    """Background thread: remove idle sessions."""
    while True:
        time.sleep(300)  # check every 5 minutes
        cutoff = time.time() - SESSION_TTL_SECONDS
        with _sessions_lock:
            stale = [k for k, v in _sessions.items() if v["last_seen"] < cutoff]
            for k in stale:
                try:
                    _sessions[k]["agent"].close()
                except Exception:  # pylint: disable=broad-except
                    pass
                del _sessions[k]


threading.Thread(target=_reap_sessions, daemon=True).start()


def _get_or_create_agent(session_id: str):
    """Return the ChatAgent for this session, creating one if needed."""
    from gm_agent.chat import ChatAgent  # lazy import — avoids startup cost if not used

    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = {"agent": ChatAgent(), "last_seen": time.time()}
        else:
            _sessions[session_id]["last_seen"] = time.time()
        return _sessions[session_id]["agent"]


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "message required"}), 400

    session_id = request.headers.get("X-Session-ID") or str(uuid.uuid4())
    agent = _get_or_create_agent(session_id)

    try:
        response = agent.chat(data["message"])
        return jsonify({"response": response, "session_id": session_id})
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    """Explicit reset — clears conversation history for this session."""
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        with _sessions_lock:
            if session_id in _sessions:
                try:
                    _sessions[session_id]["agent"].close()
                except Exception:  # pylint: disable=broad-except
                    pass
                del _sessions[session_id]
    return jsonify({"ok": True})


@app.route("/api/feedback", methods=["POST"])
def feedback():
    """Record a thumbs-up or thumbs-down on an agent response."""
    data = request.get_json()
    if not data or data.get("rating") not in ("up", "down"):
        return jsonify({"error": "rating must be 'up' or 'down'"}), 400

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": request.headers.get("X-Session-ID", "unknown"),
        "rating": data["rating"],
        "preview": (data.get("preview") or "")[:300],
    }

    try:
        with _feedback_lock:
            FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # non-fatal — feedback is best-effort

    return jsonify({"ok": True})


@app.route("/api/health")
def health():
    with _sessions_lock:
        count = len(_sessions)
    return jsonify({"ok": True, "active_sessions": count})


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="chatPF2E web UI")
    p.add_argument("--port", type=int, default=5001)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    print(f"chatPF2E running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
