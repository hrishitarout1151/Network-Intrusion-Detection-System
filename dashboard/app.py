"""
app.py -- SentinelIQ dashboard backend.

Serves the pre-computed detection payload (data/dashboard_payload.json) and
exposes a /api/simulate endpoint that actually re-runs the full pipeline
(new synthetic traffic -> features -> EWMA -> retrain -> hybrid scoring) so
the "Run New Simulation" button in the UI is a real end-to-end execution,
not a canned demo toggle.
"""

import json
import os
import subprocess
import sys

from flask import Flask, jsonify, render_template

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
PAYLOAD_PATH = os.path.join(DATA_DIR, "dashboard_payload.json")

app = Flask(__name__)


def load_payload():
    if not os.path.exists(PAYLOAD_PATH):
        return None
    with open(PAYLOAD_PATH) as f:
        return json.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard")
def api_dashboard():
    payload = load_payload()
    if payload is None:
        return jsonify({"error": "No pipeline output yet. POST /api/simulate first."}), 404
    return jsonify(payload)


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Re-runs the full pipeline synchronously: new synthetic traffic batch,
    fresh feature extraction, fresh EWMA state, model retrain, hybrid scoring."""
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "run_pipeline.py")],
        capture_output=True, text=True, cwd=ROOT,
    )
    if result.returncode != 0:
        return jsonify({"error": "Pipeline failed", "log": result.stderr[-4000:]}), 500
    return jsonify(load_payload())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=True)
