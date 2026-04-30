"""
WiFi Leak Detection — Flask File Server
========================================
Serves TF.js models, scalers, CSVs, and index.html.
All inference runs in the browser via TensorFlow.js.

AUTO-DISCOVERY: This server scans the project folder and
finds model folders + scaler files automatically — so your
actual folder/file names don't need to match exactly.

Visit  http://localhost:5000/debug  to see what was found.
"""

from flask import Flask, jsonify, send_from_directory, abort
from flask_cors import CORS
import os
import glob

# ── Base directory: folder containing this app.py ──────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")

app = Flask(__name__)
CORS(app)

# ── Auto-discovery helpers ──────────────────────────────────────────────────

def find_model_folder(hints, exclude=None):
    """
    Return the first subfolder of BASE_DIR that:
      • contains model.json
      • whose name (lowercased) contains any of the hint strings
    Falls back to the first model.json folder found if no hint matches.
    exclude: folder name to skip (so detection and localisation don't resolve identically)
    """
    candidates = []
    for entry in os.listdir(BASE_DIR):
        if entry == exclude:
            continue
        full = os.path.join(BASE_DIR, entry)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "model.json")):
            candidates.append(entry)

    # prefer hint match
    for entry in candidates:
        if any(h in entry.lower() for h in hints):
            return entry
    # fallback: first found
    return candidates[0] if candidates else None


def find_scaler_file(hints, exclude=None):
    """
    Return the first .json file in BASE_DIR whose name (lowercased)
    contains any of the hint strings.
    exclude: filename to skip.
    """
    for entry in sorted(os.listdir(BASE_DIR)):
        if entry == exclude or not entry.endswith(".json"):
            continue
        if any(h in entry.lower() for h in hints):
            return entry
    return None


# ── Discover at startup ─────────────────────────────────────────────────────
DET_MODEL_FOLDER = find_model_folder(["detect", "det"])
LOC_MODEL_FOLDER = find_model_folder(["loc", "local"], exclude=DET_MODEL_FOLDER)

DET_SCALER_FILE  = find_scaler_file(["det_scal", "csi_scal", "scaler_param"])
LOC_SCALER_FILE  = find_scaler_file(["loc_scal", "loc_scaler"], exclude=DET_SCALER_FILE)

# If scaler hints didn't match, grab the first two .json files that aren't model-related
if not DET_SCALER_FILE or not LOC_SCALER_FILE:
    json_files = [
        f for f in sorted(os.listdir(BASE_DIR))
        if f.endswith(".json") and "model" not in f.lower()
    ]
    if not DET_SCALER_FILE and len(json_files) >= 1:
        DET_SCALER_FILE = json_files[0]
    if not LOC_SCALER_FILE and len(json_files) >= 2:
        LOC_SCALER_FILE = json_files[1]

print("\n" + "=" * 54)
print("  WiFi Leak Detection  —  http://localhost:5000")
print("=" * 54)
print(f"  BASE DIR           : {BASE_DIR}")
print(f"  Detection model    : {DET_MODEL_FOLDER  or '❌ NOT FOUND'}")
print(f"  Localization model : {LOC_MODEL_FOLDER  or '❌ NOT FOUND'}")
print(f"  Detection scaler   : {DET_SCALER_FILE   or '❌ NOT FOUND'}")
print(f"  Localization scaler: {LOC_SCALER_FILE   or '❌ NOT FOUND'}")
print(f"  Test data dir      : {TEST_DATA_DIR}")
print("=" * 54)
print("  Open /debug in browser if you see 404 errors.\n")


# ── Index ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


# ── Model routes (always served at canonical URLs the frontend expects) ──────

@app.route("/detection_model/model.json")
def det_model_json():
    if not DET_MODEL_FOLDER:
        return jsonify({"error": "Detection model folder not found — check /debug"}), 404
    return send_from_directory(os.path.join(BASE_DIR, DET_MODEL_FOLDER), "model.json")

@app.route("/detection_model/<path:filename>")
def det_model_shard(filename):
    if not DET_MODEL_FOLDER:
        abort(404)
    return send_from_directory(os.path.join(BASE_DIR, DET_MODEL_FOLDER), filename)

@app.route("/localization_model/model.json")
def loc_model_json():
    if not LOC_MODEL_FOLDER:
        return jsonify({"error": "Localization model folder not found — check /debug"}), 404
    return send_from_directory(os.path.join(BASE_DIR, LOC_MODEL_FOLDER), "model.json")

@app.route("/localization_model/<path:filename>")
def loc_model_shard(filename):
    if not LOC_MODEL_FOLDER:
        abort(404)
    return send_from_directory(os.path.join(BASE_DIR, LOC_MODEL_FOLDER), filename)

@app.route("/scaler_params.json")
def det_scaler():
    if not DET_SCALER_FILE:
        return jsonify({"error": "Detection scaler not found — check /debug"}), 404
    return send_from_directory(BASE_DIR, DET_SCALER_FILE)

@app.route("/loc_scaler_params.json")
def loc_scaler():
    if not LOC_SCALER_FILE:
        return jsonify({"error": "Localization scaler not found — check /debug"}), 404
    return send_from_directory(BASE_DIR, LOC_SCALER_FILE)


# ── Test CSV routes ──────────────────────────────────────────────────────────

@app.route("/list-tests")
def list_tests():
    if not os.path.isdir(TEST_DATA_DIR):
        return jsonify({"files": [], "warning": "test_data/ folder not found"}), 200
    files = sorted(
        os.path.basename(f)
        for f in glob.glob(os.path.join(TEST_DATA_DIR, "*.csv"))
    )
    return jsonify({"files": files})

@app.route("/test-data/<path:filename>")
def serve_test_data(filename):
    if not os.path.isdir(TEST_DATA_DIR):
        abort(404)
    safe = os.path.realpath(os.path.join(TEST_DATA_DIR, filename))
    if not safe.startswith(TEST_DATA_DIR):
        abort(403)
    return send_from_directory(TEST_DATA_DIR, filename)


# ── Debug endpoint — open in browser to diagnose 404s ───────────────────────

@app.route("/debug")
def debug():
    def scan(path):
        out = []
        if not os.path.isdir(path):
            return out
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in sorted(dirs) if not d.startswith(".")]
            for f in sorted(files):
                out.append(os.path.relpath(os.path.join(root, f), BASE_DIR))
        return out

    # Everything in BASE_DIR (non-recursive top level)
    top_level = sorted(os.listdir(BASE_DIR))

    # Model folders
    model_folders = {
        entry: scan(os.path.join(BASE_DIR, entry))
        for entry in top_level
        if os.path.isdir(os.path.join(BASE_DIR, entry))
        and os.path.exists(os.path.join(BASE_DIR, entry, "model.json"))
    }

    csv_files = sorted(
        os.path.basename(f)
        for f in glob.glob(os.path.join(TEST_DATA_DIR, "*.csv"))
    ) if os.path.isdir(TEST_DATA_DIR) else []

    return jsonify({
        "instructions": "Use this to find your real file/folder names, then update the mappings if needed.",
        "base_dir": BASE_DIR,
        "top_level_entries": top_level,
        "resolved": {
            "detection_model_folder":    DET_MODEL_FOLDER,
            "localization_model_folder": LOC_MODEL_FOLDER,
            "detection_scaler_file":     DET_SCALER_FILE,
            "localization_scaler_file":  LOC_SCALER_FILE,
        },
        "canonical_urls_served": {
            "frontend_requests":       "/detection_model/model.json  →  served from resolved folder",
            "detection_model":         f"/detection_model/model.json  →  {DET_MODEL_FOLDER}/model.json",
            "localization_model":      f"/localization_model/model.json  →  {LOC_MODEL_FOLDER}/model.json",
            "detection_scaler":        f"/scaler_params.json  →  {DET_SCALER_FILE}",
            "localization_scaler":     f"/loc_scaler_params.json  →  {LOC_SCALER_FILE}",
        },
        "model_folders_found":  model_folders,
        "test_csv_files":       csv_files,
    })


# ── Health check ─────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "detection_model":    DET_MODEL_FOLDER is not None,
        "localization_model": LOC_MODEL_FOLDER is not None,
        "detection_scaler":   DET_SCALER_FILE  is not None,
        "localization_scaler":LOC_SCALER_FILE  is not None,
        "test_files": sorted(
            os.path.basename(f)
            for f in glob.glob(os.path.join(TEST_DATA_DIR, "*.csv"))
        ) if os.path.isdir(TEST_DATA_DIR) else [],
    })


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
