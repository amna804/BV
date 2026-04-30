"""
WiFi Leak Detection — Flask File Server
========================================
Folder structure expected (exactly as you have it):

  VRBV1/
  ├── app.py
  ├── index.html
  ├── models/
  │   ├── detection/
  │   │   ├── model.json
  │   │   ├── group1-shard1of1.bin
  │   │   └── scaler_params.json
  │   └── localization/
  │       ├── model.json
  │       ├── group1-shard1of1.bin
  │       └── scaler_params.json
  └── test_data/
      └── *.csv
"""

from flask import Flask, jsonify, send_from_directory, abort
from flask_cors import CORS
import os
import glob

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(BASE_DIR, "models")
DET_DIR       = os.path.join(MODELS_DIR, "detection")
LOC_DIR       = os.path.join(MODELS_DIR, "localization")
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")

app = Flask(__name__)
CORS(app)

# ── Startup check ────────────────────────────────────────────────────────────
print("\n" + "=" * 54)
print("  WiFi Leak Detection  —  http://localhost:5000")
print("=" * 54)
for label, path in [
    ("Detection model   ", os.path.join(DET_DIR, "model.json")),
    ("Detection scaler  ", os.path.join(DET_DIR, "scaler_params.json")),
    ("Loc model         ", os.path.join(LOC_DIR, "model.json")),
    ("Loc scaler        ", os.path.join(LOC_DIR, "scaler_params.json")),
]:
    status = "✓" if os.path.exists(path) else "❌ MISSING"
    print(f"  {label}: {status}  ({path})")
print("=" * 54 + "\n")


# ── Index ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


# ── Detection model + scaler ─────────────────────────────────────────────────
@app.route("/detection_model/<path:filename>")
def serve_detection_model(filename):
    safe = os.path.realpath(os.path.join(DET_DIR, filename))
    if not safe.startswith(DET_DIR):
        abort(403)
    return send_from_directory(DET_DIR, filename)

# Detection scaler lives INSIDE models/detection/
@app.route("/scaler_params.json")
def serve_det_scaler():
    return send_from_directory(DET_DIR, "scaler_params.json")


# ── Localization model + scaler ──────────────────────────────────────────────
@app.route("/localization_model/<path:filename>")
def serve_localization_model(filename):
    safe = os.path.realpath(os.path.join(LOC_DIR, filename))
    if not safe.startswith(LOC_DIR):
        abort(403)
    return send_from_directory(LOC_DIR, filename)

# Localization scaler lives INSIDE models/localization/
@app.route("/loc_scaler_params.json")
def serve_loc_scaler():
    return send_from_directory(LOC_DIR, "scaler_params.json")


# ── Test CSV files ───────────────────────────────────────────────────────────
@app.route("/list-tests")
def list_tests():
    if not os.path.isdir(TEST_DATA_DIR):
        return jsonify({"files": [], "warning": "test_data/ not found"}), 200
    files = sorted(
        os.path.basename(f)
        for f in glob.glob(os.path.join(TEST_DATA_DIR, "*.csv"))
    )
    return jsonify({"files": files})

@app.route("/test-data/<path:filename>")
def serve_test_data(filename):
    safe = os.path.realpath(os.path.join(TEST_DATA_DIR, filename))
    if not safe.startswith(TEST_DATA_DIR):
        abort(403)
    return send_from_directory(TEST_DATA_DIR, filename)


# ── Health / debug ───────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "detection_model":     os.path.exists(os.path.join(DET_DIR, "model.json")),
        "detection_scaler":    os.path.exists(os.path.join(DET_DIR, "scaler_params.json")),
        "localization_model":  os.path.exists(os.path.join(LOC_DIR, "model.json")),
        "localization_scaler": os.path.exists(os.path.join(LOC_DIR, "scaler_params.json")),
        "test_files": sorted(
            os.path.basename(f)
            for f in glob.glob(os.path.join(TEST_DATA_DIR, "*.csv"))
        ),
    })

@app.route("/debug")
def debug():
    def ls(path):
        if not os.path.isdir(path): return f"MISSING: {path}"
        return sorted(os.listdir(path))
    return jsonify({
        "det_dir_files": ls(DET_DIR),
        "loc_dir_files": ls(LOC_DIR),
        "test_csv_files": ls(TEST_DATA_DIR),
        "urls": {
            "/detection_model/model.json":    "→ models/detection/model.json",
            "/detection_model/group1-shard1of1.bin": "→ models/detection/group1-shard1of1.bin",
            "/scaler_params.json":            "→ models/detection/scaler_params.json",
            "/localization_model/model.json": "→ models/localization/model.json",
            "/loc_scaler_params.json":        "→ models/localization/scaler_params.json",
        }
    })


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
