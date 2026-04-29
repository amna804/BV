from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
import glob

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the HTML frontend

# ==============================
# FILE PATHS (MODELS ONLY)
# ==============================
DET_MODEL = "final_detection_model.h5"
LOC_MODEL = "final_localization_model.h5"
DET_SCALER = "csi_scaler.pkl"
LOC_SCALER = "csi_loc_scaler.pkl"

# ==============================
# TEST DATA FOLDER
# ==============================
TEST_DATA_DIR = "test_data"

WINDOW_SIZE = 50
STEP_SIZE = 5
PROB_THRESHOLD = 0.85
MIN_WET_WINDOWS = 40

# ==============================
# CSI PARSER
# ==============================
def parse_csi(csi_str):
    try:
        vals = [int(x) for x in str(csi_str).replace('"', '').strip().split(',') if x.strip()]
        return vals if len(vals) == 384 else None
    except:
        return None

# ==============================
# LIST AVAILABLE TEST FILES
# ==============================
@app.route("/list-tests", methods=["GET"])
def list_tests():
    """Return all available CSV test files in test_data/ folder."""
    pattern = os.path.join(TEST_DATA_DIR, "*.csv")
    files = glob.glob(pattern)
    names = [os.path.basename(f) for f in sorted(files)]
    return jsonify({"files": names})

# ==============================
# RUN TEST — AUTO-LOADS FROM test_data/
# ==============================
@app.route("/run-test", methods=["POST"])
def run_test():
    """
    Accepts optional JSON body: { "filename": "Wall 8 2ft TESTING.csv" }
    If no filename provided, uses the first CSV found in test_data/.
    """

    # 1. DETERMINE WHICH TEST FILE TO USE
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", None)

    if filename:
        test_path = os.path.join(TEST_DATA_DIR, filename)
    else:
        # Auto-pick first available CSV
        pattern = os.path.join(TEST_DATA_DIR, "*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            return jsonify({"error": f"No CSV files found in '{TEST_DATA_DIR}/' folder"}), 404
        test_path = files[0]
        filename = os.path.basename(test_path)

    if not os.path.exists(test_path):
        return jsonify({"error": f"Test file not found: {filename}"}), 404

    # 2. CHECK MODELS EXIST
    for model_file in [DET_MODEL, LOC_MODEL, DET_SCALER, LOC_SCALER]:
        if not os.path.exists(model_file):
            return jsonify({"error": f"Model file missing: {model_file}"}), 500

    # 3. LOAD MODELS
    try:
        det_model = tf.keras.models.load_model(DET_MODEL)
        loc_model = tf.keras.models.load_model(LOC_MODEL, compile=False)

        with open(DET_SCALER, "rb") as f:
            scaler_det = pickle.load(f)
        with open(LOC_SCALER, "rb") as f:
            scaler_loc = pickle.load(f)
    except Exception as e:
        return jsonify({"error": f"Failed to load models: {str(e)}"}), 500

    # 4. LOAD & PROCESS TEST DATA
    try:
        df = pd.read_csv(test_path)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 500

    if "csi_data" not in df.columns:
        return jsonify({"error": "CSV must have a 'csi_data' column"}), 400

    df["csi_parsed"] = df["csi_data"].apply(parse_csi)
    csi_matrix = np.array(df.dropna(subset=["csi_parsed"])["csi_parsed"].tolist())

    if len(csi_matrix) < WINDOW_SIZE:
        return jsonify({"error": f"Not enough valid CSI packets. Found {len(csi_matrix)}, need {WINDOW_SIZE}."}), 400

    # 5. RUN DETECTION + LOCALIZATION
    probs = []
    coords = []

    for i in range(0, len(csi_matrix) - WINDOW_SIZE + 1, STEP_SIZE):
        window = csi_matrix[i:i + WINDOW_SIZE]

        in_det = scaler_det.transform(window.reshape(-1, 384)).reshape(1, WINDOW_SIZE, 384)
        p = float(det_model.predict(in_det, verbose=0)[0][0])
        probs.append(p)

        if p > PROB_THRESHOLD:
            in_loc = scaler_loc.transform(window.reshape(-1, 384)).reshape(1, WINDOW_SIZE, 384)
            xy = loc_model.predict(in_loc, verbose=0)[0]
            coords.append([float(xy[0]), float(xy[1])])

    return jsonify({
        "is_wet": len(coords) >= MIN_WET_WINDOWS,
        "coords": coords,
        "total_windows": len(probs),
        "test_file": filename,
        "wet_windows": len(coords)
    })

# ==============================
# HEALTH CHECK
# ==============================
@app.route("/health", methods=["GET"])
def health():
    test_files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "*.csv")))
    models_ok = all(os.path.exists(f) for f in [DET_MODEL, LOC_MODEL, DET_SCALER, LOC_SCALER])
    return jsonify({
        "status": "ok",
        "models_ready": models_ok,
        "test_files": [os.path.basename(f) for f in test_files]
    })

# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
