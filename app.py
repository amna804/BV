from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os

app = Flask(__name__)

# ==============================
# FILE PATHS (MODELS ONLY)
# ==============================
DET_MODEL = "final_detection_model.h5"
LOC_MODEL = "final_localization_model.h5"
DET_SCALER = "csi_scaler.pkl"
LOC_SCALER = "csi_loc_scaler.pkl"

WINDOW_SIZE = 50
STEP_SIZE = 5
PROB_THRESHOLD = 0.85
MIN_WET_WINDOWS = 40

# ==============================
# CSI PARSER
# ==============================
def parse_csi(csi_str):
    try:
        vals = [int(x) for x in str(csi_str).replace('"','').split(',') if x]
        return vals if len(vals) == 384 else None
    except:
        return None

# ==============================
# UPLOAD + TEST API
# ==============================
@app.route("/run-test", methods=["POST"])
def run_test():

    # 1. CHECK FILE FROM REQUEST
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    df = pd.read_csv(file)

    # 2. LOAD MODELS
    det_model = tf.keras.models.load_model(DET_MODEL)
    loc_model = tf.keras.models.load_model(LOC_MODEL, compile=False)

    with open(DET_SCALER, "rb") as f:
        scaler_det = pickle.load(f)

    with open(LOC_SCALER, "rb") as f:
        scaler_loc = pickle.load(f)

    # 3. PROCESS DATA
    df["csi_parsed"] = df["csi_data"].apply(parse_csi)
    csi_matrix = np.array(df.dropna(subset=["csi_parsed"])["csi_parsed"].tolist())

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
        "total_windows": len(probs)
    })

# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)