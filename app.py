from flask import Flask, request, jsonify
from flask_cors import CORS  # Importing CORS
import numpy as np
import os
import json

# Try tflite_runtime first, fallback to TensorFlow's Lite interpreter
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    from tensorflow.lite import Interpreter  # type: ignore

app = Flask(__name__)

# Enabling CORS for all origins (or specify the allowed origins)
CORS(app, resources={r"/*": {"origins": [
    "http://127.0.0.1:5500",
    "http://localhost",  # <-- ADD THIS
    "https://kimayco.github.io",
    "https://kimayco.github.io/mediapipetest1",
    "http://localhost/capstone/signspeak2.6/translate.php"
]}})
  # Allow requests from your local HTML page

# Paths (can be overridden via environment variables on Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_PATH = os.getenv("TFLITE_PATH", os.path.join(BASE_DIR, "STGCN_model.tflite"))
LABELS_PATH = os.getenv("LABELS_PATH", os.path.join(BASE_DIR, "labels.txt"))
PREPROCESS_STATS = os.getenv("PREPROCESS_STATS", os.path.join(BASE_DIR, "preprocess_stats.npz"))

# Load label map (index -> label)
label_map = {}
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            i, name = line.split(",", 1)
            label_map[int(i)] = name

# Load normalization and shape metadata if available
norm_stats = None
if os.path.exists(PREPROCESS_STATS):
    try:
        z = np.load(PREPROCESS_STATS)
        norm_stats = {
            "feat_mean": z.get("feat_mean"),
            "feat_std": z.get("feat_std"),
            "HAND_START": int(z.get("HAND_START", np.array([22]))[()]),
            "HAND_DIM": int(z.get("HAND_DIM", np.array([84]))[()]),
            "TOTAL_NODES": int(z.get("TOTAL_NODES", np.array([42]))[()]),
            "JOINT_FEATS": int(z.get("JOINT_FEATS", np.array([2]))[()]),
            "EXPECTED_FRAMES": int(z.get("EXPECTED_FRAMES", np.array([9]))[()]),
            "FEATURE_DIM": int(z.get("FEATURE_DIM", np.array([106]))[()]),
        }
    except Exception:
        norm_stats = None

# Load the TFLite model
interpreter = Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def pad_or_truncate_time(arr: np.ndarray, expected_frames: int) -> np.ndarray:
    t = arr.shape[0]
    if t < expected_frames:
        pad = np.zeros((expected_frames - t, arr.shape[1]), dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)
    if t > expected_frames:
        return arr[:expected_frames]
    return arr

def normalize_if_possible(hand_flat: np.ndarray) -> np.ndarray:
    if not norm_stats or norm_stats.get("feat_mean") is None or norm_stats.get("feat_std") is None:
        return hand_flat
    feat_mean = norm_stats["feat_mean"]
    feat_std = norm_stats["feat_std"]
    if hand_flat.shape[1] != feat_mean.shape[1]:
        return hand_flat
    safe_std = np.where(feat_std < 1e-6, 1e-6, feat_std)
    return (hand_flat - feat_mean) / safe_std

def legacy_24_to_42x2(row_24: np.ndarray) -> np.ndarray:
    # Map legacy 24 features (wrist+fingertips x,y for 2 hands) into 42 nodes x 2 features
    # Assumes order per hand: indices [0,4,8,12,16,20] then other hand
    selected = [0, 4, 8, 12, 16, 20]
    out = np.zeros((42, 2), dtype=np.float32)
    # hand 0
    for k in range(6):
        x = row_24[k * 2]
        y = row_24[k * 2 + 1]
        out[selected[k], 0] = x
        out[selected[k], 1] = y
    # hand 1
    base = 12
    for k in range(6):
        x = row_24[base + k * 2]
        y = row_24[base + k * 2 + 1]
        out[21 + selected[k], 0] = x
        out[21 + selected[k], 1] = y
    return out.reshape(-1)  # flatten back to 84

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        seq = np.array(data.get("data", []), dtype=np.float32)
        if seq.ndim != 2:
            return jsonify({"error": f"Invalid input. Expected 2D list (T,F), got shape {seq.shape}"}), 400

        # Determine expected frames and feature handling from stats or model
        expected_time = int(norm_stats["EXPECTED_FRAMES"]) if norm_stats else int(input_details[0]["shape"][1])
        seq = pad_or_truncate_time(seq, expected_time)

        F = seq.shape[1]
        total_nodes = int(norm_stats["TOTAL_NODES"]) if norm_stats else 42
        # Handle formats: 106 (old), multiple of 42, or legacy 24
        if F == 106:
            hand_start = int(norm_stats["HAND_START"]) if norm_stats else 22
            hand_dim = int(norm_stats["HAND_DIM"]) if norm_stats else 84
            hand_flat = seq[:, hand_start:hand_start + hand_dim]
            hand_flat = normalize_if_possible(hand_flat)
            joint_feats = hand_dim // total_nodes
        elif F % total_nodes == 0:
            hand_flat = seq
            hand_flat = normalize_if_possible(hand_flat)
            joint_feats = F // total_nodes
        elif F == 24:
            # Map to 84 (42x2) then proceed
            mapped = np.stack([legacy_24_to_42x2(seq[t]) for t in range(seq.shape[0])], axis=0)
            hand_flat = normalize_if_possible(mapped)
            joint_feats = 2
        else:
            return jsonify({"error": f"Unsupported feature dimension F={F}. Expected 106, multiple of 42, or 24."}), 400

        # Reshape to (1, T, N, C) to match ST-GCN TFLite
        try:
            hand_tensor = hand_flat.reshape(1, expected_time, total_nodes, joint_feats).astype(np.float32)
        except Exception:
            return jsonify({"error": f"Cannot reshape to (1,{expected_time},{total_nodes},{joint_feats}) from {hand_flat.shape}"}), 400

        # Some exported models may expect a different static shape; adjust if needed
        model_input_shape = input_details[0]["shape"]
        if tuple(model_input_shape) != hand_tensor.shape:
            # Try to reshape time or channels if model is static
            try:
                hand_tensor = hand_tensor.astype(np.float32)
                # If model expects different time, pad/truncate along axis=1
                mt = int(model_input_shape[1])
                if mt != expected_time:
                    seq2 = pad_or_truncate_time(hand_flat, mt)
                    hand_tensor = seq2.reshape(1, mt, total_nodes, joint_feats).astype(np.float32)
            except Exception:
                pass

        interpreter.set_tensor(input_details[0]['index'], hand_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_idx = int(np.argmax(output_data[0]))
        confidence = float(output_data[0][predicted_idx])

        return jsonify({
            "prediction": label_map.get(predicted_idx, "Unknown"),
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "🧠 TFLite Model Server Running"

@app.route("/health", methods=["GET"])
def health():
    try:
        _ = input_details[0]["shape"]
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
