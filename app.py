from flask import Flask, request, jsonify
from flask_cors import CORS  # Importing CORS
import numpy as np
import os
import traceback
import json

# Use TensorFlow Lite interpreter from TensorFlow (more portable on Render)
import tensorflow as tf  # type: ignore

app = Flask(__name__)

# Enabling CORS for all origins (or specify the allowed origins)
CORS(app, resources={r"/*": {"origins": [
    "http://127.0.0.1:5500",
    "http://localhost",  # <-- ADD THIS
    "https://kimayco.github.io",
    "https://kimayco.github.io/mediapipetest1",
    "http://localhost/capstone/signspeak2.6/translate.php",
    "file:///C:/Users/admin/Desktop/advance_collectorV1.html"
    
]}})
  # Allow requests from your local HTML page

# Paths (can be overridden via environment variables on Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.getenv("LABELS_PATH", os.path.join(BASE_DIR, "labels.txt"))
PREPROCESS_STATS = os.getenv("PREPROCESS_STATS", os.path.join(BASE_DIR, "preprocess_stats.npz"))

# Support ensembling the 3 new models by default. Override with TFLITE_PATHS (comma-separated)
_default_models = [
    os.path.join(BASE_DIR, "ST-GCN (datasets 3.0 & 4.0) V1.tflite"),
    os.path.join(BASE_DIR, "ST-GCN (datasets 3.0) V1.tflite"),
    os.path.join(BASE_DIR, "ST-GCN (datasets 4.0) V1.tflite"),
]
TFLITE_PATHS = os.getenv("TFLITE_PATHS", "").strip()
if TFLITE_PATHS:
    MODEL_PATHS = [p.strip() for p in TFLITE_PATHS.split(",") if p.strip()]
else:
    MODEL_PATHS = _default_models

# Model weights: default to [0.5, 0.25, 0.25] aligned with MODEL_PATHS order above.
# Can override via TFLITE_WEIGHTS env (comma-separated floats) matching number of models.
_default_weights = [0.5, 0.25, 0.25]
TFLITE_WEIGHTS = os.getenv("TFLITE_WEIGHTS", "").strip()
if TFLITE_WEIGHTS:
    try:
        MODEL_WEIGHTS = [float(w.strip()) for w in TFLITE_WEIGHTS.split(",") if w.strip()]
    except Exception:
        MODEL_WEIGHTS = _default_weights
else:
    MODEL_WEIGHTS = _default_weights

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

# Load the TFLite models lazily to ensure environment is ready
interpreters = []
input_details_list = []
output_details_list = []
MODEL_WEIGHTS_NORM = []

def ensure_interpreters():
    global interpreters, input_details_list, output_details_list, MODEL_WEIGHTS_NORM
    if interpreters:
        return
    if not MODEL_PATHS:
        raise FileNotFoundError("No model paths provided. Set TFLITE_PATHS or place default STGCN_model(1-4).tflite files.")
    for mp in MODEL_PATHS:
        if not os.path.exists(mp):
            raise FileNotFoundError(f"TFLite model not found at {mp}")
    for mp in MODEL_PATHS:
        itp = tf.lite.Interpreter(model_path=mp)
        itp.allocate_tensors()
        interpreters.append(itp)
        input_details_list.append(itp.get_input_details())
        output_details_list.append(itp.get_output_details())
    # Prepare normalized weights
    if len(MODEL_WEIGHTS) != len(MODEL_PATHS):
        # If mismatch, fall back to equal weights
        equal_w = 1.0 / float(len(MODEL_PATHS))
        MODEL_WEIGHTS_NORM = [equal_w for _ in MODEL_PATHS]
    else:
        s = sum(MODEL_WEIGHTS) or 1.0
        MODEL_WEIGHTS_NORM = [w / s for w in MODEL_WEIGHTS]

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

        # Ensure models are loaded (needed to inspect input shapes)
        ensure_interpreters()

        # Determine expected frames and feature handling from stats or first model
        first_input = input_details_list[0]
        expected_time = int(norm_stats["EXPECTED_FRAMES"]) if norm_stats else int(first_input[0]["shape"][1])
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

        # Run all models once with weighted aggregation (default weights 0.5, 0.25, 0.25)
        agg_probs = None
        top3_by_model = []
        weights_to_use = MODEL_WEIGHTS_NORM if MODEL_WEIGHTS_NORM else [1.0/len(interpreters)]*len(interpreters)
        for weight, itp, inp_det, out_det in zip(weights_to_use, interpreters, input_details_list, output_details_list):
            model_input_shape = inp_det[0]["shape"]
            tensor = hand_tensor
            if tuple(model_input_shape) != tensor.shape:
                try:
                    tensor = tensor.astype(np.float32)
                    mt = int(model_input_shape[1])
                    if mt != expected_time:
                        seq2 = pad_or_truncate_time(hand_flat, mt)
                        tensor = seq2.reshape(1, mt, total_nodes, joint_feats).astype(np.float32)
                except Exception:
                    pass
            itp.set_tensor(inp_det[0]['index'], tensor)
            itp.invoke()
            probs = itp.get_tensor(out_det[0]['index'])[0]
            top3_idx = np.argsort(-probs)[:3]
            top3_by_model.append([
                {"label": label_map.get(int(i), str(int(i))), "prob": float(probs[i])} for i in top3_idx
            ])
            weighted = weight * probs
            agg_probs = weighted if agg_probs is None else agg_probs + weighted

        # weighted combination already applied; get final
        final_idx = int(np.argmax(agg_probs))
        final_conf = float(agg_probs[final_idx])

        return jsonify({
            "prediction": label_map.get(final_idx, "Unknown"),
            "confidence": round(final_conf, 4),
            "models_top3": top3_by_model
        })

    except Exception as e:
        details = {"error": str(e)}
        try:
            details.update({
                "F": int(F) if 'F' in locals() else None,
                "expected_time": int(expected_time) if 'expected_time' in locals() else None,
                "total_nodes": int(total_nodes) if 'total_nodes' in locals() else None,
                "joint_feats": int(joint_feats) if 'joint_feats' in locals() else None,
                "hand_flat_shape": list(hand_flat.shape) if 'hand_flat' in locals() else None,
                "hand_tensor_shape": list(hand_tensor.shape) if 'hand_tensor' in locals() else None,
                "model_input_shape": list(input_details_list[0][0]["shape"]) if input_details_list else None,
            })
        except Exception:
            pass
        details["trace"] = traceback.format_exc()
        return jsonify(details), 500

@app.route("/", methods=["GET"])
def home():
    return "🧠 TFLite Model Server Running"

@app.route("/health", methods=["GET"])
def health():
    try:
        ensure_interpreters()
        info = {
            "status": "ok",
            "models_loaded": len(interpreters),
            "model_input_shapes": [ids[0]["shape"].tolist() for ids in input_details_list] if input_details_list else [],
            "weights": MODEL_WEIGHTS_NORM,
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)

