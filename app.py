from flask import Flask, request, jsonify
from flask_cors import CORS  # Importing CORS
import numpy as np
import os
import traceback
import json

# Use TensorFlow Lite interpreter from TensorFlow (more portable on Render)
import tensorflow as tf  # type: ignore

app = Flask(__name__)

# Enabling CORS for all origins (allows file:// pages with Origin: null and localhost)
CORS(app, resources={r"/*": {"origins": "*"}})
  # Allow requests from any origin; do not use credentials

# Paths (can be overridden via environment variables on Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.getenv("LABELS_PATH", os.path.join(BASE_DIR, "labels.txt"))
PREPROCESS_STATS = os.getenv("PREPROCESS_STATS", os.path.join(BASE_DIR, "preprocess_stats.npz"))

# Use dual ST-GCN models (old and new)
OLD_MODEL_PATH = os.getenv("OLD_TFLITE_PATH", os.path.join(BASE_DIR, "OLD_ST-GCN.tflite"))
NEW_MODEL_PATH = os.getenv("NEW_TFLITE_PATH", os.path.join(BASE_DIR, "NEW_ST-GCN.tflite"))

# Weighting for ensemble prediction (new model gets 65%, old model gets 35%)
NEW_MODEL_WEIGHT = 0.65
OLD_MODEL_WEIGHT = 0.35

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
old_interpreter = None
old_input_details = None
old_output_details = None

new_interpreter = None
new_input_details = None
new_output_details = None

def ensure_interpreters():
    global old_interpreter, old_input_details, old_output_details
    global new_interpreter, new_input_details, new_output_details
    
    if old_interpreter is not None and new_interpreter is not None:
        return
    
    # Load old model
    if not os.path.exists(OLD_MODEL_PATH):
        raise FileNotFoundError(f"Old TFLite model not found at {OLD_MODEL_PATH}")
    
    old_interpreter = tf.lite.Interpreter(model_path=OLD_MODEL_PATH)
    old_interpreter.allocate_tensors()
    old_input_details = old_interpreter.get_input_details()
    old_output_details = old_interpreter.get_output_details()
    
    # Load new model
    if not os.path.exists(NEW_MODEL_PATH):
        raise FileNotFoundError(f"New TFLite model not found at {NEW_MODEL_PATH}")
    
    new_interpreter = tf.lite.Interpreter(model_path=NEW_MODEL_PATH)
    new_interpreter.allocate_tensors()
    new_input_details = new_interpreter.get_input_details()
    new_output_details = new_interpreter.get_output_details()

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

        # Determine expected frames and feature handling from stats or model
        expected_time = int(norm_stats["EXPECTED_FRAMES"]) if norm_stats else int(new_input_details[0]["shape"][1])
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

        # Run both models and combine predictions
        
        # Prepare tensor for old model
        old_model_input_shape = old_input_details[0]["shape"]
        old_tensor = hand_tensor.copy()
        if tuple(old_model_input_shape) != old_tensor.shape:
            try:
                old_tensor = old_tensor.astype(np.float32)
                mt = int(old_model_input_shape[1])
                if mt != expected_time:
                    seq2 = pad_or_truncate_time(hand_flat, mt)
                    old_tensor = seq2.reshape(1, mt, total_nodes, joint_feats).astype(np.float32)
            except Exception:
                pass
        
        # Prepare tensor for new model
        new_model_input_shape = new_input_details[0]["shape"]
        new_tensor = hand_tensor.copy()
        if tuple(new_model_input_shape) != new_tensor.shape:
            try:
                new_tensor = new_tensor.astype(np.float32)
                mt = int(new_model_input_shape[1])
                if mt != expected_time:
                    seq2 = pad_or_truncate_time(hand_flat, mt)
                    new_tensor = seq2.reshape(1, mt, total_nodes, joint_feats).astype(np.float32)
            except Exception:
                pass
        
        # Run old model
        old_interpreter.set_tensor(old_input_details[0]['index'], old_tensor)
        old_interpreter.invoke()
        old_probs = old_interpreter.get_tensor(old_output_details[0]['index'])[0]
        
        # Run new model
        new_interpreter.set_tensor(new_input_details[0]['index'], new_tensor)
        new_interpreter.invoke()
        new_probs = new_interpreter.get_tensor(new_output_details[0]['index'])[0]
        
        # Combine predictions: 65% new model, 35% old model
        # Handle different number of classes if needed
        max_classes = max(len(old_probs), len(new_probs))
        old_probs_padded = np.zeros(max_classes, dtype=np.float32)
        new_probs_padded = np.zeros(max_classes, dtype=np.float32)
        old_probs_padded[:len(old_probs)] = old_probs
        new_probs_padded[:len(new_probs)] = new_probs
        
        # Weighted average
        combined_probs = (NEW_MODEL_WEIGHT * new_probs_padded) + (OLD_MODEL_WEIGHT * old_probs_padded)
        
        # Normalize to ensure probabilities sum to 1
        combined_probs = combined_probs / combined_probs.sum()
        
        # Get top 3 predictions from combined
        top3_idx = np.argsort(-combined_probs)[:3]
        top3_predictions = [
            {
                "label": label_map.get(int(i), str(int(i))), 
                "prob": float(combined_probs[i]),
                "old_prob": float(old_probs_padded[i]),
                "new_prob": float(new_probs_padded[i])
            } for i in top3_idx
        ]
        
        # Get final prediction from combined
        final_idx = int(np.argmax(combined_probs))
        final_conf = float(combined_probs[final_idx])

        return jsonify({
            "prediction": label_map.get(final_idx, "Unknown"),
            "confidence": round(final_conf, 4),
            "top3": top3_predictions
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
                "old_model_input_shape": list(old_input_details[0]["shape"]) if old_input_details else None,
                "new_model_input_shape": list(new_input_details[0]["shape"]) if new_input_details else None,
            })
        except Exception:
            pass
        details["trace"] = traceback.format_exc()
        return jsonify(details), 500

@app.route("/", methods=["GET"])
def home():
    return "🧠 Dual TFLite Model Server Running (65% New + 35% Old)"

@app.route("/health", methods=["GET"])
def health():
    try:
        ensure_interpreters()
        info = {
            "status": "ok",
            "ensemble_mode": "weighted_average",
            "old_model": {
                "loaded": old_interpreter is not None,
                "path": OLD_MODEL_PATH,
                "weight": OLD_MODEL_WEIGHT,
                "input_shape": old_input_details[0]["shape"].tolist() if old_input_details else None,
            },
            "new_model": {
                "loaded": new_interpreter is not None,
                "path": NEW_MODEL_PATH,
                "weight": NEW_MODEL_WEIGHT,
                "input_shape": new_input_details[0]["shape"].tolist() if new_input_details else None,
            },
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)


