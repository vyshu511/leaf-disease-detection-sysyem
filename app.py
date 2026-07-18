"""
Plant Leaf Disease Detection — Flask Backend (v2)

New in this version:
  - JSON API endpoint (/api/predict) in addition to the HTML form flow
  - Top-3 predictions with confidence + High/Medium/Low reliability label
  - Healthy-leaf and unknown-image (out-of-distribution) detection
  - Grad-CAM heatmap overlay generation
  - Structured disease info (symptoms/causes/prevention/remedies) per prediction
  - Image validation (type + size) and centralized error handling
  - Prediction history persisted to a local JSON file
  - Environment-based config, ready for gunicorn on Render
"""

import json
import os
import time
import uuid
from datetime import datetime

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from utils.disease_info import get_disease_info, to_dict as disease_info_to_dict
from utils.prediction import confidence_level, predict_top_k, preprocess_image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")
HISTORY_FILE = os.path.join(BASE_DIR, "static", "history", "predictions.json")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "model", "leaf_model.keras"))
CLASS_NAMES_PATH = os.environ.get(
    "CLASS_NAMES_PATH", os.path.join(BASE_DIR, "model", "class_names.npy")
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB
MAX_HISTORY_ITEMS = 50

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------------------------------------------------------------------
# Load model + classes once at startup
# ---------------------------------------------------------------------------
_model = None
_class_names = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train.py first."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def get_class_names():
    global _class_names
    if _class_names is None:
        if not os.path.exists(CLASS_NAMES_PATH):
            raise FileNotFoundError(
                f"class_names.npy not found at {CLASS_NAMES_PATH}. Run train.py first."
            )
        _class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
    return _class_names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_history(history: list) -> None:
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-MAX_HISTORY_ITEMS:], f, indent=2)


def append_history(entry: dict) -> None:
    history = load_history()
    history.append(entry)
    save_history(history)


class APIError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@app.errorhandler(APIError)
def handle_api_error(err: APIError):
    return jsonify({"success": False, "error": err.message}), err.status_code


@app.errorhandler(413)
def handle_too_large(err):
    return (
        jsonify({"success": False, "error": "File too large. Max size is 8 MB."}),
        413,
    )


@app.errorhandler(500)
def handle_server_error(err):
    return jsonify({"success": False, "error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Core prediction pipeline (shared by HTML route + JSON API)
# ---------------------------------------------------------------------------
def run_prediction(filepath: str, run_gradcam: bool = True) -> dict:
    model = get_model()
    class_names = get_class_names()

    img_array = preprocess_image(filepath)
    top_predictions, is_unknown = predict_top_k(model, img_array, class_names, k=3)

    top_class = top_predictions[0]["class_name"]
    top_confidence = top_predictions[0]["confidence"]
    reliability = confidence_level(top_confidence)

    info = get_disease_info(top_class)

    gradcam_url = None
    if run_gradcam and not is_unknown:
        try:
            from utils.gradcam import make_gradcam_heatmap, save_gradcam_overlay

            heatmap, _ = make_gradcam_heatmap(img_array, model)
            gradcam_filename = f"gradcam_{uuid.uuid4().hex[:10]}.jpg"
            gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)
            save_gradcam_overlay(filepath, heatmap, gradcam_path)
            gradcam_url = f"/static/gradcam/{gradcam_filename}"
        except Exception as e:
            # Grad-CAM is a bonus visualization — never let it break a prediction
            app.logger.warning(f"Grad-CAM generation failed: {e}")
            gradcam_url = None

    result = {
        "success": True,
        "is_unknown": bool(is_unknown),
        "is_healthy": bool(info.is_healthy) if not is_unknown else False,
        "top_prediction": top_class,
        "confidence": top_confidence,
        "reliability": reliability,
        "top_3": top_predictions,
        "gradcam_url": gradcam_url,
        "disease_info": disease_info_to_dict(info) if not is_unknown else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/history")
def history_page():
    return jsonify(load_history())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        raise APIError("No image file provided under the 'image' field.")

    file = request.files["image"]

    if file.filename == "":
        raise APIError("No file selected.")

    if not allowed_file(file.filename):
        raise APIError(
            f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    filename = secure_filename(f"{uuid.uuid4().hex[:10]}_{file.filename}")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(filepath)
    except Exception:
        raise APIError("Failed to save uploaded file.", status_code=500)

    # Basic image integrity check
    try:
        from PIL import Image

        with Image.open(filepath) as im:
            im.verify()
    except Exception:
        os.remove(filepath)
        raise APIError("Uploaded file is not a valid image.")

    try:
        result = run_prediction(filepath)
    except FileNotFoundError as e:
        raise APIError(str(e), status_code=503)
    except Exception as e:
        app.logger.exception("Prediction failed")
        raise APIError(f"Prediction failed: {e}", status_code=500)

    result["image_url"] = f"/static/uploads/{filename}"

    append_history(
        {
            "id": uuid.uuid4().hex[:10],
            "image_url": result["image_url"],
            "prediction": result["top_prediction"],
            "confidence": result["confidence"],
            "reliability": result["reliability"],
            "is_healthy": result["is_healthy"],
            "is_unknown": result["is_unknown"],
            "timestamp": result["timestamp"],
        }
    )

    return jsonify(result)


@app.route("/predict", methods=["POST"])
def predict_form():
    """Legacy/simple HTML form submission path (progressive enhancement
    fallback for when JavaScript is disabled)."""
    try:
        if "image" not in request.files or request.files["image"].filename == "":
            return render_template("index.html", error="No image selected.")

        file = request.files["image"]
        if not allowed_file(file.filename):
            return render_template("index.html", error="Unsupported file type.")

        filename = secure_filename(f"{uuid.uuid4().hex[:10]}_{file.filename}")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        result = run_prediction(filepath)
        result["image_url"] = f"/static/uploads/{filename}"

        return render_template("index.html", result=result)
    except Exception as e:
        app.logger.exception("Prediction failed")
        return render_template("index.html", error=f"Prediction failed: {e}")


@app.route("/api/model-info")
def model_info():
    plots_dir = os.path.join(BASE_DIR, "static", "plots")
    history_path = os.path.join(plots_dir, "history.json")
    report_path = os.path.join(plots_dir, "classification_report.json")

    data = {"history": None, "classification_report": None, "class_names": []}

    if os.path.exists(history_path):
        with open(history_path) as f:
            data["history"] = json.load(f)

    if os.path.exists(report_path):
        with open(report_path) as f:
            data["classification_report"] = json.load(f)

    try:
        data["class_names"] = [str(c) for c in get_class_names()]
    except FileNotFoundError:
        pass

    return jsonify(data)


@app.route("/healthz")
def health_check():
    """Simple liveness endpoint for Render / uptime monitors."""
    return jsonify({"status": "ok", "time": time.time()})


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug)