"""
Flask REST API for FloraAI Flower Classification.
Exposes:
  - GET  /api/health
  - GET  /api/model_info
  - POST /api/predict
"""
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from predictor import FlowerPredictor, ALLOWED_EXTENSIONS

app = Flask(__name__)

# Configure CORS: allow frontend origins via CORS_ORIGINS env var, defaults to *
allowed_origins = os.environ.get("CORS_ORIGINS", "*")
CORS(app, resources={r"/api/*": {"origins": allowed_origins.split(",") if allowed_origins != "*" else "*"}})

# Initialize predictor singleton on startup
predictor = FlowerPredictor()


def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "FloraAI Flower Recognition API",
        "model_loaded": predictor.is_loaded,
        "model_name": predictor.config.get("model_name", "ResNet-18"),
        "checkpoint_source": predictor.checkpoint_source,
        "error": predictor.load_error
    }), 200 if predictor.is_loaded else 503


@app.route("/api/model_info", methods=["GET"])
def model_info():
    config = predictor.config
    return jsonify({
        "model_architecture": config.get("model_name", "ResNet-18"),
        "task": config.get("task", "Five-Class Flower Classification"),
        "supported_classes": predictor.classes,
        "input_dimensions": config.get("input_size", [3, 224, 224]),
        "image_preprocessing": {
            "resize": 256,
            "center_crop": 224,
            "mean": config.get("image_mean", [0.485, 0.456, 0.406]),
            "std": config.get("image_std", [0.229, 0.224, 0.225])
        },
        "evaluation_metrics": config.get("evaluation_metrics", {}),
        "comparison_metrics": config.get("comparison_metrics", {})
    }), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    if not predictor.is_loaded:
        return jsonify({
            "error": "Model is not loaded on backend server.",
            "details": predictor.load_error or "Checkpoint missing or failed to initialize."
        }), 503

    if "image" not in request.files and "file" not in request.files:
        return jsonify({"error": "No image file provided. Please send a file under 'image' or 'file' field."}), 400

    file_obj = request.files.get("image") or request.files.get("file")

    if not file_obj or file_obj.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file_obj.filename):
        return jsonify({
            "error": f"Unsupported file type. Supported extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    try:
        image_bytes = file_obj.read()
        if len(image_bytes) == 0:
            return jsonify({"error": "Uploaded image file is empty."}), 400

        result = predictor.predict_image_bytes(image_bytes)
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Avoid exposing raw server tracebacks to clients
        return jsonify({"error": "An error occurred while processing the image."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=False)
