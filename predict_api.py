"""
Minimal Flask API for oral cancer prediction. Used by the Next.js frontend.
Run: python predict_api.py  (default port 5001)
"""
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "oral_cancer_model_clean.h5")

_model = None


def load_model():
    global _model
    if _model is not None:
        return _model
    head = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 1280)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name="head")
    head.load_weights(MODEL_PATH, by_name=True)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    base.trainable = False
    _model = tf.keras.Sequential([base, head], name="oral_cancer_model")
    return _model


def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files and not request.files.get("file"):
        return jsonify({"error": "No image file provided"}), 400
    file = request.files.get("file")
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        image = Image.open(file).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400
    model = load_model()
    processed = preprocess_image(image)
    prob_value = float(model.predict(processed, verbose=0)[0][0])
    is_cancer = prob_value < 0.5
    confidence = abs(prob_value - 0.5) * 2 * 100
    return jsonify({
        "detected_class": "Cancer" if is_cancer else "Normal",
        "confidence": round(confidence, 1),
        "is_cancer": is_cancer,
        "prob_value": prob_value,
        "prob_cancer_pct": round((1 - prob_value) * 100, 1),
        "prob_normal_pct": round(prob_value * 100, 1),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
