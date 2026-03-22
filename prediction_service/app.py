"""
=================================================================
Driver Activity Detection — Prediction Microservice
=================================================================
A self-contained Flask web application that serves a trained VGG16
model for real-time driver distraction classification.

Supports two input modes:
  1. Live Webcam  — streams the default camera feed
  2. Video Upload — accepts an uploaded video file

Both modes overlay prediction labels and confidence on each frame
and stream the annotated video back to the browser via MJPEG.

Run:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
=================================================================
"""

import os

# ---- Suppress TensorFlow / oneDNN warnings ----
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # Disable buggy MSMF backend on Windows

import warnings
warnings.filterwarnings("ignore")

import cv2
import time
import threading
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import mixed_precision

# ================================================================
# CONFIG
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model_finetuned_v1.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Use float32 for safe CPU/GPU inference
mixed_precision.set_global_policy("float32")

# ================================================================
# HUMAN-READABLE CLASS LABELS
# ================================================================
CLASS_NAMES = {
    "c0": "SAFE DRIVING",
    "c1": "TEXTING (RIGHT)",
    "c2": "PHONE CALL (RIGHT)",
    "c3": "TEXTING (LEFT)",
    "c4": "PHONE CALL (LEFT)",
    "c5": "OPERATING RADIO",
    "c6": "DRINKING",
    "c7": "REACHING BEHIND",
    "c8": "HAIR & MAKEUP",
    "c9": "TALKING TO PASSENGER",
}

# Color coding: green for safe, red-ish for dangerous
CLASS_COLORS = {
    "c0": (0, 220, 100),    # Green — safe
    "c1": (0, 80, 255),     # Red-orange
    "c2": (0, 80, 255),
    "c3": (0, 80, 255),
    "c4": (0, 80, 255),
    "c5": (0, 180, 255),    # Orange
    "c6": (0, 180, 255),
    "c7": (0, 140, 255),
    "c8": (0, 140, 255),
    "c9": (0, 220, 255),    # Yellow-ish
}

# ================================================================
# LOAD MODEL (once at startup)
# ================================================================
print("[INFO] Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("[INFO] Model loaded successfully.")

# ================================================================
# FLASK APP
# ================================================================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload

# Global state for managing video sources
current_upload_path = None
stream_active = {"webcam": False, "upload": False}


def allowed_file(filename):
    """Check if the uploaded file has an allowed video extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_frame(frame):
    """
    Resize and preprocess a single BGR frame for VGG16 inference.
    Returns a numpy array of shape (1, 224, 224, 3).
    """
    resized = cv2.resize(frame, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(rgb.astype(np.float32), axis=0)
    x = preprocess_input(x)
    return x


def annotate_frame(frame, label, confidence, class_key, fps):
    """
    Draw a styled overlay on the frame with the prediction result.
    """
    h, w = frame.shape[:2]
    color = CLASS_COLORS.get(class_key, (0, 255, 0))

    # Semi-transparent banner at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Prediction text
    text = f"{label}"
    conf_text = f"{confidence * 100:.1f}%"
    fps_text = f"FPS: {fps:.1f}"

    cv2.putText(frame, text, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(frame, conf_text, (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (w - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    # Colored indicator bar at the very top
    cv2.rectangle(frame, (0, 0), (w, 5), color, -1)

    return frame


def generate_frames(source="webcam"):
    """
    Generator that yields MJPEG frames from either webcam or uploaded video.
    Each frame is annotated with the model's prediction.
    """
    global current_upload_path

    if source == "webcam":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        if current_upload_path is None or not os.path.exists(current_upload_path):
            return
        cap = cv2.VideoCapture(current_upload_path)

    if not cap.isOpened():
        return

    stream_active[source] = True
    prev_time = time.time()
    fps = 0.0

    try:
        while stream_active[source]:
            ret, frame = cap.read()
            if not ret:
                if source == "upload":
                    # Loop uploaded video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            # Run inference
            x = preprocess_frame(frame)
            preds = model.predict(x, verbose=0)[0]
            class_idx = int(np.argmax(preds))
            confidence = float(preds[class_idx])
            class_key = f"c{class_idx}"
            label = CLASS_NAMES[class_key]

            # Calculate FPS with exponential smoothing
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
            prev_time = curr_time

            # Draw overlay
            frame = annotate_frame(frame, label, confidence, class_key, fps)

            # Encode to JPEG and yield as MJPEG chunk
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        cap.release()
        stream_active[source] = False


# ================================================================
# ROUTES
# ================================================================

@app.route("/")
def index():
    """Landing page with webcam / upload options."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG streaming endpoint. Query param: ?source=webcam or ?source=upload"""
    source = request.args.get("source", "webcam")
    return Response(
        generate_frames(source),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/upload", methods=["POST"])
def upload_video():
    """Handle video file upload and redirect to the upload stream view."""
    global current_upload_path

    if "video" not in request.files:
        return redirect(url_for("index"))

    file = request.files["video"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    current_upload_path = filepath

    return redirect(url_for("index", mode="upload"))


@app.route("/stop")
def stop_stream():
    """Stop any active stream. Returns JSON so the client JS stays in control."""
    source = request.args.get("source", "webcam")
    stream_active[source] = False
    return jsonify({"status": "stopped", "source": source})


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("[INFO] Starting prediction service on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
