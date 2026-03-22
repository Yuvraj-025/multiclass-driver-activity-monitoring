import os
# Suppress TensorFlow and Python warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import cv2
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import mixed_precision

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # disables buggy MSMF backend on Windows

# -------- CONFIG --------
# Derive BASE_DIR dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model_finetuned_v1.h5")  # change to final_model.h5 if needed
USE_WEBCAM = False              # True = webcam, False = video file
VIDEO_PATH = os.path.join(BASE_DIR, "media", "input_video.mp4") # used if USE_WEBCAM = False
CAMERA_INDEX = 0              # default webcam

# Disable mixed precision for CPU inference (safe default)
mixed_precision.set_global_policy("float32")

# -------- Human-readable class names --------
class_name = {
    "c0": "SAFE_DRIVING",
    "c1": "TEXTING_RIGHT",
    "c2": "TALKING_PHONE_RIGHT",
    "c3": "TEXTING_LEFT",
    "c4": "TALKING_PHONE_LEFT",
    "c5": "OPERATING_RADIO",
    "c6": "DRINKING",
    "c7": "REACHING_BEHIND",
    "c8": "HAIR_AND_MAKEUP",
    "c9": "TALKING_TO_PASSENGER"
}

# -------- Load model --------
print("[INFO] Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("[INFO] Model loaded.")

# -------- Video source --------
if USE_WEBCAM:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video source")

# -------- Utility: preprocess frame --------
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(frame_rgb.astype(np.float32), axis=0)
    x = preprocess_input(x)
    return x

# -------- Real-time loop --------
prev_time = time.time()
fps = 0

print("[INFO] Starting real-time inference. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    x = preprocess_frame(frame)

    # Predict
    preds = model.predict(x, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])

    class_key = f"c{class_idx}"
    label = class_name[class_key]

    # FPS calculation
    curr_time = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
    prev_time = curr_time

    # Overlay text
    text = f"{label} ({confidence*100:.1f}%)"
    fps_text = f"FPS: {fps:.1f}"

    cv2.rectangle(frame, (10, 10), (520, 90), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, fps_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Driver Activity Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
