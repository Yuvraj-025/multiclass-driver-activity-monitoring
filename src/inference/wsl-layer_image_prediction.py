import os
# Suppress TensorFlow and Python warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import mixed_precision

# -------- CONFIG --------
# Derive BASE_DIR from script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model_finetuned.keras")   # change to final_model.keras if needed
IMG_PATH = os.path.join(BASE_DIR, "media", "test2.jpg")  # <-- path to the image you want to predict
JSON_DIR = os.path.join(BASE_DIR, "json")

os.makedirs(JSON_DIR, exist_ok=True)

# -------- Disable mixed precision for CPU inference --------
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

# Save mapping (optional, as you requested)
with open(os.path.join(JSON_DIR, "class_name_map.json"), "w") as f:
    json.dump(class_name, f, indent=4, sort_keys=True)

# -------- Load model --------
model = load_model(MODEL_PATH, compile=False)
#model.summary()

# -------- Prediction function --------
def predict_image(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # VGG16 preprocessing (must match training)
    x = preprocess_input(x)

    # Predict
    preds = model.predict(x)[0]

    # Get top prediction
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])

    # Map index -> c0..c9
    class_key = f"c{class_idx}"
    readable_label = class_name[class_key]

    return class_key, readable_label, confidence, preds

# -------- Run prediction --------
if __name__ == "__main__":
    class_key, readable_label, confidence, probs = predict_image(IMG_PATH)

    print("\nPrediction Result")
    print("-----------------")
    print(f"Raw class     : {class_key}")
    print(f"Readable class: {readable_label}")
    print(f"Confidence    : {confidence:.4f}")

    print("\nAll class probabilities:")
    for i, p in enumerate(probs):
        ck = f"c{i}"
        print(f"{ck:>3} ({class_name[ck]:<22}): {p:.4f}")
