import os
# Suppress TensorFlow and Python warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True 

# Derive BASE_DIR from the current file's location to support execution from anywhere
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PICKLE_DIR = os.path.join(BASE_DIR, "pickle")
JSON_DIR = os.path.join(BASE_DIR, "json_files")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model_v5.h5")

os.makedirs(JSON_DIR, exist_ok=True)

# Load model
model = load_model(MODEL_PATH, compile=False)

# Load label mapping
with open(os.path.join(PICKLE_DIR, "labels_id.pkl"), "rb") as handle:
    labels_id = pickle.load(handle)

id_to_label = {v: k for k, v in labels_id.items()}

# Human-readable mapping
class_name_map = {
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

# Save JSON once (optional)
with open(os.path.join(JSON_DIR, "class_name_map.json"), "w") as f:
    json.dump(class_name_map, f, indent=4, sort_keys=True)

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # IMPORTANT for VGG16
    return x

def predict_result(img_path):
    image_tensor = path_to_tensor(img_path)
    ypred = model.predict(image_tensor, verbose=0)
    class_idx = int(np.argmax(ypred, axis=1)[0])

    class_key = id_to_label[class_idx]
    human_label = class_name_map[class_key]

    print("Predicted class ID:", class_idx)
    print("Predicted class key:", class_key)
    print("Human-readable label:", human_label)

    return human_label

# Test prediction on a sample image located in the media directory
result = predict_result(os.path.join(BASE_DIR, "media", "test3.jpg"))
print("Final Prediction:", result)
