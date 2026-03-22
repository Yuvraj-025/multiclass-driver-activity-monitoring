import os
# Suppress TensorFlow and Python warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

# ---- Paths ----
import os, math, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

# Base directory relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Data is now inside the data folder
TRAIN_DIR = os.path.join(BASE_DIR, "data", "imgs", "train")
CSV_PATH = os.path.join(BASE_DIR, "data", "driver_imgs_list.csv")
IMG_SIZE = (224, 224)
BATCH_SIZE = 8

data_train = pd.read_csv(CSV_PATH)
data_train['img_path'] = data_train.apply(
    lambda row: os.path.join(TRAIN_DIR, row['classname'], row['img']), axis=1
)
data_train = data_train[data_train['img_path'].apply(os.path.exists)].reset_index(drop=True)

xtrain, xtest = np.split(data_train.sample(frac=1, random_state=42), [int(len(data_train)*0.8)])

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=xtrain,
    x_col="img_path",
    y_col="classname",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=xtest,
    x_col="img_path",
    y_col="classname",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model_v5.h5")

# ---- Load trained model (Phase 1 result) ----
model = load_model(MODEL_PATH, compile=False)
model.summary()

# ---- Freeze everything first ----
for layer in model.layers:
    layer.trainable = False

# ---- Unfreeze top VGG16 blocks (block4 + block5) ----
for layer in model.layers:
    if layer.name.startswith("block4_") or layer.name.startswith("block5_"):
        layer.trainable = True

# ---- Recompile with low LR for fine-tuning ----
model.compile(
    loss="categorical_crossentropy",
    optimizer=SGD(learning_rate=1e-5, momentum=0.9),
    metrics=["accuracy"]
)

# ---- Fine-tuning callbacks ----
fine_tune_checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_model_finetuned_v1.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

fine_tune_early_stop = EarlyStopping(
    monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1
)

fine_tune_reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
)

# ---- Fine-tune ----
fine_tune_history = model.fit(
    train_generator,          # MUST be defined (from your original script)
    validation_data=valid_generator,
    epochs=20,
    callbacks=[fine_tune_checkpoint, fine_tune_reduce_lr, fine_tune_early_stop],
    workers=4,
    use_multiprocessing=True,
    verbose=1
)

# ---- Save final fine-tuned model ----
final_finetuned_path = os.path.join(MODEL_DIR, "final_model_finetuned_v1.h5")
model.save(final_finetuned_path)
