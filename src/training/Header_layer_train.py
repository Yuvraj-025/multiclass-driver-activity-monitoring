import os
# Suppress TensorFlow and Python warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf
from tensorflow.keras import mixed_precision

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

mixed_precision.set_global_policy('mixed_float16')
print("GPUs:", gpus)


# ========== CONFIG ==========
# Derive BASE_DIR from the current script location to allow execution from anywhere
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "imgs", "train")
MODEL_PATH = os.path.join(BASE_DIR, "model")
PICKLE_PATH = os.path.join(BASE_DIR, "pickle")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PICKLE_PATH, exist_ok=True)

CSV_PATH = os.path.join(BASE_DIR, "data", "driver_imgs_list.csv")

BATCH_SIZE = 8                
IMG_SIZE = (224, 224)
EPOCHS = 50
TARGET_ACC = 0.80             

# CPU/GPU tuning
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")

# ========== DATA ==========
data_train = pd.read_csv(CSV_PATH)

# labels mapping
labels_list = sorted(data_train['classname'].unique())
labels_id = {label: idx for idx, label in enumerate(labels_list)}
with open(os.path.join(PICKLE_PATH, "labels_id.pkl"), "wb") as f:
    pickle.dump(labels_id, f)

data_train['class_id'] = data_train['classname'].map(labels_id)
data_train['img_path'] = data_train.apply(
    lambda row: os.path.join(TRAIN_DIR, row['classname'], row['img']),
    axis=1
)

data_train = data_train[data_train['img_path'].apply(os.path.exists)].reset_index(drop=True)

xtrain, xtest = np.split(data_train.sample(frac=1, random_state=42), [int(len(data_train)*0.8)])

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

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

train_steps = math.ceil(train_generator.n / BATCH_SIZE)
val_steps = math.ceil(valid_generator.n / BATCH_SIZE)

# ========== MODEL ==========
base_model = VGG16(include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
outputs = Dense(len(labels_list), activation="softmax", dtype="float32")(x)

model = Model(inputs=base_model.input, outputs=outputs)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    loss="categorical_crossentropy",
    optimizer=SGD(learning_rate=1e-4, momentum=0.9),
    metrics=["accuracy"]
)

model.summary()

checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_PATH, "best_model_v5.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)

class StopAtAccuracy(Callback):
    def __init__(self, target=TARGET_ACC):
        super().__init__()
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy")
        if val_acc is not None:
            if val_acc >= self.target:
                print(f"\nReached target val_accuracy = {val_acc:.4f} >= {self.target}. Stopping training.")
                self.model.stop_training = True

stop_at_acc = StopAtAccuracy(target=TARGET_ACC)

callbacks = [checkpoint, reduce_lr, early_stop, stop_at_acc]

# ========== TRAIN ==========
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=valid_generator,
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    workers=4,               
    use_multiprocessing=True,
    verbose=1
)

final_model_path = os.path.join(MODEL_PATH, "final_model.keras")
model.save(final_model_path)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history.get('loss', []), label='train')
plt.plot(history.history.get('val_loss', []), label='val')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history.get('accuracy', []), label='train')
plt.plot(history.history.get('val_accuracy', []), label='val')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
