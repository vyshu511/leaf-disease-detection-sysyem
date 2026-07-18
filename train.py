"""
Plant Leaf Disease Detection — Training Script (v2)
----------------------------------------------------
Improvements over the original version:
  - Data augmentation (rotation, zoom, horizontal flip, brightness)
  - EarlyStopping + ModelCheckpoint + ReduceLROnPlateau
  - Two-phase training: frozen-base warmup, then fine-tuning the top
    layers of MobileNetV2 at a low learning rate
  - Confusion matrix + classification report saved to static/plots/
  - Training accuracy/loss curves saved to static/plots/ and history.json
    (consumed by the dashboard's data-visualization charts)

Run:
    python train.py
"""

import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("model", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
np.save("model/class_names.npy", class_names)
print(f"Classes ({len(class_names)}): {class_names}")

# Keep an untouched copy of the validation set (as arrays) for the
# confusion matrix / classification report at the end, before prefetch
# batching makes iteration awkward.
val_ds_for_eval = val_ds

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# ---------------------------------------------------------------------------
# Data augmentation — applied only during training, only to images
# ---------------------------------------------------------------------------
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),          # ~54 degrees max
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomBrightness(0.15),
        tf.keras.layers.RandomContrast(0.1),
    ],
    name="data_augmentation",
)

# ---------------------------------------------------------------------------
# Model — MobileNetV2 transfer learning
# ---------------------------------------------------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # Phase 1: frozen base

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.layers.Rescaling(1.0 / 255)(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
callbacks_phase1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "model/leaf_model.keras", monitor="val_accuracy", save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    ),
]

# ---------------------------------------------------------------------------
# Phase 1: train the classifier head with the base frozen
# ---------------------------------------------------------------------------
EPOCHS_PHASE1 = 15

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks_phase1,
)

# ---------------------------------------------------------------------------
# Phase 2: fine-tune the top layers of MobileNetV2
# ---------------------------------------------------------------------------
# Small dataset (808 images) → only unfreeze the last ~30 layers to avoid
# overfitting, and use a much lower learning rate so we don't destroy the
# pretrained ImageNet features.
base_model.trainable = True
FINE_TUNE_AT = len(base_model.layers) - 30
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "model/leaf_model.keras", monitor="val_accuracy", save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7
    ),
]

EPOCHS_PHASE2 = 10
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks_phase2,
)

# ---------------------------------------------------------------------------
# Merge histories and save for the dashboard's training charts
# ---------------------------------------------------------------------------
full_history = {
    "accuracy": history1.history["accuracy"] + history2.history["accuracy"],
    "val_accuracy": history1.history["val_accuracy"] + history2.history["val_accuracy"],
    "loss": history1.history["loss"] + history2.history["loss"],
    "val_loss": history1.history["val_loss"] + history2.history["val_loss"],
    "fine_tune_start_epoch": len(history1.history["accuracy"]),
    "class_distribution": {},
}

# Class distribution (image count per class) for the dashboard bar chart
for cname in class_names:
    class_dir = os.path.join("dataset/train", cname)
    if os.path.isdir(class_dir):
        full_history["class_distribution"][cname] = len(os.listdir(class_dir))

with open("static/plots/history.json", "w") as f:
    json.dump(full_history, f, indent=2)

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------
loss, accuracy = model.evaluate(val_ds)
print(f"\nFinal Validation Accuracy: {accuracy * 100:.2f}%")

if accuracy < 0.90:
    print(
        "WARNING: validation accuracy fell below the 90% requirement. "
        "Consider: more epochs, more data, less aggressive augmentation, "
        "or unfreezing fewer/more base layers."
    )

# ---------------------------------------------------------------------------
# Confusion matrix + classification report
# ---------------------------------------------------------------------------
y_true = []
y_pred = []
for images, labels in val_ds_for_eval:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
report = classification_report(
    y_true, y_pred, target_names=class_names, output_dict=True
)

with open("static/plots/classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("static/plots/confusion_matrix.png", dpi=150)
plt.close()

# Accuracy / loss curves (also saved as static PNGs; history.json powers
# the interactive dashboard chart)
epochs_range = range(1, len(full_history["accuracy"]) + 1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, full_history["accuracy"], label="Train")
plt.plot(epochs_range, full_history["val_accuracy"], label="Validation")
plt.axvline(full_history["fine_tune_start_epoch"], color="gray", linestyle="--", label="Fine-tune start")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, full_history["loss"], label="Train")
plt.plot(epochs_range, full_history["val_loss"], label="Validation")
plt.axvline(full_history["fine_tune_start_epoch"], color="gray", linestyle="--", label="Fine-tune start")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("static/plots/training_curves.png", dpi=150)
plt.close()

model.save("model/leaf_model.keras")
print("Model, plots, and history saved successfully!")