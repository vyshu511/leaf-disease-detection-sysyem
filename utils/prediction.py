"""
Prediction utilities: top-3 class predictions, confidence-level labeling,
and an "unknown image" guard for out-of-distribution inputs.
"""

import numpy as np
import tensorflow as tf

# Below this confidence, we tell the user we aren't sure it's even a
# recognizable leaf / one of our trained classes.
UNKNOWN_THRESHOLD = 40.0  # percent

HIGH_CONFIDENCE = 85.0
MEDIUM_CONFIDENCE = 60.0


def confidence_level(confidence_pct: float) -> str:
    if confidence_pct >= HIGH_CONFIDENCE:
        return "High"
    if confidence_pct >= MEDIUM_CONFIDENCE:
        return "Medium"
    return "Low"


def preprocess_image(filepath: str, target_size=(224, 224)):
    img = tf.keras.utils.load_img(filepath, target_size=target_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_top_k(model, img_array, class_names, k=3):
    """
    Returns:
        top_predictions: list of {"class_name", "confidence"} sorted desc.
        is_unknown: bool, True if the top confidence is below UNKNOWN_THRESHOLD
    """
    preds = model.predict(img_array, verbose=0)[0]  # shape (num_classes,)
    top_idx = np.argsort(preds)[::-1][:k]

    top_predictions = [
        {
            "class_name": str(class_names[i]),
            "confidence": round(float(preds[i]) * 100, 2),
        }
        for i in top_idx
    ]

    top_confidence = top_predictions[0]["confidence"]
    is_unknown = top_confidence < UNKNOWN_THRESHOLD

    return top_predictions, is_unknown
