"""
Grad-CAM utility for the MobileNetV2-based leaf disease model.

Generates a heatmap over the input leaf image showing which regions
most influenced the model's prediction, then overlays it on the
original image and saves it to static/gradcam/.
"""

import numpy as np
import tensorflow as tf
import cv2
import os


def _find_last_conv_layer(model: tf.keras.Model) -> str:
    """Walk the model (including nested Sequential/Functional sub-models,
    e.g. the MobileNetV2 base) to find the last Conv2D layer's name."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                return _find_last_conv_layer(layer)
            except ValueError:
                continue
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def _get_submodel_containing(model, layer_name):
    """Return the (possibly nested) sub-model that directly owns layer_name."""
    for layer in model.layers:
        if layer.name == layer_name:
            return model
        if isinstance(layer, tf.keras.Model):
            found = _get_submodel_containing(layer, layer_name)
            if found is not None:
                return found
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    img_array: preprocessed batch of shape (1, H, W, 3), values in [0, 255]
               (Rescaling layer inside the model handles normalization)
    model:     the full trained tf.keras model
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer(model)

    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
    if base_model is None:
        raise ValueError("Could not locate the MobileNetV2 base sub-model.")

    # Build a model that maps the base model's input to:
    #  - the activations of the last conv layer
    #  - the base model's final output
    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output],
    )

    # Recreate the "head" (GAP -> Dropout -> Dense) on top of the base output
    head_layers = [l for l in model.layers if l is not base_model
                    and not isinstance(l, tf.keras.layers.Rescaling)]

    rescale_layer = None
    for l in model.layers:
        if isinstance(l, tf.keras.layers.Rescaling):
            rescale_layer = l
            break

    x = rescale_layer(img_array) if rescale_layer is not None else img_array

    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(x)
        tape.watch(conv_output)
        y = base_output
        for layer in head_layers:
            y = layer(y)
        pred_index = tf.argmax(y[0])
        class_channel = y[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index.numpy())


def save_gradcam_overlay(original_img_path, heatmap, output_path, alpha=0.4):
    """Overlay a Grad-CAM heatmap on the original image and save to disk."""
    img = cv2.imread(original_img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {original_img_path}")

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)
    return output_path
