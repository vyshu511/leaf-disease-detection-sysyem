"""
Grad-CAM utility for MobileNetV2 leaf disease detection model.
Generates heatmap and overlays it on original leaf image.
"""

import numpy as np
import tensorflow as tf
import cv2
import os


def _find_last_conv_layer(model):
    """
    Find last convolution layer inside nested MobileNetV2 model
    """

    for layer in reversed(model.layers):

        # MobileNetV2 is a nested model
        if isinstance(layer, tf.keras.Model):
            try:
                return _find_last_conv_layer(layer)
            except:
                continue

        # Conv2D layer
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    # MobileNetV2 fallback layer
    return "out_relu"



def make_gradcam_heatmap(img_array, model):
    """
    Create Grad-CAM heatmap

    img_array:
        shape (1,224,224,3)

    model:
        trained keras model
    """


    # Find MobileNetV2 base model
    base_model = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break


    if base_model is None:
        raise Exception("MobileNetV2 base model not found")


    # Get last convolution layer
    last_conv_layer_name = _find_last_conv_layer(base_model)


    print("GradCAM Layer:", last_conv_layer_name)



    # Grad-CAM model

    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            base_model.output
        ]
    )


    # Remove rescaling layer
    x = img_array

    if isinstance(model.layers[0], tf.keras.layers.Rescaling):
        x = model.layers[0](x)



    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(x)

        # pass through classifier layers

        y = predictions

        for layer in model.layers[2:]:
            y = layer(y)


        predicted_class = tf.argmax(y[0])

        loss = y[:, predicted_class]



    grads = tape.gradient(
        loss,
        conv_outputs
    )


    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )


    conv_outputs = conv_outputs[0]


    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]


    heatmap = tf.squeeze(heatmap)


    heatmap = tf.maximum(
        heatmap,
        0
    )


    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )


    return heatmap.numpy(), int(predicted_class.numpy())




def save_gradcam_overlay(
        original_img_path,
        heatmap,
        output_path,
        alpha=0.4
):

    """
    Save Grad-CAM overlay image
    """


    img = cv2.imread(original_img_path)


    if img is None:
        raise FileNotFoundError(
            original_img_path
        )


    heatmap = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0])
    )


    heatmap = np.uint8(
        255 * heatmap
    )


    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    overlay = cv2.addWeighted(
        img,
        1-alpha,
        heatmap_color,
        alpha,
        0
    )


    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )


    cv2.imwrite(
        output_path,
        overlay
    )


    return output_path