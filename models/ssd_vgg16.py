"""VGG16-based SSD model definition."""

from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Layer, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from .header import get_head_from_outputs


class L2Normalization(Layer):
    """Normalize feature-map channels before they are fused into SSD heads."""

    def __init__(self, scale_factor: float, **kwargs: Any) -> None:
        """Store the initial scaling factor used by the layer.

        Args:
            scale_factor (float): Initial per-channel scaling factor.
            **kwargs (Any): Extra Keras layer keyword arguments.

        Returns:
            None: The layer is initialized in place.
        """
        super(L2Normalization, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def get_config(self) -> Dict[str, Any]:
        """Serialize the layer configuration for model saving.

        Returns:
            Dict[str, Any]: Serializable Keras layer configuration.
        """
        config = super(L2Normalization, self).get_config()
        config.update({"scale_factor": self.scale_factor})
        return config

    def build(self, input_shape: Any) -> None:
        """Create the learnable channel-wise scaling weights.

        Args:
            input_shape (Any): Shape of the incoming feature map.

        Returns:
            None: Layer weights are created in place.
        """
        # SSD learns a separate scale factor for each feature-map channel.
        init_scale_factor = tf.fill((input_shape[-1],), float(self.scale_factor))
        self.scale = tf.Variable(init_scale_factor, trainable=True)

    def call(self, inputs: Any) -> Any:
        """Apply L2 normalization followed by the learnable scale.

        Args:
            inputs (Any): Feature map tensor to normalize.

        Returns:
            Any: Normalized feature map tensor.
        """
        return tf.nn.l2_normalize(inputs, axis=-1) * self.scale


def get_model(hyper_params: Dict[str, Any]) -> Model:
    """Build the SSD model that uses a VGG16-style backbone.

    Args:
        hyper_params (Dict[str, Any]): SSD hyper-parameters including class counts.

    Returns:
        Model: Keras model that predicts SSD box deltas and class scores.
    """
    scale_factor = 20.0
    reg_factor = 5e-4

    inputs = Input(shape=(None, None, 3), name="input")
    # Keep the original SSD/VGG block structure so saved weights remain compatible.
    conv1_1 = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv1_1")(inputs)
    conv1_2 = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv1_2")(conv1_1)
    pool1 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv2_1")(pool1)
    conv2_2 = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv2_2")(conv2_1)
    pool2 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv3_1")(pool2)
    conv3_2 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv3_2")(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv3_3")(conv3_2)
    pool3 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv4_1")(pool3)
    conv4_2 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv4_2")(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv4_3")(conv4_2)
    pool4 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool4")(conv4_3)

    conv5_1 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv5_1")(pool4)
    conv5_2 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv5_2")(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv5_3")(conv5_2)
    pool5 = MaxPool2D((3, 3), strides=(1, 1), padding="same", name="pool5")(conv5_3)

    conv6 = Conv2D(1024, (3, 3), dilation_rate=6, padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv6")(pool5)
    conv7 = Conv2D(1024, (1, 1), strides=(1, 1), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv7")(conv6)

    conv8_1 = Conv2D(256, (1, 1), strides=(1, 1), padding="valid", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv8_1")(conv7)
    conv8_2 = Conv2D(512, (3, 3), strides=(2, 2), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv8_2")(conv8_1)
    conv9_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv9_1")(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv9_2")(conv9_1)
    conv10_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv10_1")(conv9_2)
    conv10_2 = Conv2D(256, (3, 3), strides=(1, 1), padding="valid", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv10_2")(conv10_1)
    conv11_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv11_1")(conv10_2)
    conv11_2 = Conv2D(256, (3, 3), strides=(1, 1), padding="valid", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv11_2")(conv11_1)

    # The low-level conv4_3 features are rescaled before they join the SSD heads.
    conv4_3_norm = L2Normalization(scale_factor)(conv4_3)
    pred_deltas, pred_labels = get_head_from_outputs(
        hyper_params,
        [conv4_3_norm, conv7, conv8_2, conv9_2, conv10_2, conv11_2],
    )
    return Model(inputs=inputs, outputs=[pred_deltas, pred_labels])


def init_model(model: Model) -> None:
    """Build the model graph eagerly with a dummy input batch.

    Args:
        model (Model): SSD model instance to initialize.

    Returns:
        None: The model graph is built in place.
    """
    model(tf.random.uniform((1, 512, 512, 3)))
