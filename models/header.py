"""Prediction-head builders for SSD feature maps."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Layer


class HeadWrapper(Layer):
    """Flatten and concatenate multiple detection feature maps."""

    def __init__(self, last_dimension: int, **kwargs: Any) -> None:
        """Store the output dimension expected for each flattened feature map.

        Args:
            last_dimension (int): Final channel count for each flattened prediction tensor.
            **kwargs (Any): Extra Keras layer keyword arguments.

        Returns:
            None: The layer is initialized in place.
        """
        super(HeadWrapper, self).__init__(**kwargs)
        self.last_dimension = last_dimension

    def get_config(self) -> Dict[str, Any]:
        """Serialize the layer configuration for model saving.

        Returns:
            Dict[str, Any]: Serializable Keras layer configuration.
        """
        config = super(HeadWrapper, self).get_config()
        config.update({"last_dimension": self.last_dimension})
        return config

    def call(self, inputs: Sequence[Any]) -> Any:
        """Reshape each feature map and concatenate them along the box axis.

        Args:
            inputs (Sequence[Any]): Detection head feature maps.

        Returns:
            Any: Concatenated prediction tensor.
        """
        batch_size = tf.shape(inputs[0])[0]
        outputs = []
        for conv_layer in inputs:
            # Flatten the spatial grid and anchor dimensions into one detection axis.
            outputs.append(tf.reshape(conv_layer, (batch_size, -1, self.last_dimension)))
        return tf.concat(outputs, axis=1)


def get_head_from_outputs(hyper_params: Dict[str, Any], outputs: Sequence[Any]) -> Tuple[Any, Any]:
    """Build SSD classification and box-regression heads from feature maps.

    Args:
        hyper_params (Dict[str, Any]): SSD hyper-parameters including class counts.
        outputs (Sequence[Any]): Feature maps used for SSD detection heads.

    Returns:
        Tuple[Any, Any]: Box-regression head and classification head outputs.
    """
    total_labels = hyper_params["total_labels"]
    len_aspect_ratios = [len(aspect_ratio) + 1 for aspect_ratio in hyper_params["aspect_ratios"]]
    labels_head = []
    boxes_head = []
    for index, output in enumerate(outputs):
        aspect_ratio = len_aspect_ratios[index]
        labels_head.append(
            Conv2D(
                aspect_ratio * total_labels,
                (3, 3),
                padding="same",
                name="{}_conv_label_output".format(index + 1),
            )(output)
        )
        boxes_head.append(
            Conv2D(
                aspect_ratio * 4,
                (3, 3),
                padding="same",
                name="{}_conv_boxes_output".format(index + 1),
            )(output)
        )

    pred_labels = HeadWrapper(total_labels, name="labels_head")(labels_head)
    pred_labels = Activation("softmax", name="conf")(pred_labels)
    pred_deltas = HeadWrapper(4, name="loc")(boxes_head)
    return pred_deltas, pred_labels
