"""Decoding layers that convert SSD outputs into final detections."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from utils import bbox_utils


class SSDDecoder(Layer):
    """Convert SSD outputs into filtered boxes, labels, and scores."""

    def __init__(
        self,
        prior_boxes: Any,
        variances: Sequence[float],
        max_total_size: int = 200,
        score_threshold: float = 0.5,
        **kwargs: Any
    ) -> None:
        """Store the decoding configuration for inference.

        Args:
            prior_boxes (Any): Prior boxes used to decode regression outputs.
            variances (Sequence[float]): Scaling factors for SSD deltas.
            max_total_size (int): Maximum number of decoded boxes to keep.
            score_threshold (float): Minimum score required before NMS.
            **kwargs (Any): Extra Keras layer keyword arguments.

        Returns:
            None: The layer is initialized in place.
        """
        super(SSDDecoder, self).__init__(**kwargs)
        self.prior_boxes = prior_boxes
        self.variances = variances
        self.max_total_size = max_total_size
        self.score_threshold = score_threshold

    def get_config(self) -> Dict[str, Any]:
        """Serialize the decoder configuration for model saving.

        Returns:
            Dict[str, Any]: Serializable Keras layer configuration.
        """
        config = super(SSDDecoder, self).get_config()
        config.update(
            {
                "prior_boxes": self.prior_boxes.numpy(),
                "variances": self.variances,
                "max_total_size": self.max_total_size,
                "score_threshold": self.score_threshold,
            }
        )
        return config

    def call(self, inputs: Sequence[Any]) -> Tuple[Any, Any, Any]:
        """Decode SSD outputs and apply non-max suppression.

        Args:
            inputs (Sequence[Any]): Predicted deltas and class probabilities.

        Returns:
            Tuple[Any, Any, Any]: Final boxes, labels, and scores.
        """
        pred_deltas = inputs[0]
        pred_label_probs = inputs[1]
        batch_size = tf.shape(pred_deltas)[0]

        # SSD predicts encoded offsets, so scale them back before decoding.
        pred_deltas *= self.variances
        pred_bboxes = bbox_utils.get_bboxes_from_deltas(self.prior_boxes, pred_deltas)

        # Suppress background scores so only foreground classes compete in NMS.
        pred_labels_map = tf.expand_dims(tf.argmax(pred_label_probs, -1), -1)
        pred_labels = tf.where(
            tf.not_equal(pred_labels_map, 0),
            pred_label_probs,
            tf.zeros_like(pred_label_probs),
        )
        pred_bboxes = tf.reshape(pred_bboxes, (batch_size, -1, 1, 4))

        final_bboxes, final_scores, final_labels, _ = bbox_utils.non_max_suppression(
            pred_bboxes,
            pred_labels,
            max_output_size_per_class=self.max_total_size,
            max_total_size=self.max_total_size,
            score_threshold=self.score_threshold,
        )
        return final_bboxes, final_labels, final_scores


def get_decoder_model(base_model: Any, prior_boxes: Any, hyper_params: Dict[str, Any]) -> Model:
    """Wrap an SSD base model with the repository's decoding layer.

    Args:
        base_model (Any): Base SSD model that predicts deltas and class scores.
        prior_boxes (Any): Prior boxes used for decoding.
        hyper_params (Dict[str, Any]): SSD hyper-parameters including variances.

    Returns:
        Model: Keras model that outputs decoded detections.
    """
    bboxes, classes, scores = SSDDecoder(prior_boxes, hyper_params["variances"])(base_model.output)
    return Model(inputs=base_model.input, outputs=[bboxes, classes, scores])
