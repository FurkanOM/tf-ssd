"""Loss helpers used by the SSD training script."""

from __future__ import annotations

from typing import Any

import tensorflow as tf


class CustomLoss(object):
    """Bundle the localization and confidence losses used by SSD."""

    def __init__(self, neg_pos_ratio: int, loc_loss_alpha: float) -> None:
        """Store the SSD loss hyper-parameters as tensors.

        Args:
            neg_pos_ratio (int): Negative-to-positive sample ratio for hard negative mining.
            loc_loss_alpha (float): Weight applied to the localization loss term.

        Returns:
            None: The instance is initialized in place.
        """
        self.neg_pos_ratio = tf.constant(neg_pos_ratio, dtype=tf.float32)
        self.loc_loss_alpha = tf.constant(loc_loss_alpha, dtype=tf.float32)

    def loc_loss_fn(self, actual_deltas: Any, pred_deltas: Any) -> Any:
        """Compute localization loss over positive matches only.

        Args:
            actual_deltas (Any): Encoded regression targets for each prior box.
            pred_deltas (Any): Predicted regression deltas for each prior box.

        Returns:
            Any: Per-batch localization loss tensor.
        """
        loc_loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
        loc_loss_for_all = loc_loss_fn(actual_deltas, pred_deltas)
        # TensorFlow changed the Huber reduction shape across versions, so normalize here.
        loc_loss_for_all = tf.cond(
            tf.greater(tf.rank(loc_loss_for_all), tf.constant(2)),
            lambda: tf.reduce_sum(loc_loss_for_all, axis=-1),
            lambda: loc_loss_for_all * tf.cast(tf.shape(pred_deltas)[-1], dtype=tf.float32),
        )

        # Only matched priors should contribute to localization loss.
        pos_cond = tf.reduce_any(tf.not_equal(actual_deltas, tf.constant(0.0)), axis=2)
        pos_mask = tf.cast(pos_cond, dtype=tf.float32)
        total_pos_bboxes = tf.reduce_sum(pos_mask, axis=1)

        loc_loss = tf.reduce_sum(pos_mask * loc_loss_for_all, axis=-1)
        total_pos_bboxes = tf.where(
            tf.equal(total_pos_bboxes, tf.constant(0.0)),
            tf.constant(1.0),
            total_pos_bboxes,
        )
        loc_loss = loc_loss / total_pos_bboxes
        return loc_loss * self.loc_loss_alpha

    def conf_loss_fn(self, actual_labels: Any, pred_labels: Any) -> Any:
        """Compute confidence loss with SSD-style hard negative mining.

        Args:
            actual_labels (Any): One-hot classification targets for each prior box.
            pred_labels (Any): Predicted classification probabilities for each prior box.

        Returns:
            Any: Per-batch confidence loss tensor.
        """
        conf_loss_fn = tf.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
        conf_loss_for_all = conf_loss_fn(actual_labels, pred_labels)

        pos_cond = tf.reduce_any(tf.not_equal(actual_labels[..., 1:], tf.constant(0.0)), axis=2)
        pos_mask = tf.cast(pos_cond, dtype=tf.float32)
        total_pos_bboxes = tf.reduce_sum(pos_mask, axis=1)
        total_neg_bboxes = tf.cast(total_pos_bboxes * self.neg_pos_ratio, tf.int32)

        # Rank background candidates by loss and keep the hardest negatives only.
        masked_loss = conf_loss_for_all * actual_labels[..., 0]
        sorted_loss = tf.argsort(masked_loss, direction="DESCENDING")
        sorted_loss = tf.argsort(sorted_loss)
        neg_cond = tf.less(sorted_loss, tf.expand_dims(total_neg_bboxes, axis=1))
        neg_mask = tf.cast(neg_cond, dtype=tf.float32)

        final_mask = pos_mask + neg_mask
        conf_loss = tf.reduce_sum(final_mask * conf_loss_for_all, axis=-1)
        total_pos_bboxes = tf.where(
            tf.equal(total_pos_bboxes, tf.constant(0.0)),
            tf.constant(1.0),
            total_pos_bboxes,
        )
        return conf_loss / total_pos_bboxes
