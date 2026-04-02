"""Training configuration and target-generation helpers for SSD."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterator, Tuple

import tensorflow as tf

from utils import bbox_utils

SSD = {
    "vgg16": {
        "img_size": 300,
        "feature_map_shapes": [38, 19, 10, 5, 3, 1],
        "aspect_ratios": [[1., 2., 1./2.],
                         [1., 2., 1./2., 3., 1./3.],
                         [1., 2., 1./2., 3., 1./3.],
                         [1., 2., 1./2., 3., 1./3.],
                         [1., 2., 1./2.],
                         [1., 2., 1./2.]],
    },
    "mobilenet_v2": {
        "img_size": 300,
        "feature_map_shapes": [19, 10, 5, 3, 2, 1],
        "aspect_ratios": [[1., 2., 1./2.],
                         [1., 2., 1./2., 3., 1./3.],
                         [1., 2., 1./2., 3., 1./3.],
                         [1., 2., 1./2., 3., 1./3.],
                         [1., 2., 1./2.],
                         [1., 2., 1./2.]],
    }
}

def get_hyper_params(backbone: str, **kwargs: Any) -> Dict[str, Any]:
    """Return a copy of the configured hyper-parameters for a backbone.

    Args:
        backbone (str): Backbone key from the ``SSD`` configuration map.
        **kwargs (Any): Optional hyper-parameter overrides.

    Returns:
        Dict[str, Any]: Backbone-specific SSD hyper-parameters.
    """
    hyper_params = copy.deepcopy(SSD[backbone])
    hyper_params["iou_threshold"] = 0.5
    hyper_params["neg_pos_ratio"] = 3
    hyper_params["loc_loss_alpha"] = 1
    hyper_params["variances"] = [0.1, 0.1, 0.2, 0.2]
    for key, value in kwargs.items():
        if key in hyper_params and value:
            hyper_params[key] = value
    #
    return hyper_params

def scheduler(epoch: int) -> float:
    """Return the learning rate scheduled for a training epoch.

    Args:
        epoch (int): Zero-based epoch index.

    Returns:
        float: Learning rate used for the epoch.
    """
    if epoch < 100:
        return 1e-3
    if epoch < 125:
        return 1e-4
    return 1e-5

def get_step_size(total_items: int, batch_size: int) -> int:
    """Return the number of steps needed to cover a dataset split.

    Args:
        total_items (int): Number of examples in the split.
        batch_size (int): Batch size used for iteration.

    Returns:
        int: Number of steps required to consume the split.
    """
    return math.ceil(total_items / batch_size)

def generator(dataset: Any, prior_boxes: Any, hyper_params: Dict[str, Any]) -> Iterator[Tuple[Any, Tuple[Any, Any]]]:
    """Yield model inputs and SSD training targets indefinitely.

    Args:
        dataset (Any): Batched dataset yielding images, boxes, and labels.
        prior_boxes (Any): Prior boxes shared across the whole dataset.
        hyper_params (Dict[str, Any]): SSD hyper-parameters.

    Returns:
        Iterator[Tuple[Any, Tuple[Any, Any]]]: Infinite generator for ``model.fit``.
    """
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_labels = image_data
            # Convert raw annotations into the two SSD supervision tensors.
            actual_deltas, actual_labels = calculate_actual_outputs(prior_boxes, gt_boxes, gt_labels, hyper_params)
            yield img, (actual_deltas, actual_labels)

def calculate_actual_outputs(
    prior_boxes: Any,
    gt_boxes: Any,
    gt_labels: Any,
    hyper_params: Dict[str, Any]
) -> Tuple[Any, Any]:
    """Generate SSD delta and class targets for a batch of ground truths.

    Args:
        prior_boxes (Any): Prior boxes shared across the batch.
        gt_boxes (Any): Ground-truth boxes for the batch.
        gt_labels (Any): Ground-truth labels for the batch.
        hyper_params (Dict[str, Any]): SSD hyper-parameters.

    Returns:
        Tuple[Any, Any]: Encoded box deltas and one-hot class targets.
    """
    total_labels = hyper_params["total_labels"]
    iou_threshold = hyper_params["iou_threshold"]
    variances = hyper_params["variances"]
    # Match each prior to the ground-truth box with the best overlap.
    iou_map = bbox_utils.generate_iou_map(prior_boxes, gt_boxes)
    max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    pos_cond = tf.greater(merged_iou_map, iou_threshold)

    # Unmatched priors stay zero so they become background examples.
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_cond, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    bbox_deltas = bbox_utils.get_deltas_from_bboxes(prior_boxes, expanded_gt_boxes) / variances

    gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_labels = tf.where(pos_cond, gt_labels_map, tf.zeros_like(gt_labels_map))
    bbox_labels = tf.one_hot(expanded_gt_labels, total_labels)
    return bbox_deltas, bbox_labels
