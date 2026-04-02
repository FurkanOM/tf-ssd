"""Bounding-box helpers shared across training and inference."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import tensorflow as tf


def non_max_suppression(pred_bboxes: Any, pred_labels: Any, **kwargs: Any) -> Any:
    """Run TensorFlow's combined non-max suppression helper.

    Args:
        pred_bboxes (Any): Bounding boxes shaped for ``combined_non_max_suppression``.
        pred_labels (Any): Per-class scores aligned with ``pred_bboxes``.
        **kwargs (Any): Extra keyword arguments forwarded to TensorFlow.

    Returns:
        Any: TensorFlow's non-max suppression result tuple.
    """
    return tf.image.combined_non_max_suppression(pred_bboxes, pred_labels, **kwargs)


def generate_iou_map(
    bboxes: Any,
    gt_boxes: Any,
    transpose_perm: Optional[Sequence[int]] = None
) -> Any:
    """Compute an IoU tensor between proposal boxes and ground-truth boxes.

    Args:
        bboxes (Any): Proposal boxes or priors in ``[y1, x1, y2, x2]`` format.
        gt_boxes (Any): Ground-truth boxes in ``[y1, x1, y2, x2]`` format.
        transpose_perm (Optional[Sequence[int]]): Permutation used to align ground-truth axes.

    Returns:
        Any: Pairwise IoU tensor aligned with the input ranks.
    """
    transpose_perm = transpose_perm or [0, 2, 1]
    gt_rank = tf.rank(gt_boxes)
    gt_expand_axis = gt_rank - 2

    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(bboxes, 4, axis=-1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=-1)
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1)

    # Broadcast proposal and ground-truth corners into the same comparison space.
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, transpose_perm))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, transpose_perm))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, transpose_perm))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, transpose_perm))
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    union_area = tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, gt_expand_axis) - intersection_area
    return intersection_area / union_area


def get_bboxes_from_deltas(prior_boxes: Any, deltas: Any) -> Any:
    """Decode SSD deltas back into bounding boxes.

    Args:
        prior_boxes (Any): Prior boxes in normalized ``[y1, x1, y2, x2]`` format.
        deltas (Any): Predicted deltas in ``[dy, dx, dh, dw]`` format.

    Returns:
        Any: Decoded boxes in normalized ``[y1, x1, y2, x2]`` format.
    """
    all_pbox_width = prior_boxes[..., 3] - prior_boxes[..., 1]
    all_pbox_height = prior_boxes[..., 2] - prior_boxes[..., 0]
    all_pbox_ctr_x = prior_boxes[..., 1] + 0.5 * all_pbox_width
    all_pbox_ctr_y = prior_boxes[..., 0] + 0.5 * all_pbox_height

    all_bbox_width = tf.exp(deltas[..., 3]) * all_pbox_width
    all_bbox_height = tf.exp(deltas[..., 2]) * all_pbox_height
    all_bbox_ctr_x = (deltas[..., 1] * all_pbox_width) + all_pbox_ctr_x
    all_bbox_ctr_y = (deltas[..., 0] * all_pbox_height) + all_pbox_ctr_y

    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_height + y1
    x2 = all_bbox_width + x1
    return tf.stack([y1, x1, y2, x2], axis=-1)


def get_deltas_from_bboxes(bboxes: Any, gt_boxes: Any) -> Any:
    """Encode ground-truth boxes as SSD deltas relative to prior boxes.

    Args:
        bboxes (Any): Prior boxes in normalized ``[y1, x1, y2, x2]`` format.
        gt_boxes (Any): Ground-truth boxes matched to the priors.

    Returns:
        Any: Encoded deltas in ``[dy, dx, dh, dw]`` format.
    """
    bbox_width = bboxes[..., 3] - bboxes[..., 1]
    bbox_height = bboxes[..., 2] - bboxes[..., 0]
    bbox_ctr_x = bboxes[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = bboxes[..., 0] + 0.5 * bbox_height

    gt_width = gt_boxes[..., 3] - gt_boxes[..., 1]
    gt_height = gt_boxes[..., 2] - gt_boxes[..., 0]
    gt_ctr_x = gt_boxes[..., 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[..., 0] + 0.5 * gt_height

    # Guard against zero-sized priors before taking divisions and logarithms.
    bbox_width = tf.where(tf.equal(bbox_width, 0), 1e-3, bbox_width)
    bbox_height = tf.where(tf.equal(bbox_height, 0), 1e-3, bbox_height)
    delta_x = tf.where(
        tf.equal(gt_width, 0),
        tf.zeros_like(gt_width),
        tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width),
    )
    delta_y = tf.where(
        tf.equal(gt_height, 0),
        tf.zeros_like(gt_height),
        tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height),
    )
    delta_w = tf.where(
        tf.equal(gt_width, 0),
        tf.zeros_like(gt_width),
        tf.math.log(gt_width / bbox_width),
    )
    delta_h = tf.where(
        tf.equal(gt_height, 0),
        tf.zeros_like(gt_height),
        tf.math.log(gt_height / bbox_height),
    )
    return tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1)


def get_scale_for_nth_feature_map(
    k: int,
    m: int = 6,
    scale_min: float = 0.2,
    scale_max: float = 0.9
) -> float:
    """Return the SSD paper's scale value for the ``k``-th feature map.

    Args:
        k (int): One-based feature map index.
        m (int): Total number of feature maps used for detection.
        scale_min (float): Minimum scale from the SSD paper.
        scale_max (float): Maximum scale from the SSD paper.

    Returns:
        float: Normalized scale assigned to the selected feature map.
    """
    return scale_min + ((scale_max - scale_min) / (m - 1)) * (k - 1)


def generate_base_prior_boxes(
    aspect_ratios: Sequence[float],
    feature_map_index: int,
    total_feature_map: int
) -> Any:
    """Create the normalized prior boxes for a single feature map cell.

    Args:
        aspect_ratios (Sequence[float]): Aspect ratios configured for the feature map.
        feature_map_index (int): One-based feature map index.
        total_feature_map (int): Total number of detection feature maps.

    Returns:
        Any: Base prior boxes centered around the origin for one cell.
    """
    current_scale = get_scale_for_nth_feature_map(feature_map_index, m=total_feature_map)
    next_scale = get_scale_for_nth_feature_map(feature_map_index + 1, m=total_feature_map)
    base_prior_boxes = []
    for aspect_ratio in aspect_ratios:
        height = current_scale / tf.sqrt(aspect_ratio)
        width = current_scale * tf.sqrt(aspect_ratio)
        base_prior_boxes.append([-height / 2, -width / 2, height / 2, width / 2])
    # SSD adds one extra square box that interpolates between consecutive scales.
    height = width = tf.sqrt(current_scale * next_scale)
    base_prior_boxes.append([-height / 2, -width / 2, height / 2, width / 2])
    return tf.cast(base_prior_boxes, dtype=tf.float32)


def generate_prior_boxes(feature_map_shapes: Sequence[int], aspect_ratios: Sequence[Sequence[float]]) -> Any:
    """Generate all SSD prior boxes for the configured feature pyramid.

    Args:
        feature_map_shapes (Sequence[int]): Spatial sizes for each detection feature map.
        aspect_ratios (Sequence[Sequence[float]]): Aspect ratios configured per feature map.

    Returns:
        Any: Concatenated prior boxes clipped into the ``[0, 1]`` range.
    """
    prior_boxes = []
    for index, feature_map_shape in enumerate(feature_map_shapes):
        base_prior_boxes = generate_base_prior_boxes(
            aspect_ratios[index],
            index + 1,
            len(feature_map_shapes),
        )
        # Build normalized center coordinates for every cell in the current feature map.
        stride = 1 / feature_map_shape
        grid_coords = tf.cast(
            tf.range(0, feature_map_shape) / feature_map_shape + stride / 2,
            dtype=tf.float32,
        )
        grid_x, grid_y = tf.meshgrid(grid_coords, grid_coords)
        flat_grid_x = tf.reshape(grid_x, (-1,))
        flat_grid_y = tf.reshape(grid_y, (-1,))
        grid_map = tf.stack([flat_grid_y, flat_grid_x, flat_grid_y, flat_grid_x], axis=-1)

        prior_boxes_for_feature_map = tf.reshape(base_prior_boxes, (1, -1, 4)) + tf.reshape(
            grid_map,
            (-1, 1, 4),
        )
        prior_boxes_for_feature_map = tf.reshape(prior_boxes_for_feature_map, (-1, 4))
        prior_boxes.append(prior_boxes_for_feature_map)
    prior_boxes = tf.concat(prior_boxes, axis=0)
    return tf.clip_by_value(prior_boxes, 0, 1)


def renormalize_bboxes_with_min_max(bboxes: Any, min_max: Any) -> Any:
    """Renormalize boxes into a new coordinate frame defined by min/max bounds.

    Args:
        bboxes (Any): Bounding boxes in normalized ``[y1, x1, y2, x2]`` format.
        min_max (Any): Crop boundaries in ``[y_min, x_min, y_max, x_max]`` format.

    Returns:
        Any: Boxes renormalized to the supplied boundaries.
    """
    y_min, x_min, y_max, x_max = tf.split(min_max, 4)
    renormalized_bboxes = bboxes - tf.concat([y_min, x_min, y_min, x_min], axis=-1)
    renormalized_bboxes /= tf.concat(
        [y_max - y_min, x_max - x_min, y_max - y_min, x_max - x_min],
        axis=-1,
    )
    return tf.clip_by_value(renormalized_bboxes, 0, 1)


def normalize_bboxes(bboxes: Any, height: Any, width: Any) -> Any:
    """Normalize pixel-space boxes into the ``[0, 1]`` range.

    Args:
        bboxes (Any): Bounding boxes in pixel coordinates.
        height (Any): Image height used for normalization.
        width (Any): Image width used for normalization.

    Returns:
        Any: Bounding boxes in normalized ``[y1, x1, y2, x2]`` format.
    """
    y1 = bboxes[..., 0] / height
    x1 = bboxes[..., 1] / width
    y2 = bboxes[..., 2] / height
    x2 = bboxes[..., 3] / width
    return tf.stack([y1, x1, y2, x2], axis=-1)


def denormalize_bboxes(bboxes: Any, height: Any, width: Any) -> Any:
    """Convert normalized boxes back to pixel coordinates.

    Args:
        bboxes (Any): Bounding boxes in normalized ``[y1, x1, y2, x2]`` format.
        height (Any): Image height used for denormalization.
        width (Any): Image width used for denormalization.

    Returns:
        Any: Rounded bounding boxes in pixel coordinates.
    """
    y1 = bboxes[..., 0] * height
    x1 = bboxes[..., 1] * width
    y2 = bboxes[..., 2] * height
    x2 = bboxes[..., 3] * width
    return tf.round(tf.stack([y1, x1, y2, x2], axis=-1))
