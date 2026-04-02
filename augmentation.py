"""Data augmentation helpers used during SSD training."""

from __future__ import annotations

from typing import Any, Callable, Tuple

import tensorflow as tf

from utils import bbox_utils


TensorPair = Tuple[Any, Any]
AugmentationFn = Callable[..., TensorPair]


def apply(img: Any, gt_boxes: Any) -> TensorPair:
    """Apply the repository's SSD augmentation pipeline.

    Args:
        img (Any): Image tensor shaped like ``(height, width, channels)``.
        gt_boxes (Any): Bounding boxes in normalized ``[y1, x1, y2, x2]`` format.

    Returns:
        TensorPair: Augmented image tensor and adjusted bounding boxes.
    """
    color_methods = [random_brightness, random_contrast, random_hue, random_saturation]
    geometric_methods = [patch, flip_horizontally]
    for augmentation_method in geometric_methods + color_methods:
        img, gt_boxes = randomly_apply_operation(augmentation_method, img, gt_boxes)
    img = tf.clip_by_value(img, 0, 1)
    return img, gt_boxes


def get_random_bool() -> Any:
    """Sample a boolean tensor with equal probability.

    Returns:
        Any: Scalar boolean tensor used to gate a random augmentation branch.
    """
    return tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5)


def randomly_apply_operation(
    operation: AugmentationFn,
    img: Any,
    gt_boxes: Any,
    *args: Any
) -> TensorPair:
    """Apply an augmentation function conditionally.

    Args:
        operation (AugmentationFn): Augmentation function to invoke.
        img (Any): Image tensor shaped like ``(height, width, channels)``.
        gt_boxes (Any): Bounding boxes aligned with ``img``.
        *args (Any): Extra positional arguments forwarded to ``operation``.

    Returns:
        TensorPair: Either the transformed tensors or the original inputs.
    """
    return tf.cond(
        get_random_bool(),
        lambda: operation(img, gt_boxes, *args),
        lambda: (img, gt_boxes),
    )


def random_brightness(img: Any, gt_boxes: Any, max_delta: float = 0.12) -> TensorPair:
    """Randomly perturb image brightness while keeping boxes unchanged.

    Args:
        img (Any): Image tensor shaped like ``(height, width, channels)``.
        gt_boxes (Any): Bounding boxes aligned with ``img``.
        max_delta (float): Maximum brightness delta applied by TensorFlow.

    Returns:
        TensorPair: Brightness-adjusted image tensor and unchanged boxes.
    """
    return tf.image.random_brightness(img, max_delta), gt_boxes


def random_contrast(
    img: Any,
    gt_boxes: Any,
    lower: float = 0.5,
    upper: float = 1.5
) -> TensorPair:
    """Randomly perturb image contrast while keeping boxes unchanged.

    Args:
        img (Any): Image tensor shaped like ``(height, width, channels)``.
        gt_boxes (Any): Bounding boxes aligned with ``img``.
        lower (float): Minimum contrast multiplier.
        upper (float): Maximum contrast multiplier.

    Returns:
        TensorPair: Contrast-adjusted image tensor and unchanged boxes.
    """
    return tf.image.random_contrast(img, lower, upper), gt_boxes


def random_hue(img: Any, gt_boxes: Any, max_delta: float = 0.08) -> TensorPair:
    """Randomly perturb image hue while keeping boxes unchanged.

    Args:
        img (Any): Image tensor shaped like ``(height, width, channels)``.
        gt_boxes (Any): Bounding boxes aligned with ``img``.
        max_delta (float): Maximum hue delta applied by TensorFlow.

    Returns:
        TensorPair: Hue-adjusted image tensor and unchanged boxes.
    """
    return tf.image.random_hue(img, max_delta), gt_boxes


def random_saturation(
    img: Any,
    gt_boxes: Any,
    lower: float = 0.5,
    upper: float = 1.5
) -> TensorPair:
    """Randomly perturb image saturation while keeping boxes unchanged.

    Args:
        img (Any): Image tensor shaped like ``(height, width, channels)``.
        gt_boxes (Any): Bounding boxes aligned with ``img``.
        lower (float): Minimum saturation multiplier.
        upper (float): Maximum saturation multiplier.

    Returns:
        TensorPair: Saturation-adjusted image tensor and unchanged boxes.
    """
    return tf.image.random_saturation(img, lower, upper), gt_boxes


def flip_horizontally(img: Any, gt_boxes: Any) -> TensorPair:
    """Flip an image horizontally and mirror its bounding boxes.

    Args:
        img (Any): Image tensor shaped like ``(height, width, channels)``.
        gt_boxes (Any): Bounding boxes in normalized ``[y1, x1, y2, x2]`` format.

    Returns:
        TensorPair: Flipped image tensor and mirrored boxes.
    """
    flipped_img = tf.image.flip_left_right(img)
    flipped_gt_boxes = tf.stack(
        [
            gt_boxes[..., 0],
            1.0 - gt_boxes[..., 3],
            gt_boxes[..., 2],
            1.0 - gt_boxes[..., 1],
        ],
        axis=-1,
    )
    return flipped_img, flipped_gt_boxes


def get_random_min_overlap() -> Any:
    """Sample the minimum object-overlap threshold used for random crops.

    Returns:
        Any: Scalar tensor holding one of the SSD crop overlap thresholds.
    """
    overlaps = tf.constant([0.1, 0.3, 0.5, 0.7, 0.9], dtype=tf.float32)
    index = tf.random.uniform((), minval=0, maxval=tf.shape(overlaps)[0], dtype=tf.int32)
    return overlaps[index]


def expand_image(img: Any, gt_boxes: Any, height: Any, width: Any) -> TensorPair:
    """Place an image on a larger canvas and update the box coordinates.

    Args:
        img (Any): Image tensor shaped like ``(height, width, channels)``.
        gt_boxes (Any): Bounding boxes in normalized ``[y1, x1, y2, x2]`` format.
        height (Any): Original image height tensor.
        width (Any): Original image width tensor.

    Returns:
        TensorPair: Expanded image tensor and renormalized boxes.
    """
    expansion_ratio = tf.random.uniform((), minval=1, maxval=4, dtype=tf.float32)
    final_height = tf.round(height * expansion_ratio)
    final_width = tf.round(width * expansion_ratio)
    pad_left = tf.round(tf.random.uniform((), minval=0, maxval=final_width - width, dtype=tf.float32))
    pad_top = tf.round(tf.random.uniform((), minval=0, maxval=final_height - height, dtype=tf.float32))
    pad_right = final_width - (width + pad_left)
    pad_bottom = final_height - (height + pad_top)

    # Fill padded regions with the image mean so expansion does not introduce hard borders.
    mean, _ = tf.nn.moments(img, [0, 1])
    expanded_image = tf.pad(
        img,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        constant_values=-1,
    )
    expanded_image = tf.where(expanded_image == -1, mean, expanded_image)

    min_max = tf.stack(
        [-pad_top, -pad_left, pad_bottom + height, pad_right + width],
        axis=-1,
    ) / [height, width, height, width]
    modified_gt_boxes = bbox_utils.renormalize_bboxes_with_min_max(gt_boxes, min_max)
    return expanded_image, modified_gt_boxes


def patch(img: Any, gt_boxes: Any) -> TensorPair:
    """Sample a distorted patch and remap the boxes into the sampled region.

    Args:
        img (Any): Image tensor shaped like ``(height, width, channels)``.
        gt_boxes (Any): Bounding boxes in normalized ``[y1, x1, y2, x2]`` format.

    Returns:
        TensorPair: Cropped-and-resized image tensor and remapped boxes.
    """
    img_shape = tf.cast(tf.shape(img), dtype=tf.float32)
    org_height, org_width = img_shape[0], img_shape[1]
    # SSD optionally expands the canvas before sampling a crop to keep more context.
    img, gt_boxes = randomly_apply_operation(expand_image, img, gt_boxes, org_height, org_width)
    min_overlap = get_random_min_overlap()

    # TensorFlow returns both the crop window and its normalized box coordinates.
    begin, size, new_boundaries = tf.image.sample_distorted_bounding_box(
        tf.shape(img),
        bounding_boxes=tf.expand_dims(gt_boxes, 0),
        aspect_ratio_range=[0.5, 2.0],
        min_object_covered=min_overlap,
    )

    img = tf.slice(img, begin, size)
    # Resize back to the original resolution so the rest of the pipeline stays unchanged.
    img = tf.image.resize(img, (org_height, org_width))
    gt_boxes = bbox_utils.renormalize_bboxes_with_min_max(gt_boxes, new_boundaries[0, 0])
    return img, gt_boxes
