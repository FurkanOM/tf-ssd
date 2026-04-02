"""Visualization helpers for grids, boxes, and SSD predictions."""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageDraw

from utils import bbox_utils


def draw_grid_map(img: Any, grid_map: Any, stride: int) -> None:
    """Draw grid intersections on an image for debugging feature maps.

    Args:
        img (Any): Image array shown with PIL and matplotlib.
        grid_map (Any): Grid coordinates to render.
        stride (int): Feature stride associated with the grid.

    Returns:
        None: The function renders the visualization directly.
    """
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    for grid in grid_map:
        draw.rectangle(
            (
                grid[0] + stride // 2 - 2,
                grid[1] + stride // 2 - 2,
                grid[2] + stride // 2 + 2,
                grid[3] + stride // 2 + 2,
            ),
            fill=(255, 255, 255, 0),
        )
    plt.figure()
    plt.imshow(image)
    plt.show()


def draw_bboxes(imgs: Any, bboxes: Any) -> None:
    """Draw normalized bounding boxes over a batch of images.

    Args:
        imgs (Any): Batch of images.
        bboxes (Any): Normalized bounding boxes aligned with ``imgs``.

    Returns:
        None: The function renders the visualization directly.
    """
    colors = tf.constant([[1, 0, 0, 1]], dtype=tf.float32)
    imgs_with_bb = tf.image.draw_bounding_boxes(imgs, bboxes, colors)
    plt.figure()
    for img_with_bb in imgs_with_bb:
        plt.imshow(img_with_bb)
        plt.show()


def draw_bboxes_with_labels(
    img: Any,
    bboxes: Any,
    label_indices: Any,
    probs: Any,
    labels: Sequence[str]
) -> None:
    """Draw class labels and scores on top of denormalized bounding boxes.

    Args:
        img (Any): Image tensor to annotate.
        bboxes (Any): Bounding boxes in pixel coordinates.
        label_indices (Any): Predicted class indices.
        probs (Any): Predicted confidence scores.
        labels (Sequence[str]): Label names used for the prediction indices.

    Returns:
        None: The function renders the visualization directly.
    """
    colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)
    image = tf.keras.preprocessing.image.array_to_img(img)
    draw = ImageDraw.Draw(image)
    for index, bbox in enumerate(bboxes):
        y1, x1, y2, x2 = tf.split(bbox, 4)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        # Skip degenerate boxes that can appear in zero-padded predictions.
        label_index = int(label_indices[index])
        color = tuple(colors[label_index].numpy())
        label_text = "{0} {1:0.3f}".format(labels[label_index], probs[index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    plt.figure()
    plt.imshow(image)
    plt.show()


def draw_predictions(
    dataset: Any,
    pred_bboxes: Any,
    pred_labels: Any,
    pred_scores: Any,
    labels: Sequence[str],
    batch_size: int
) -> None:
    """Render decoded SSD predictions for every image in a dataset.

    Args:
        dataset (Any): Batched dataset used for prediction.
        pred_bboxes (Any): Decoded bounding boxes for the full dataset.
        pred_labels (Any): Predicted class indices for the full dataset.
        pred_scores (Any): Predicted confidence scores for the full dataset.
        labels (Sequence[str]): Label names used for the prediction indices.
        batch_size (int): Prediction batch size.

    Returns:
        None: The function renders the visualization directly.
    """
    for batch_id, image_data in enumerate(dataset):
        imgs, _, _ = image_data
        img_size = imgs.shape[1]
        start = batch_id * batch_size
        end = start + batch_size
        batch_bboxes = pred_bboxes[start:end]
        batch_labels = pred_labels[start:end]
        batch_scores = pred_scores[start:end]
        for index, img in enumerate(imgs):
            denormalized_bboxes = bbox_utils.denormalize_bboxes(batch_bboxes[index], img_size, img_size)
            draw_bboxes_with_labels(
                img,
                denormalized_bboxes,
                batch_labels[index],
                batch_scores[index],
                labels,
            )
