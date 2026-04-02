"""Dataset loading and preprocessing helpers for SSD."""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image


def preprocessing(
    image_data: Any,
    final_height: int,
    final_width: int,
    augmentation_fn: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
    evaluate: bool = False
) -> Tuple[Any, Any, Any]:
    """Resize and optionally augment a single dataset example.

    Args:
        image_data (Any): Sample from TensorFlow Datasets.
        final_height (int): Output image height.
        final_width (int): Output image width.
        augmentation_fn (Optional[Callable[[Any, Any], Tuple[Any, Any]]]): Optional augmentation callable.
        evaluate (bool): Whether to drop difficult examples for evaluation.

    Returns:
        Tuple[Any, Any, Any]: Resized image, bounding boxes, and labels.
    """
    img = image_data["image"]
    gt_boxes = image_data["objects"]["bbox"]
    gt_labels = tf.cast(image_data["objects"]["label"] + 1, tf.int32)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (final_height, final_width))
    if evaluate:
        not_diff = tf.logical_not(image_data["objects"]["is_difficult"])
        gt_boxes = gt_boxes[not_diff]
        gt_labels = gt_labels[not_diff]
    if augmentation_fn is not None:
        img, gt_boxes = augmentation_fn(img, gt_boxes)
    return img, gt_boxes, gt_labels


def get_dataset(name: str, split: str, data_dir: str = "~/tensorflow_datasets") -> Tuple[Any, Any]:
    """Load a TensorFlow Datasets split together with its metadata.

    Args:
        name (str): Dataset name such as ``voc/2007``.
        split (str): Dataset split identifier.
        data_dir (str): TFDS storage directory.

    Returns:
        Tuple[Any, Any]: Loaded dataset split and the associated dataset info object.
    """
    assert split in ["train", "train+validation", "validation", "test"]
    dataset, info = tfds.load(name, split=split, data_dir=data_dir, with_info=True)
    return dataset, info


def get_total_item_size(info: Any, split: str) -> int:
    """Return the number of examples available for a dataset split.

    Args:
        info (Any): Dataset metadata object from TFDS.
        split (str): Dataset split identifier.

    Returns:
        int: Number of examples in the requested split.
    """
    assert split in ["train", "train+validation", "validation", "test"]
    if split == "train+validation":
        return info.splits["train"].num_examples + info.splits["validation"].num_examples
    return info.splits[split].num_examples


def get_labels(info: Any) -> List[str]:
    """Return the dataset label names.

    Args:
        info (Any): Dataset metadata object from TFDS.

    Returns:
        List[str]: Foreground label names in dataset order.
    """
    return info.features["labels"].names


def get_custom_imgs(custom_image_path: str) -> List[str]:
    """Collect top-level custom image paths from a directory.

    Args:
        custom_image_path (str): Directory containing local prediction images.

    Returns:
        List[str]: File paths discovered in the top-level directory.
    """
    img_paths = []
    for path, _, filenames in os.walk(custom_image_path):
        for filename in filenames:
            img_paths.append(os.path.join(path, filename))
        break
    return img_paths


def custom_data_generator(
    img_paths: Iterable[str],
    final_height: int,
    final_width: int
) -> Iterator[Tuple[Any, Any, Any]]:
    """Yield resized local images in the same shape as the dataset pipeline.

    Args:
        img_paths (Iterable[str]): Local image paths to load.
        final_height (int): Output image height.
        final_width (int): Output image width.

    Returns:
        Iterator[Tuple[Any, Any, Any]]: Generator yielding image tensors plus dummy targets.
    """
    for img_path in img_paths:
        image = Image.open(img_path)
        resized_image = image.resize((final_width, final_height), Image.LANCZOS)
        img = np.array(resized_image)
        img = tf.image.convert_image_dtype(img, tf.float32)
        yield img, tf.constant([[]], dtype=tf.float32), tf.constant([], dtype=tf.int32)


def get_data_types() -> Tuple[Any, Any, Any]:
    """Return the TensorFlow dtypes used by the repository datasets.

    Returns:
        Tuple[Any, Any, Any]: Dtypes for image, boxes, and labels tensors.
    """
    return (tf.float32, tf.float32, tf.int32)


def get_data_shapes() -> Tuple[List[Optional[int]], List[Optional[int]], List[Optional[int]]]:
    """Return the padded tensor shapes used by the repository datasets.

    Returns:
        Tuple[List[Optional[int]], List[Optional[int]], List[Optional[int]]]: Padded tensor shapes.
    """
    return ([None, None, None], [None, None], [None])


def get_padding_values() -> Tuple[Any, Any, Any]:
    """Return the padding values used for batched dataset examples.

    Returns:
        Tuple[Any, Any, Any]: Padding values for image, boxes, and labels tensors.
    """
    return (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
