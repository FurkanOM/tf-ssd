"""Filesystem, argument parsing, and runtime environment helpers."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import tensorflow as tf


def get_log_path(model_type: str, custom_postfix: str = "") -> str:
    """Build a timestamped TensorBoard log directory path.

    Args:
        model_type (str): Backbone or model identifier used in the log path.
        custom_postfix (str): Optional suffix appended to the log directory name.

    Returns:
        str: Timestamped TensorBoard log directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return "logs/{}{}/{}".format(model_type, custom_postfix, timestamp)


def get_model_path(model_type: str) -> str:
    """Build the model-weight path and create the output directory if needed.

    Args:
        model_type (str): Backbone or model identifier used in the file name.

    Returns:
        str: Output path used for saving and loading model weights.
    """
    main_path = "trained"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    return os.path.join(main_path, "ssd_{}_model_weights.h5".format(model_type))


def handle_args() -> argparse.Namespace:
    """Parse the repository's command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="SSD: Single Shot MultiBox Detector Implementation")
    parser.add_argument("-handle-gpu", action="store_true", help="Tensorflow 2 GPU compatibility flag")
    parser.add_argument(
        "--backbone",
        required=False,
        default="mobilenet_v2",
        metavar="['mobilenet_v2', 'vgg16']",
        help="Which backbone used for the ssd",
    )
    return parser.parse_args()


def is_valid_backbone(backbone: str) -> None:
    """Assert that the requested backbone is supported by this repository.

    Args:
        backbone (str): Backbone name supplied by the caller.

    Returns:
        None: The function raises ``AssertionError`` for invalid values.
    """
    assert backbone in ["mobilenet_v2", "vgg16"]


def handle_gpu_compatibility() -> None:
    """Enable memory growth on visible GPUs when TensorFlow allows it.

    Returns:
        None: TensorFlow GPU runtime is configured in place when possible.
    """
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:
        print(exc)
