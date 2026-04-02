"""Prediction entry point for the TensorFlow SSD models."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import tensorflow as tf

from models.decoder import get_decoder_model
from utils import bbox_utils, data_utils, drawing_utils, eval_utils, io_utils, train_utils


def _get_model_fns(backbone: str) -> Tuple[Callable[[Dict[str, Any]], Any], Callable[[Any], None]]:
    """Return the SSD model factory functions for the requested backbone.

    Args:
        backbone (str): Backbone name selected by the caller.

    Returns:
        Tuple[Callable[[Dict[str, Any]], Any], Callable[[Any], None]]: Model factory and initializer.
    """
    if backbone == "mobilenet_v2":
        from models.ssd_mobilenet_v2 import get_model, init_model
    else:
        from models.ssd_vgg16 import get_model, init_model
    return get_model, init_model


def main() -> None:
    """Load a trained SSD model and run prediction or evaluation.

    Returns:
        None: Predictions are rendered or evaluated directly.
    """
    args = io_utils.handle_args()
    if args.handle_gpu:
        io_utils.handle_gpu_compatibility()

    batch_size = 32
    evaluate = False
    use_custom_images = False
    custom_image_path = "data/images/"
    backbone = args.backbone
    io_utils.is_valid_backbone(backbone)

    get_model, init_model = _get_model_fns(backbone)
    hyper_params = train_utils.get_hyper_params(backbone)

    test_data, info = data_utils.get_dataset("voc/2007", "test")
    total_items = data_utils.get_total_item_size(info, "test")
    labels = ["bg"] + data_utils.get_labels(info)
    hyper_params["total_labels"] = len(labels)
    img_size = hyper_params["img_size"]

    data_types = data_utils.get_data_types()
    data_shapes = data_utils.get_data_shapes()
    padding_values = data_utils.get_padding_values()

    if use_custom_images:
        # Keep custom-image inference on the same tensor contract as TFDS examples.
        img_paths = data_utils.get_custom_imgs(custom_image_path)
        total_items = len(img_paths)
        test_data = tf.data.Dataset.from_generator(
            lambda: data_utils.custom_data_generator(img_paths, img_size, img_size),
            data_types,
            data_shapes,
        )
    else:
        test_data = test_data.map(
            lambda x: data_utils.preprocessing(x, img_size, img_size, evaluate=evaluate)
        )

    test_data = test_data.padded_batch(
        batch_size,
        padded_shapes=data_shapes,
        padding_values=padding_values,
    )

    # Build the base model first so the saved weights can be restored safely.
    ssd_model = get_model(hyper_params)
    init_model(ssd_model)
    ssd_model_path = io_utils.get_model_path(backbone)
    ssd_model.load_weights(ssd_model_path)

    # Priors are deterministic for a backbone, so they can be created once per run.
    prior_boxes = bbox_utils.generate_prior_boxes(
        hyper_params["feature_map_shapes"],
        hyper_params["aspect_ratios"],
    )
    ssd_decoder_model = get_decoder_model(ssd_model, prior_boxes, hyper_params)

    step_size = train_utils.get_step_size(total_items, batch_size)
    pred_bboxes, pred_labels, pred_scores = ssd_decoder_model.predict(
        test_data,
        steps=step_size,
        verbose=1,
    )

    if evaluate:
        eval_utils.evaluate_predictions(
            test_data,
            pred_bboxes,
            pred_labels,
            pred_scores,
            labels,
            batch_size,
        )
    else:
        drawing_utils.draw_predictions(
            test_data,
            pred_bboxes,
            pred_labels,
            pred_scores,
            labels,
            batch_size,
        )


if __name__ == "__main__":
    main()
