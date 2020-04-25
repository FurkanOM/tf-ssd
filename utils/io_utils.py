import os
import argparse
import tensorflow as tf
from datetime import datetime

def get_log_path(model_type, custom_postfix=""):
    """Generating log path from model_type value for tensorboard.
    inputs:
        model_type = "mobilenet_v2"
        custom_postfix = any custom string for log folder name

    outputs:
        log_path = tensorboard log path, for example: "logs/mobilenet_v2/{date}"
    """
    return "logs/{}{}/{}".format(model_type, custom_postfix, datetime.now().strftime("%Y%m%d-%H%M%S"))

def get_model_path(model_type):
    """Generating model path from model_type value for save/load model weights.
    inputs:
        model_type = "vgg16", "mobilenet_v2"

    outputs:
        model_path = os model path, for example: "trained/ssd_vgg16_model_weights.h5"
    """
    main_path = "trained"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = os.path.join(main_path, "ssd_{}_model_weights.h5".format(model_type))
    return model_path

def handle_args():
    """Handling of command line arguments using argparse library.

    outputs:
        args = parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="SSD: Single Shot MultiBox Detector Implementation")
    parser.add_argument("-handle-gpu", action="store_true", help="Tensorflow 2 GPU compatibility flag")
    parser.add_argument("--backbone", required=False,
                        default="mobilenet_v2",
                        metavar="['mobilenet_v2', 'vgg16']",
                        help="Which backbone used for the ssd")
    args = parser.parse_args()
    return args

def is_valid_backbone(backbone):
    """Handling control of given backbone is valid or not.
    inputs:
        backbone = given string from command line

    """
    assert backbone in ["mobilenet_v2", "vgg16"]

def handle_gpu_compatibility():
    """Handling of GPU issues for cuDNN initialize error and memory issues."""
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)
