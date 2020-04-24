import os
import argparse
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import bbox_utils

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

def scheduler(epoch):
    """Generating learning rate value for a given epoch.
    inputs:
        epoch = number of current epoch

    outputs:
        learning_rate = float learning rate value
    """
    if epoch < 80:
        return 1e-3
    elif epoch < 120:
        return 1e-4
    else:
        return 1e-5

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

def get_hyper_params(backbone, **kwargs):
    """Generating hyper params in a dynamic way.
    inputs:
        **kwargs = any value could be updated in the hyper_params

    outputs:
        hyper_params = dictionary
    """
    hyper_params = SSD[backbone]
    hyper_params["iou_threshold"] = 0.5
    hyper_params["neg_pos_ratio"] = 3
    hyper_params["loc_loss_alpha"] = 1
    hyper_params["variances"] = [0.1, 0.1, 0.2, 0.2]
    for key, value in kwargs.items():
        if key in hyper_params and value:
            hyper_params[key] = value
    #
    return hyper_params

def get_padded_batch_params():
    """Generating padded batch params for tensorflow datasets.
    outputs:
        padded_shapes = output shapes for (images, ground truth boxes, labels)
        padding_values = padding values with dtypes for (images, ground truth boxes, labels)
    """
    padded_shapes = ([None, None, None], [None, None], [None,])
    padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
    return padded_shapes, padding_values

def get_dataset(name, split, data_dir="~/tensorflow_datasets"):
    """Get tensorflow dataset split and info.
    inputs:
        name = name of the dataset, voc/2007, voc/2012, etc.
        split = data split string, should be one of ["train", "validation", "test"]
        data_dir = read/write path for tensorflow datasets

    outputs:
        dataset = tensorflow dataset split
        info = tensorflow dataset info
    """
    assert split in ["train", "train+validation", "validation", "test"]
    dataset, info = tfds.load(name, split=split, data_dir=data_dir, with_info=True)
    return dataset, info

def get_step_size(total_items, batch_size):
    """Get step size for given total item size and batch size.
    inputs:
        total_items = number of total items
        batch_size = number of batch size during training or validation

    outputs:
        step_size = number of step size for model training
    """
    return int(np.ceil(total_items / batch_size))

def get_total_item_size(info, split):
    """Get total item size for given split.
    inputs:
        info = tensorflow dataset info
        split = data split string, should be one of ["train", "validation", "test"]

    outputs:
        total_item_size = number of total items
    """
    assert split in ["train", "train+validation", "validation", "test"]
    if split == "train+validation":
        return info.splits["train"].num_examples + info.splits["validation"].num_examples
    return info.splits[split].num_examples

def get_labels(info):
    """Get label names list.
    inputs:
        info = tensorflow dataset info

    outputs:
        labels = [labels list]
    """
    return info.features["labels"].names

def get_image_data_from_folder(custom_image_path, final_height, final_width):
    """Generating image data like tensorflow dataset format for a given image path.
    This method could be used for custom image predictions.
    inputs:
        custom_image_path = folder of the custom images
        final_height = final image height after resizing
        final_width = final image width after resizing

    outputs:
        image_data = (img, dummy_gt_boxes, dummy_gt_labels)
            img = (1, final_height, final_width, depth)
            dummy_gt_boxes = None
            dummy_gt_labels = None
    """
    image_data = []
    for path, dir, filenames in os.walk(custom_image_path):
        for filename in filenames:
            img_path = os.path.join(path, filename)
            image = Image.open(img_path)
            resized_image = image.resize((final_width, final_height), Image.LANCZOS)
            img = tf.expand_dims(array_from_img(resized_image), 0)
            img = tf.image.convert_image_dtype(img, tf.float32)
            image_data.append((img, None, None))
        break
    return image_data

def preprocessing(image_data, final_height, final_width, augmentation_fn=None):
    """Image resizing operation handled before batch operations.
    inputs:
        image_data = tensorflow dataset image_data
        final_height = final image height after resizing
        final_width = final image width after resizing

    outputs:
        img = (final_height, final_width, channels)
        gt_boxes = (gt_box_size, [y1, x1, y2, x2])
        gt_labels = (gt_box_size)
    """
    img = image_data["image"]
    gt_boxes = image_data["objects"]["bbox"]
    gt_labels = tf.cast(image_data["objects"]["label"] + 1, tf.int32)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = resize_image(img, final_height, final_width)
    if augmentation_fn:
        img, gt_boxes = augmentation_fn(img, gt_boxes)
        img = resize_image(img, final_height, final_width)
    return img, gt_boxes, gt_labels

def resize_image(img, final_height, final_width):
    """Resize image to given height and width values.
    inputs:
        img = (height, width, channels)
        final_height = final image height after resizing
        final_width = final image width after resizing

    outputs:
        resized_img = (final_height, final_width, channels)
    """
    return tf.image.resize(img, (final_height, final_width))

def img_from_array(array):
    """Getting pillow image object from numpy array.
    inputs:
        array = (height, width, channels)

    outputs:
        image = Pillow image object
    """
    return Image.fromarray(array)

def array_from_img(image):
    """Getting numpy array from pillow image object.
    inputs:
        image = Pillow image object

    outputs:
        array = (height, width, channels)
    """
    return np.array(image)

def draw_grid_map(img, grid_map, stride):
    """Drawing grid intersection on given image.
    inputs:
        img = (height, width, channels)
        grid_map = (output_height * output_width, [y_index, x_index, y_index, x_index])
            tiled x, y coordinates
        stride = number of stride

    outputs:
        array = (height, width, channels)
    """
    image = img_from_array(img)
    draw = ImageDraw.Draw(image)
    counter = 0
    for grid in grid_map:
        draw.rectangle((
            grid[0] + stride // 2 - 2,
            grid[1] + stride // 2 - 2,
            grid[2] + stride // 2 + 2,
            grid[3] + stride // 2 + 2), fill=(255, 255, 255, 0))
        counter += 1
    plt.figure()
    plt.imshow(image)
    plt.show()

def draw_bboxes(img, bboxes):
    """Drawing bounding boxes on given image.
    inputs:
        img = (batch_size, height, width, channels)
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
            in normalized form [0, 1]
    """
    colors = tf.cast(np.array([[1, 0, 0, 1]] * 10), dtype=tf.float32)
    img_with_bounding_boxes = tf.image.draw_bounding_boxes(
        img,
        bboxes,
        colors
    )
    plt.figure()
    plt.imshow(img_with_bounding_boxes[0])
    plt.show()

def draw_bboxes_with_labels(img, bboxes, label_indices, probs, labels):
    """Drawing bounding boxes with labels on given image.
    inputs:
        img = (height, width, channels)
        bboxes = (total_bboxes, [y1, x1, y2, x2])
            in normalized form [0, 1]
        label_indices = (total_bboxes)
        probs = (total_bboxes)
        labels = [labels string list]
    """
    colors = []
    for i in range(len(labels)):
        colors.append(tuple(np.random.choice(range(256), size=4)))
    image = tf.keras.preprocessing.image.array_to_img(img)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    denormalized_bboxes = bbox_utils.denormalize_bboxes(bboxes, height, width)
    for index, bbox in enumerate(denormalized_bboxes):
        y1, x1, y2, x2 = np.split(bbox, 4)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        label_index = int(label_indices[index])
        color = colors[label_index]
        label_text = "{0} {1:0.3f}".format(labels[label_index], probs[index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    #
    plt.figure()
    plt.imshow(image)
    plt.show()

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

def handle_gpu_compatibility():
    """Handling of GPU issues for cuDNN initialize error and memory issues."""
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)
