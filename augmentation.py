import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import random
import math

def apply(img, gt_boxes, gt_labels):
    # Color operations
    # Randomly change hue, saturation, brightness and contrast of image
    color_methods = [random_brightness, random_contrast, random_hue, random_saturation]
    for color_method in color_methods:
        img = color_method(img)
    # Geometric operations
    # Randomly rotate or flip horizontally image and ground truth boxes
    geometric_methods = [random_rotate, random_flip_horizontally]
    for geometric_method in geometric_methods:
        img, gt_boxes = geometric_method(img, gt_boxes)
    return img, gt_boxes, gt_labels

def random_brightness(img, max_delta=0.12):
    return tf.image.random_brightness(img, max_delta)

def random_contrast(img, lower=0.5, upper=1.5):
    return tf.image.random_contrast(img, lower, upper)

def random_hue(img, max_delta=0.08):
    return tf.image.random_hue(img, max_delta)

def random_saturation(img, lower=0.5, upper=1.5):
    return tf.image.random_saturation(img, lower, upper)

def random_flip_horizontally(img, gt_boxes):
    if random.random() > 0.5:
        img, gt_boxes = flip_horizontally(img, gt_boxes)
    return img, gt_boxes

def flip_horizontally(img, gt_boxes):
    flipped_img = tf.image.flip_left_right(img)
    flipped_gt_boxes = tf.stack([gt_boxes[..., 0],
                                1.0 - gt_boxes[..., 3],
                                gt_boxes[..., 2],
                                1.0 - gt_boxes[..., 1]], -1)
    return flipped_img, flipped_gt_boxes

def get_corners(gt_boxes):
    # Upper left corner
    x1 = gt_boxes[:, 1]
    y1 = gt_boxes[:, 0]
    # Lower right corner
    x3 = gt_boxes[:, 3]
    y3 = gt_boxes[:, 2]
    # Upper right corner
    x2 = x3
    y2 = y1
    # Lower left corner
    x4 = x1
    y4 = y3
    corners = tf.stack([x1, y1, x2, y2, x3, y3, x4, y4], axis=-1)
    return tf.reshape(corners, (-1, 4, 2))

def rotate_bboxes(gt_boxes, angle):
    # center of image 0.5 for height and 0.5 for width in normalized form
    new_origin = tf.constant([[0.5, 0.5]], tf.float32)
    # the rotation matrix for the given angle
    rotation_matrix = tf.constant([[math.cos(angle), -math.sin(angle)],
                                   [math.sin(angle), math.cos(angle)]], tf.float32)
    # we translate all corners to new origin
    corners = get_corners(gt_boxes) - new_origin
    # rotate all corners by the given angle
    rotated_corners = tf.matmul(corners, rotation_matrix, transpose_b=True) + new_origin
    # select min and max coordinates of corners for new bounding boxes
    x_min = tf.reduce_min(rotated_corners[..., 0], 1)
    x_max = tf.reduce_max(rotated_corners[..., 0], 1)
    y_min = tf.reduce_min(rotated_corners[..., 1], 1)
    y_max = tf.reduce_max(rotated_corners[..., 1], 1)
    #
    rotated_gt_boxes = tf.stack([y_min, x_min, y_max, x_max], axis=-1)
    rotated_gt_boxes = tf.clip_by_value(rotated_gt_boxes, 0, 1)
    #
    return rotated_gt_boxes, rotated_corners

def random_rotate(img, gt_boxes):
    if random.random() > 0.5:
        img, gt_boxes = rotate(img, gt_boxes)
    return img, gt_boxes

def rotate(img, gt_boxes):
    max_rotation_degree = 15
    random_degree = random.uniform(-max_rotation_degree, max_rotation_degree)
    random_angle = random_degree * math.pi / 180
    rotated_img = tfa.image.rotate(img, angles=-random_angle)
    rotated_gt_boxes, rotated_corners = rotate_bboxes(gt_boxes, random_angle)
    return rotated_img, rotated_gt_boxes

def draw_rotated_bboxes(img, original_gt_boxes, rotated_gt_boxes, rotated_corners):
    image = tf.keras.preprocessing.image.array_to_img(img)
    width, height = image.size
    #
    denormalized_original_gt_boxes = tf.stack([
        original_gt_boxes[..., 1] * width,
        original_gt_boxes[..., 0] * height,
        original_gt_boxes[..., 3] * width,
        original_gt_boxes[..., 2] * height], axis=1)
    denormalized_rotated_gt_boxes = tf.stack([
        rotated_gt_boxes[..., 1] * width,
        rotated_gt_boxes[..., 0] * height,
        rotated_gt_boxes[..., 3] * width,
        rotated_gt_boxes[..., 2] * height], axis=1)
    denormalized_rotated_corners = tf.stack([
        rotated_corners[..., 0] * width,
        rotated_corners[..., 1] * height], axis=-1)
    #
    draw = ImageDraw.Draw(image)
    draw.polygon(denormalized_rotated_corners[1], outline=(255, 0, 0, 1))
    draw.rectangle(denormalized_rotated_gt_boxes[1], outline=(0, 255, 0, 1), width=2)
    draw.rectangle(denormalized_original_gt_boxes[1], outline=(0, 0, 255, 1), width=1)
    #
    plt.figure()
    plt.imshow(image)
    plt.show()
