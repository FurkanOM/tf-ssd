import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math
import helpers

def apply(img, gt_boxes, gt_labels):
    # Color operations
    # Randomly change hue, saturation, brightness and contrast of image
    color_methods = [random_brightness, random_contrast, random_hue, random_saturation]
    # Geometric operations
    # Randomly sample a patch, rotate, flip horizontally image and ground truth boxes
    geometric_methods = [patch, rotate, flip_horizontally]
    #
    for augmentation_method in color_methods + geometric_methods:
        img, gt_boxes = randomly_apply_operation(augmentation_method, img, gt_boxes)
    #
    return img, gt_boxes, gt_labels

def get_random_bool():
    return tf.random.uniform((), dtype=tf.float32) > 0.5

def randomly_apply_operation(operation, img, gt_boxes):
    return tf.cond(
        get_random_bool(),
        lambda: operation(img, gt_boxes),
        lambda: (img, gt_boxes)
    )

def random_brightness(img, gt_boxes, max_delta=0.12):
    return tf.image.random_brightness(img, max_delta), gt_boxes

def random_contrast(img, gt_boxes, lower=0.5, upper=1.5):
    return tf.image.random_contrast(img, lower, upper), gt_boxes

def random_hue(img, gt_boxes, max_delta=0.08):
    return tf.image.random_hue(img, max_delta), gt_boxes

def random_saturation(img, gt_boxes, lower=0.5, upper=1.5):
    return tf.image.random_saturation(img, lower, upper), gt_boxes

def flip_horizontally(img, gt_boxes):
    flipped_img = tf.image.flip_left_right(img)
    flipped_gt_boxes = tf.stack([gt_boxes[..., 0],
                                1.0 - gt_boxes[..., 3],
                                gt_boxes[..., 2],
                                1.0 - gt_boxes[..., 1]], -1)
    return flipped_img, flipped_gt_boxes

##############################################################################
## Sample patch start
##############################################################################

def generate_random_height_width(height, width):
    random_height = tf.random.uniform((), minval=height*0.1, maxval=height, dtype=tf.float32)
    random_width = tf.random.uniform((), minval=width*0.1, maxval=width, dtype=tf.float32)
    return random_height, random_width

def generate_random_patch(height, width):
    min_aspect_ratio = 0.5
    max_aspect_ratio = 2
    cond = lambda h, w: tf.logical_or(h / w < min_aspect_ratio, h / w > max_aspect_ratio)
    body = lambda h, w: generate_random_height_width(height, width)
    random_height, random_width = tf.while_loop(cond, body, [0.0, 1.0])
    random_top = tf.random.uniform((), minval=0, maxval=height-random_height, dtype=tf.float32)
    random_left = tf.random.uniform((), minval=0, maxval=width-random_width, dtype=tf.float32)
    return tf.round([random_top, random_left, random_top+random_height, random_left+random_width])

def get_centers_of_bboxes(bboxes):
    width = bboxes[..., 3] - bboxes[..., 1]
    height = bboxes[..., 2] - bboxes[..., 0]
    center_x = bboxes[..., 1] + width / 2
    center_y = bboxes[..., 0] + height / 2
    return center_x, center_y

def get_center_in_patch_condition(patch, gt_boxes):
    gt_center_x, gt_center_y = get_centers_of_bboxes(gt_boxes)
    patch_y1, patch_x1, patch_y2, patch_x2 = tf.split(patch, 4, axis=-1)
    center_x_in_patch = tf.logical_and(gt_center_x >= patch_x1, gt_center_x <= patch_x2)
    center_y_in_patch = tf.logical_and(gt_center_y >= patch_y1, gt_center_y <= patch_y2)
    center_in_patch = tf.logical_and(center_x_in_patch, center_y_in_patch)
    return center_in_patch

def get_random_valid_patch(img, gt_boxes, height, width, counter):
    counter = tf.add(counter, 1)
    # Get random minimum overlap value
    min_overlap = get_random_min_overlap()
    # Generating random patch using image height and width values
    random_patch = generate_random_patch(height, width)
    # Calculate jaccard/iou value for each bounding box
    iou_map = helpers.generate_iou_map(random_patch, gt_boxes, transpose_perm=[1, 0])
    # Check each ground truth box center in the generated patch and return a boolean condition list
    center_in_patch_condition = get_center_in_patch_condition(random_patch, gt_boxes)
    # Check and merge center condition and minimum intersection condition
    valid_patch_condition = tf.logical_and(center_in_patch_condition, iou_map > min_overlap)
    # Check at least one valid ground truth box in new patch
    has_valid_patch = tf.reduce_any(valid_patch_condition)
    #
    if has_valid_patch:
        height = random_patch[2] - random_patch[0]
        width = random_patch[3] - random_patch[1]
        gt_boxes = update_bboxes_for_patch(random_patch, gt_boxes)
        gt_boxes = tf.where(tf.expand_dims(center_in_patch_condition, 1), gt_boxes, tf.zeros_like(gt_boxes))
        random_patch = tf.cast(random_patch, tf.int32)
        img = tf.image.crop_to_bounding_box(img, random_patch[0], random_patch[1], random_patch[2] - random_patch[0], random_patch[3] - random_patch[1])
    #
    return has_valid_patch, counter, (img, gt_boxes, height, width)

def get_random_min_overlap():
    overlaps = tf.constant([0.1, 0.3, 0.5, 0.7, 0.9], dtype=tf.float32)
    i = tf.random.uniform((), minval=0, maxval=tf.shape(overlaps)[0], dtype=tf.int32)
    return overlaps[i]

def update_bboxes_for_patch(patch, gt_boxes):
    y1 = gt_boxes[..., 0] - patch[0]
    x1 = gt_boxes[..., 1] - patch[1]
    y2 = gt_boxes[..., 2] - patch[0]
    x2 = gt_boxes[..., 3] - patch[1]
    return tf.stack([y1, x1, y2, x2], -1)

def expand_image(img, denormalized_gt_boxes, height, width):
    expansion_ratio = tf.random.uniform((), minval=1.5, maxval=3, dtype=tf.float32)
    final_height, final_width = tf.round(height * expansion_ratio), tf.round(width * expansion_ratio)
    random_left = tf.round(tf.random.uniform((), minval=0, maxval=final_width - width, dtype=tf.float32))
    random_top = tf.round(tf.random.uniform((), minval=0, maxval=final_height - height, dtype=tf.float32))
    expanded_image = tf.image.pad_to_bounding_box(
        img,
        tf.cast(random_top, tf.int32),
        tf.cast(random_left, tf.int32),
        tf.cast(final_height, tf.int32),
        tf.cast(final_width, tf.int32),
    )
    y1 = denormalized_gt_boxes[..., 0] + random_top
    x1 = denormalized_gt_boxes[..., 1] + random_left
    y2 = denormalized_gt_boxes[..., 2] + random_top
    x2 = denormalized_gt_boxes[..., 3] + random_left
    denormalized_gt_boxes = tf.stack([y1, x1, y2, x2], axis=-1)
    return expanded_image, denormalized_gt_boxes, final_height, final_width

def patch(img, gt_boxes):
    img_shape = tf.shape(img)
    height, width = tf.cast(img_shape[0], tf.float32), tf.cast(img_shape[1], tf.float32)
    # Denormalizing bounding boxes for further operations
    denormalized_gt_boxes = helpers.denormalize_bboxes(gt_boxes, height, width)
    # Randomly expand image and adjust bounding boxes
    img, denormalized_gt_boxes, height, width = tf.cond(
        get_random_bool(),
        lambda: expand_image(img, denormalized_gt_boxes, height, width),
        lambda: (img, denormalized_gt_boxes, height, width)
    )
    # while loop start
    cond = lambda has_valid_patch, counter, data: tf.logical_and(tf.logical_not(has_valid_patch), tf.less(counter, 10))
    body = lambda has_valid_patch, counter, data: get_random_valid_patch(img, denormalized_gt_boxes, height, width, counter)
    _,_, (img, denormalized_gt_boxes, height, width) = tf.while_loop(cond, body, [tf.constant(False, tf.bool), tf.constant(0, tf.int32), (img, denormalized_gt_boxes, height, width)])
    # while loop end
    gt_boxes = helpers.normalize_bboxes(denormalized_gt_boxes, height, width)
    gt_boxes = tf.clip_by_value(gt_boxes, 0, 1)
    return img, gt_boxes

##############################################################################
## Sample patch end
##############################################################################

##############################################################################
## Rotate start
##############################################################################
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
    rotation_matrix = tf.stack([[tf.cos(angle), -tf.sin(angle)],
                                [tf.sin(angle), tf.cos(angle)]])
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

def rotate(img, gt_boxes):
    max_rotation_degree = 15
    random_degree = tf.random.uniform((), minval=-max_rotation_degree, maxval=max_rotation_degree, dtype=tf.float32)
    random_angle = random_degree * math.pi / 180
    rotated_img = tfa.image.rotate(img, angles=-random_angle)
    rotated_gt_boxes, rotated_corners = rotate_bboxes(gt_boxes, random_angle)
    return rotated_img, rotated_gt_boxes

##############################################################################
## Rotate end
##############################################################################

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
