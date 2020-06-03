import tensorflow as tf
from utils import bbox_utils

def apply(img, gt_boxes):
    """Randomly applying data augmentation methods to image and ground truth boxes.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
            in normalized form [0, 1]
    outputs:
        modified_img = (final_height, final_width, depth)
        modified_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
            in normalized form [0, 1]
    """
    # Color operations
    # Randomly change hue, saturation, brightness and contrast of image
    color_methods = [random_brightness, random_contrast, random_hue, random_saturation]
    # Geometric operations
    # Randomly sample a patch and flip horizontally image and ground truth boxes
    geometric_methods = [patch, flip_horizontally]
    #
    for augmentation_method in geometric_methods + color_methods:
        img, gt_boxes = randomly_apply_operation(augmentation_method, img, gt_boxes)
    #
    img = tf.clip_by_value(img, 0, 1)
    return img, gt_boxes

def get_random_bool():
    """Generating random boolean.
    outputs:
        random boolean 0d tensor
    """
    return tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5)

def randomly_apply_operation(operation, img, gt_boxes, *args):
    """Randomly applying given method to image and ground truth boxes.
    inputs:
        operation = callable method
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        modified_or_not_img = (final_height, final_width, depth)
        modified_or_not_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    return tf.cond(
        get_random_bool(),
        lambda: operation(img, gt_boxes, *args),
        lambda: (img, gt_boxes, *args)
    )

def random_brightness(img, gt_boxes, max_delta=0.12):
    """Randomly change brightness of the image.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        modified_img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    return tf.image.random_brightness(img, max_delta), gt_boxes

def random_contrast(img, gt_boxes, lower=0.5, upper=1.5):
    """Randomly change contrast of the image.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        modified_img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    return tf.image.random_contrast(img, lower, upper), gt_boxes

def random_hue(img, gt_boxes, max_delta=0.08):
    """Randomly change hue of the image.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        modified_img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    return tf.image.random_hue(img, max_delta), gt_boxes

def random_saturation(img, gt_boxes, lower=0.5, upper=1.5):
    """Randomly change saturation of the image.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        modified_img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    return tf.image.random_saturation(img, lower, upper), gt_boxes

def flip_horizontally(img, gt_boxes):
    """Flip image horizontally and adjust the ground truth boxes.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        modified_img = (height, width, depth)
        modified_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    flipped_img = tf.image.flip_left_right(img)
    flipped_gt_boxes = tf.stack([gt_boxes[..., 0],
                                1.0 - gt_boxes[..., 3],
                                gt_boxes[..., 2],
                                1.0 - gt_boxes[..., 1]], -1)
    return flipped_img, flipped_gt_boxes

##############################################################################
## Sample patch start
##############################################################################

def get_random_min_overlap():
    """Generating random minimum overlap value.
    outputs:
        min_overlap = random minimum overlap value 0d tensor
    """
    overlaps = tf.constant([0.1, 0.3, 0.5, 0.7, 0.9], dtype=tf.float32)
    i = tf.random.uniform((), minval=0, maxval=tf.shape(overlaps)[0], dtype=tf.int32)
    return overlaps[i]

def expand_image(img, gt_boxes):
    """Randomly expanding image and adjusting ground truth object coordinates.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        height = height of the image
        width = width of the image
    outputs:
        img = (final_height, final_width, depth)
        modified_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        final_height = final height of the image
        final_width = final width of the image
    """
    img_shape = tf.cast(tf.shape(img), dtype=tf.float32)
    height, width = img_shape[0], img_shape[1]
    expansion_ratio = tf.random.uniform((), minval=1, maxval=4, dtype=tf.float32)
    final_height, final_width = tf.round(height * expansion_ratio), tf.round(width * expansion_ratio)
    pad_left = tf.round(tf.random.uniform((), minval=0, maxval=final_width - width, dtype=tf.float32))
    pad_top = tf.round(tf.random.uniform((), minval=0, maxval=final_height - height, dtype=tf.float32))
    pad_right = final_width - (width + pad_left)
    pad_bottom = final_height - (height + pad_top)
    #
    mean, _ = tf.nn.moments(img, [0, 1])
    expanded_image = tf.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), constant_values=-1)
    expanded_image = tf.where(expanded_image == -1, mean, expanded_image)
    #
    min_max = tf.stack([-pad_top, -pad_left, pad_bottom+height, pad_right+width], -1) / [height, width, height, width]
    modified_gt_boxes = bbox_utils.renormalize_bboxes_with_min_max(gt_boxes, min_max)
    #
    return expanded_image, modified_gt_boxes

def patch(img, gt_boxes):
    """Generating random patch and adjusting image and ground truth objects to this patch.
    After this operation some of the ground truth boxes / objects could be removed from the image.
    However, these objects are not excluded from the output, only the coordinates are changed as zero.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
            in normalized form [0, 1]
    outputs:
        modified_img = (final_height, final_width, depth)
        modified_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
            in normalized form [0, 1]
    """
    # Randomly expand image and adjust bounding boxes
    img, gt_boxes = randomly_apply_operation(expand_image, img, gt_boxes)
    # Get random minimum overlap value
    min_overlap = get_random_min_overlap()
    #
    begin, size, new_limits = tf.image.sample_distorted_bounding_box(
        tf.shape(img),
        bounding_boxes=tf.expand_dims(gt_boxes, 0),
        aspect_ratio_range=[0.5, 2.0],
        min_object_covered=min_overlap)
    #
    img = tf.slice(img, begin, size)
    gt_boxes = bbox_utils.renormalize_bboxes_with_min_max(gt_boxes, new_limits[0, 0])
    #
    return img, gt_boxes
