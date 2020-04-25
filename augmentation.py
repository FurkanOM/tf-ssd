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
    for augmentation_method in color_methods + geometric_methods:
        img, gt_boxes = randomly_apply_operation(augmentation_method, img, gt_boxes)
    #
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

def get_valid_patch_randomly(patches, valid_cond, center_cond):
    """Selecting one valid patch and center conditions of this patch randomly.
    inputs:
        patches = (number_of_valid_patches, [y1, x1, y2, x2])
        valid_cond = (number_of_valid_patches, [ground_truth_object_count_bool])
        center_cond = (number_of_valid_patches, [ground_truth_object_count_bool])
    outputs:
        random_patch = ([y1, x1, y2, x2])
        center_cond_for_patch = ([ground_truth_object_count_bool])
    """
    valid_cond = tf.reduce_any(valid_cond, axis=-1)
    valid_indices = tf.where(valid_cond)
    random_index = tf.random.uniform((), minval=0, maxval=tf.shape(valid_indices)[0], dtype=tf.int32)
    random_index = valid_indices[random_index, 0]
    random_patch = patches[random_index]
    center_cond_for_patch = center_cond[random_index]
    return random_patch, center_cond_for_patch

def select_and_apply_patch(img, gt_boxes, patches, valid_cond, center_cond):
    """Selecting randomly one valid patch and adjusting image and ground truth objects to this patch.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        patches = (number_of_valid_patches, [y1, x1, y2, x2])
        valid_cond = (number_of_valid_patches, [ground_truth_object_count_bool])
        center_cond = (number_of_valid_patches, [ground_truth_object_count_bool])
    outputs:
        modified_img = (final_height, final_width, depth)
        modified_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        height = final_height
        width = final_width
    """
    random_patch, center_in_patch_condition = get_valid_patch_randomly(patches, valid_cond, center_cond)
    #
    height = random_patch[2] - random_patch[0]
    width = random_patch[3] - random_patch[1]
    gt_boxes = update_bboxes_for_patch(random_patch, gt_boxes)
    gt_boxes = tf.where(tf.expand_dims(center_in_patch_condition, 1), gt_boxes, tf.zeros_like(gt_boxes))
    random_patch = tf.cast(random_patch, tf.int32)
    img = tf.image.crop_to_bounding_box(img, random_patch[0], random_patch[1], random_patch[2] - random_patch[0], random_patch[3] - random_patch[1])
    return img, gt_boxes, height, width

def get_centers_of_bboxes(bboxes):
    """Calculating centers of the given boxes.
    inputs:
        bboxes = (total_bbox_count, [y1, x1, y2, x2])
    outputs:
        center_x = (total_bbox_count, center_x)
        center_y = (total_bbox_count, center_y)
    """
    width = bboxes[..., 3] - bboxes[..., 1]
    height = bboxes[..., 2] - bboxes[..., 0]
    center_x = bboxes[..., 1] + width / 2
    center_y = bboxes[..., 0] + height / 2
    return center_x, center_y

def get_center_in_patch_condition(patch, gt_boxes):
    """Determine whether the center points of the given
    ground truth objects are in the given patch.
    inputs:
        patch = ([y1, x1, y2, x2])
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        center_in_patch = ([ground_truth_object_count_bool])
    """
    gt_center_x, gt_center_y = get_centers_of_bboxes(gt_boxes)
    patch_y1, patch_x1, patch_y2, patch_x2 = tf.split(patch, 4, axis=-1)
    center_x_in_patch = tf.logical_and(tf.greater(gt_center_x, patch_x1), tf.less(gt_center_x, patch_x2))
    center_y_in_patch = tf.logical_and(tf.greater(gt_center_y, patch_y1), tf.less(gt_center_y, patch_y2))
    center_in_patch = tf.logical_and(center_x_in_patch, center_y_in_patch)
    return center_in_patch

def get_random_min_overlap():
    """Generating random minimum overlap value.
    outputs:
        min_overlap = random minimum overlap value 0d tensor
    """
    overlaps = tf.constant([0.1, 0.3, 0.5, 0.7, 0.9], dtype=tf.float32)
    i = tf.random.uniform((), minval=0, maxval=tf.shape(overlaps)[0], dtype=tf.int32)
    return overlaps[i]

def update_bboxes_for_patch(patch, gt_boxes):
    """Updating the coordinates of ground truth objects according to the new patch.
    inputs:
        patch = ([y1, x1, y2, x2])
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        modified_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    y1 = gt_boxes[..., 0] - patch[0]
    x1 = gt_boxes[..., 1] - patch[1]
    y2 = gt_boxes[..., 2] - patch[0]
    x2 = gt_boxes[..., 3] - patch[1]
    return tf.stack([y1, x1, y2, x2], -1)

def generate_random_patches(height, width):
    """Generating approximately 100 valid patches according to the min and max aspect ratios.
    inputs:
        height = height of the image
        width = width of the image
    outputs:
        patches = (number_of_valid_patches, [y1, x1, y2, x2])
    """
    min_aspect_ratio = tf.constant(0.5, dtype=tf.float32)
    max_aspect_ratio = tf.constant(2.0, dtype=tf.float32)
    coords = tf.random.uniform((1000, 4), minval=0., maxval=1., dtype=tf.float32)
    coords = tf.round(coords * [height, width, height, width])
    h = coords[..., 2] - coords[..., 0]
    w = coords[..., 3] - coords[..., 1]
    hw_ratio = h / w
    pos_cond = tf.logical_and(tf.greater(h, 0.0), tf.greater(w, 0.0))
    aspect_ratio_cond = tf.logical_and(tf.greater(hw_ratio, min_aspect_ratio), tf.less(hw_ratio, max_aspect_ratio))
    valid_cond = tf.logical_and(pos_cond, aspect_ratio_cond)
    return coords[valid_cond]

def expand_image(img, denormalized_gt_boxes, height, width):
    """Randomly expanding image and adjusting ground truth object coordinates.
    inputs:
        img = (height, width, depth)
        denormalized_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        height = height of the image
        width = width of the image
    outputs:
        img = (final_height, final_width, depth)
        modified_denormalized_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
        final_height = final height of the image
        final_width = final width of the image
    """
    expansion_ratio = tf.random.uniform((), minval=1, maxval=4, dtype=tf.float32)
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
    img_shape = tf.shape(img)
    height, width = tf.cast(img_shape[0], tf.float32), tf.cast(img_shape[1], tf.float32)
    # Denormalizing bounding boxes for further operations
    denormalized_gt_boxes = bbox_utils.denormalize_bboxes(gt_boxes, height, width)
    # Randomly expand image and adjust bounding boxes
    img, denormalized_gt_boxes, height, width = randomly_apply_operation(expand_image, img, denormalized_gt_boxes, height, width)
    # Generate random patches
    patches = generate_random_patches(height, width)
    # Calculate jaccard/iou value for each bounding box
    iou_map = bbox_utils.generate_iou_map(patches, denormalized_gt_boxes, transpose_perm=[1, 0])
    # Check each ground truth box center in the generated patch and return a boolean condition list
    center_in_patch_condition = get_center_in_patch_condition(patches, denormalized_gt_boxes)
    # Get random minimum overlap value
    min_overlap = get_random_min_overlap()
    # Check and merge center condition and minimum intersection condition
    valid_patch_condition = tf.logical_and(center_in_patch_condition, tf.greater(iou_map, min_overlap))
    # Check at least one valid patch then apply patch
    img, denormalized_gt_boxes, height, width = tf.cond(tf.reduce_any(valid_patch_condition),
        lambda: select_and_apply_patch(img, denormalized_gt_boxes, patches, valid_patch_condition, center_in_patch_condition),
        lambda: (img, denormalized_gt_boxes, height, width)
    )
    # Finally normalized ground truth boxes
    gt_boxes = bbox_utils.normalize_bboxes(denormalized_gt_boxes, height, width)
    gt_boxes = tf.clip_by_value(gt_boxes, 0, 1)
    #
    return img, gt_boxes
