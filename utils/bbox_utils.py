import tensorflow as tf
import numpy as np

def non_max_suppression(pred_bboxes, pred_labels, **kwargs):
    """Applying non maximum suppression.
    Details could be found on tensorflow documentation.
    https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
    inputs:
        pred_bboxes = (batch_size, total_bboxes, total_labels, [y1, x1, y2, x2])
            total_labels should be 1 for binary operations like in rpn
        pred_labels = (batch_size, total_bboxes, total_labels)
        **kwargs = other parameters

    outputs:
        nms_boxes = (batch_size, max_detections, [y1, x1, y2, x2])
        nmsed_scores = (batch_size, max_detections)
        nmsed_classes = (batch_size, max_detections)
        valid_detections = (batch_size)
            Only the top valid_detections[i] entries in nms_boxes[i], nms_scores[i] and nms_class[i] are valid.
            The rest of the entries are zero paddings.
    """
    return tf.image.combined_non_max_suppression(
        pred_bboxes,
        pred_labels,
        **kwargs
    )

def generate_iou_map(bboxes, gt_boxes, transpose_perm=[0, 2, 1]):
    """Calculating intersection over union values for each ground truth boxes in a dynamic manner.
    It is supported from 1d to 3d dimensions for bounding boxes.
    Even if bboxes have different rank from gt_boxes it should be work.
    inputs:
        bboxes = (dynamic_dimension, [y1, x1, y2, x2])
        gt_boxes = (dynamic_dimension, [y1, x1, y2, x2])
        transpose_perm = (transpose_perm_order)
            for 3d gt_boxes => [0, 2, 1]

    outputs:
        iou_map = (dynamic_dimension, total_gt_boxes)
            same rank with the gt_boxes
    """
    gt_rank = tf.rank(gt_boxes)
    gt_expand_axis = gt_rank - 2
    #
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(bboxes, 4, axis=-1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)
    # Calculate bbox and ground truth boxes areas
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=-1)
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1)
    #
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, transpose_perm))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, transpose_perm))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, transpose_perm))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, transpose_perm))
    ### Calculate intersection area
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    ### Calculate union area
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, gt_expand_axis) - intersection_area)
    # Intersection over Union
    return intersection_area / union_area

def get_bboxes_from_deltas(prior_boxes, deltas):
    """Calculating bounding boxes for given bounding box and delta values.
    inputs:
        prior_boxes = (total_bboxes, [y1, x1, y2, x2])
        deltas = (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])

    outputs:
        final_boxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
    """
    all_pbox_width = prior_boxes[..., 3] - prior_boxes[..., 1]
    all_pbox_height = prior_boxes[..., 2] - prior_boxes[..., 0]
    all_pbox_ctr_x = prior_boxes[..., 1] + 0.5 * all_pbox_width
    all_pbox_ctr_y = prior_boxes[..., 0] + 0.5 * all_pbox_height
    #
    all_bbox_width = tf.exp(deltas[..., 3]) * all_pbox_width
    all_bbox_height = tf.exp(deltas[..., 2]) * all_pbox_height
    all_bbox_ctr_x = (deltas[..., 1] * all_pbox_width) + all_pbox_ctr_x
    all_bbox_ctr_y = (deltas[..., 0] * all_pbox_height) + all_pbox_ctr_y
    #
    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_height + y1
    x2 = all_bbox_width + x1
    #
    return tf.stack([y1, x1, y2, x2], axis=-1)

def get_deltas_from_bboxes(bboxes, gt_boxes):
    """Calculating bounding box deltas for given bounding box and ground truth boxes.
    inputs:
        bboxes = (total_bboxes, [y1, x1, y2, x2])
        gt_boxes = (batch_size, total_bboxes, [y1, x1, y2, x2])

    outputs:
        final_deltas = (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])
    """
    bbox_width = bboxes[..., 3] - bboxes[..., 1]
    bbox_height = bboxes[..., 2] - bboxes[..., 0]
    bbox_ctr_x = bboxes[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = bboxes[..., 0] + 0.5 * bbox_height
    #
    gt_width = gt_boxes[..., 3] - gt_boxes[..., 1]
    gt_height = gt_boxes[..., 2] - gt_boxes[..., 0]
    gt_ctr_x = gt_boxes[..., 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[..., 0] + 0.5 * gt_height
    #
    bbox_width = tf.where(tf.equal(bbox_width, 0), 1e-3, bbox_width)
    bbox_height = tf.where(tf.equal(bbox_height, 0), 1e-3, bbox_height)
    delta_x = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width))
    delta_y = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height))
    delta_w = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.math.log(gt_width / bbox_width))
    delta_h = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.math.log(gt_height / bbox_height))
    #
    return tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1)

def get_scale_for_nth_feature_map(k, m=6, scale_min=0.2, scale_max=0.9):
    """Calculating scale value for nth feature map using the given method in the paper.
    inputs:
        k = nth feature map for scale calculation
        m = length of all using feature maps for detections, 6 for ssd300

    outputs:
        scale = calculated scale value for given index
    """
    return scale_min + ((scale_max - scale_min) / (m - 1)) * (k - 1)

def get_height_width_pairs(aspect_ratios, feature_map_index, total_feature_map):
    """Generating height and width pairs for different aspect ratios and feature map shapes.
    inputs:
        aspect_ratios = for all feature map shapes + 1 for ratio 1
        feature_map_index = nth feature maps for scale calculation
        total_feature_map = length of all using feature map for detections, 6 for ssd300

    outputs:
        height_width_pairs = [(height1, width1), ..., (heightN, widthN)]
    """
    current_scale = get_scale_for_nth_feature_map(feature_map_index, m=total_feature_map)
    next_scale = get_scale_for_nth_feature_map(feature_map_index + 1, m=total_feature_map)
    height_width_pairs = []
    for aspect_ratio in aspect_ratios:
        height = current_scale / np.sqrt(aspect_ratio)
        width = current_scale * np.sqrt(aspect_ratio)
        height_width_pairs.append((height, width))
    # 1 extra pair for ratio 1
    height = width = np.sqrt(current_scale * next_scale)
    height_width_pairs.append((height, width))
    return height_width_pairs

def generate_base_prior_boxes(stride, height_width_pairs):
    """Generating top left prior boxes for given stride, height and width pairs of different aspect ratios.
    These prior boxes same with the anchors in Faster-RCNN.
    inputs:
        stride = step size
        height_width_pairs = [(height1, width1), ..., (heightN, widthN)]

    outputs:
        base_prior_boxes = (prior_box_count, [y1, x1, y2, x2])
    """
    center = stride / 2
    base_prior_boxes = []
    for height_width in height_width_pairs:
        height, width = height_width
        x_min = center - width / 2
        y_min = center - height / 2
        x_max = center + width / 2
        y_max = center + height / 2
        base_prior_boxes.append([y_min, x_min, y_max, x_max])
    return np.array(base_prior_boxes, dtype=np.float32)

def generate_prior_boxes(feature_map_shapes, aspect_ratios):
    """Generating top left prior boxes for given stride, height and width pairs of different aspect ratios.
    These prior boxes same with the anchors in Faster-RCNN.
    inputs:
        feature_map_shapes = for all feature map output size
        aspect_ratios = for all feature map shapes + 1 for ratio 1

    outputs:
        prior_boxes = (total_prior_boxes, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
    """
    prior_boxes = []
    for i, feature_map_shape in enumerate(feature_map_shapes):
        prior_box_count = len(aspect_ratios[i]) + 1
        height_width_pairs = get_height_width_pairs(aspect_ratios[i], i+1, len(feature_map_shapes))
        base_prior_boxes = generate_base_prior_boxes(1. / feature_map_shape, height_width_pairs)
        #
        grid_coords = np.arange(0, feature_map_shape)
        #
        grid_x, grid_y = np.meshgrid(grid_coords, grid_coords)
        grid_map = np.vstack((grid_y.ravel(), grid_x.ravel(), grid_y.ravel(), grid_x.ravel())).transpose()
        grid_map = grid_map / feature_map_shape
        #
        output_area = feature_map_shape ** 2
        prior_boxes_for_feature_map = base_prior_boxes.reshape((1, prior_box_count, 4)) + \
                                      grid_map.reshape((1, output_area, 4)).transpose((1, 0, 2))
        prior_boxes_for_feature_map = prior_boxes_for_feature_map.reshape((output_area * prior_box_count, 4)).astype(np.float32)
        prior_boxes.append(prior_boxes_for_feature_map)
    prior_boxes = np.concatenate(prior_boxes, axis=0)
    return np.clip(prior_boxes, 0, 1)

def normalize_bboxes(bboxes, height, width):
    """Normalizing bounding boxes.
    inputs:
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
        height = image height
        width = image width

    outputs:
        normalized_bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
            in normalized form [0, 1]
    """
    y1 = bboxes[..., 0] / height
    x1 = bboxes[..., 1] / width
    y2 = bboxes[..., 2] / height
    x2 = bboxes[..., 3] / width
    return tf.stack([y1, x1, y2, x2], axis=-1)

def denormalize_bboxes(bboxes, height, width):
    """Denormalizing bounding boxes.
    inputs:
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
            in normalized form [0, 1]
        height = image height
        width = image width

    outputs:
        denormalized_bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
    """
    y1 = bboxes[..., 0] * height
    x1 = bboxes[..., 1] * width
    y2 = bboxes[..., 2] * height
    x2 = bboxes[..., 3] * width
    return tf.round(tf.stack([y1, x1, y2, x2], axis=-1))
