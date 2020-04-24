import tensorflow as tf
import numpy as np
import helpers

def get_scale_for_nth_feature_map(k, m=6, scale_min=0.2, scale_max=0.9):
    """Calculating scale value for nth feature map using the given method in the paper.
    inputs:
        aspect_ratios = for all default box shapes + 1 for ratio 1
        feature_map_index = nth feature map for scale calculation
        total_feature_map = length of all using feature map for detections 6 for ssd300

    outputs:
        height_width_pairs = [(height1, width1), ..., (heightN, widthN)]
    """
    return scale_min + ((scale_max - scale_min) / (m - 1)) * (k - 1)

def get_height_width_pairs(aspect_ratios, feature_map_index, total_feature_map):
    """Generating height and width pairs for different aspect ratios and feature map shapes.
    inputs:
        aspect_ratios = for all feature map shapes + 1 for ratio 1
        feature_map_index = nth feature map for scale calculation
        total_feature_map = length of all using feature map for detections 6 for ssd300

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

def generator(dataset, prior_boxes, hyper_params):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
    inputs:
        dataset = tf.data.Dataset, PaddedBatchDataset
        prior_boxes = (total_prior_boxes, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        hyper_params = dictionary

    outputs:
        yield inputs, outputs
    """
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_labels = image_data
            actual_bbox_deltas, actual_labels = calculate_actual_outputs(prior_boxes, gt_boxes, gt_labels, hyper_params)
            yield img, (actual_bbox_deltas, actual_labels)

def calculate_actual_outputs(prior_boxes, gt_boxes, gt_labels, hyper_params):
    """Calculate ssd actual output values.
    Batch operations supported.
    inputs:
        prior_boxes = (total_prior_boxes, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_boxes (batch_size, gt_box_size, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_labels (batch_size, gt_box_size)
        hyper_params = dictionary

    outputs:
        bbox_deltas = (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])
        bbox_labels = (batch_size, total_bboxes, [0,0,...,0])
    """
    batch_size = tf.shape(gt_boxes)[0]
    total_labels = hyper_params["total_labels"]
    iou_threshold = hyper_params["iou_threshold"]
    variances = hyper_params["variances"]
    total_prior_boxes = prior_boxes.shape[0]
    # Calculate iou values between each bboxes and ground truth boxes
    iou_map = helpers.generate_iou_map(prior_boxes, gt_boxes)
    # Get max index value for each row
    max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    #
    pos_cond = tf.greater(merged_iou_map, iou_threshold)
    #
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_cond, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    bbox_deltas = helpers.get_deltas_from_bboxes(prior_boxes, expanded_gt_boxes) / variances
    #
    gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_labels = tf.where(pos_cond, gt_labels_map, tf.zeros_like(gt_labels_map))
    bbox_labels = tf.one_hot(expanded_gt_labels, total_labels)
    #
    return bbox_deltas, bbox_labels
