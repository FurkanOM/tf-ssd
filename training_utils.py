import tensorflow as tf
import bbox_utils

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
    iou_map = bbox_utils.generate_iou_map(prior_boxes, gt_boxes)
    # Get max index value for each row
    max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    #
    pos_cond = tf.greater(merged_iou_map, iou_threshold)
    #
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_cond, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    bbox_deltas = bbox_utils.get_deltas_from_bboxes(prior_boxes, expanded_gt_boxes) / variances
    #
    gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_labels = tf.where(pos_cond, gt_labels_map, tf.zeros_like(gt_labels_map))
    bbox_labels = tf.one_hot(expanded_gt_labels, total_labels)
    #
    return bbox_deltas, bbox_labels
