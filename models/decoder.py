import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPool2D
from tensorflow.keras.models import Model
from utils import bbox_utils

class SSDDecoder(Layer):
    """Generating bounding boxes and labels from ssd predictions.
    First calculating the boxes from predicted deltas and label probs.
    Then applied non max suppression and selecting top_n boxes by scores.
    inputs:
        pred_deltas = (batch_size, total_prior_boxes, [delta_y, delta_x, delta_h, delta_w])
        pred_label_probs = (batch_size, total_prior_boxes, [0,0,...,0])
    outputs:
        pred_bboxes = (batch_size, top_n, [y1, x1, y2, x2])
        pred_labels = (batch_size, top_n)
            1 to total label number
        pred_scores = (batch_size, top_n)
    """
    def __init__(self, prior_boxes, variances, max_total_size=200, score_threshold=0.5, **kwargs):
        super(SSDDecoder, self).__init__(**kwargs)
        self.prior_boxes = prior_boxes
        self.variances = variances
        self.max_total_size = max_total_size
        self.score_threshold = score_threshold

    def get_config(self):
        config = super(SSDDecoder, self).get_config()
        config.update({
            "prior_boxes": self.prior_boxes.numpy(),
            "variances": self.variances,
            "max_total_size": self.max_total_size,
            "score_threshold": self.score_threshold
        })
        return config

    def call(self, inputs):
        pred_deltas = inputs[0]
        pred_label_probs = inputs[1]
        batch_size = tf.shape(pred_deltas)[0]
        #
        pred_deltas *= self.variances
        pred_bboxes = bbox_utils.get_bboxes_from_deltas(self.prior_boxes, pred_deltas)
        pred_bboxes = tf.clip_by_value(pred_bboxes, 0, 1)
        #
        pred_labels_map = tf.expand_dims(tf.argmax(pred_label_probs, -1), -1)
        pred_labels = tf.where(tf.not_equal(pred_labels_map, 0), pred_label_probs, tf.zeros_like(pred_label_probs))
        # Reshape bboxes for non max suppression
        pred_bboxes = tf.reshape(pred_bboxes, (batch_size, -1, 1, 4))
        #
        final_bboxes, final_scores, final_labels, _ = bbox_utils.non_max_suppression(
                                                                    pred_bboxes, pred_labels,
                                                                    max_output_size_per_class=self.max_total_size,
                                                                    max_total_size=self.max_total_size,
                                                                    score_threshold=self.score_threshold)
        #
        return final_bboxes, final_labels, final_scores

def get_decoder_model(base_model, prior_boxes, hyper_params):
    """Decoding ssd predictions to valid bounding boxes and labels.
    inputs:
        base_model = tf.keras.model, base ssd model
        prior_boxes = (total_prior_boxes, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        hyper_params = dictionary

    outputs:
        ssd_decoder_model = tf.keras.model
    """
    bboxes, classes, scores = SSDDecoder(prior_boxes, hyper_params["variances"])(base_model.output)
    return Model(inputs=base_model.input, outputs=[bboxes, classes, scores])
