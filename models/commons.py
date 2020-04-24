import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPool2D, Activation

class HeadWrapper(Layer):
    """Merging all feature maps for detections.
    inputs:
        conv4_3 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            ssd300 conv4_3 shape => (38 x 38 x 4) = 5776
        conv7 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            ssd300 conv7 shape => (19 x 19 x 6) = 2166
        conv8_2 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            ssd300 conv8_2 shape => (10 x 10 x 6) = 600
        conv9_2 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            ssd300 conv9_2 shape => (5 x 5 x 6) = 150
        conv10_2 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            ssd300 conv10_2 shape => (3 x 3 x 4) = 36
        conv11_2 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            ssd300 conv11_2 shape => (1 x 1 x 4) = 4
                                           Total = 8732 default box

    outputs:
        merged_head = (batch_size, total_prior_boxes, last_dimension)
    """

    def __init__(self, last_dimension, **kwargs):
        super(HeadWrapper, self).__init__(**kwargs)
        self.last_dimension = last_dimension

    def get_config(self):
        config = super(HeadWrapper, self).get_config()
        config.update({"last_dimension": self.last_dimension})
        return config

    def call(self, inputs):
        last_dimension = self.last_dimension
        batch_size = tf.shape(inputs[0])[0]
        outputs = []
        for conv_layer in inputs:
            outputs.append(tf.reshape(conv_layer, (batch_size, -1, last_dimension)))
        #
        return tf.concat(outputs, axis=1)

def get_head_from_outputs(hyper_params, outputs):
    """Generating ssd model for hyper params.
    inputs:
        hyper_params = dictionary
        outputs = list of ssd layers output to be used for prediction

    outputs:
        pred_bbox_deltas = merged outputs for bbox delta head
        pred_labels = merged outputs for bbox label head
    """
    total_labels = hyper_params["total_labels"]
    # +1 for ratio 1
    len_aspect_ratios = [len(x) + 1 for x in hyper_params["aspect_ratios"]]
    labels_head = []
    boxes_head = []
    for i, output in enumerate(outputs):
        aspect_ratio = len_aspect_ratios[i]
        labels_head.append(Conv2D(aspect_ratio * total_labels, (3, 3), padding="same", name="{}_conv_label_output".format(i+1))(output))
        boxes_head.append(Conv2D(aspect_ratio * 4, (3, 3), padding="same", name="{}_conv_boxes_output".format(i+1))(output))
    #
    pred_labels = HeadWrapper(total_labels, name="labels_head")(labels_head)
    pred_labels = Activation("softmax", name="conf")(pred_labels)
    #
    pred_bbox_deltas = HeadWrapper(4, name="loc")(boxes_head)
    return pred_bbox_deltas, pred_labels
