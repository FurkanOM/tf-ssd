import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from .commons import get_head_from_outputs

class L2Normalization(Layer):
    """Normalizing different scale features for fusion.
    paper: https://arxiv.org/abs/1506.04579
    inputs:
        feature_map = (batch_size, feature_map_height, feature_map_width, depth)

    outputs:
        normalized_feature_map = (batch_size, feature_map_height, feature_map_width, depth)
    """
    def __init__(self, scale_factor, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def get_config(self):
        config = super(L2Normalization, self).get_config()
        config.update({"scale_factor": self.scale_factor})
        return config

    def build(self, input_shape):
        # Network need to learn scale factor for each channel
        init_scale_factor = tf.fill((input_shape[-1],), float(self.scale_factor))
        self.scale = tf.Variable(init_scale_factor, trainable=True)

    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=-1) * self.scale

def get_model(hyper_params):
    """Generating ssd model for hyper params.
    inputs:
        hyper_params = dictionary

    outputs:
        ssd_model = tf.keras.model
    """
    # Initial scale factor 20 in the paper.
    # Even if this scale factor could cause loss value to be NaN in some of the cases,
    # it was decided to remain the same after some tests.
    scale_factor = 20.0
    reg_factor = 5e-4
    total_labels = hyper_params["total_labels"]
    # +1 for ratio 1
    len_aspect_ratios = [len(x) + 1 for x in hyper_params["aspect_ratios"]]
    #
    input = Input(shape=(None, None, 3), name="input")
    # conv1 block
    conv1_1 = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv1_1")(input)
    conv1_2 = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv1_2")(conv1_1)
    pool1 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1_2)
    # conv2 block
    conv2_1 = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv2_1")(pool1)
    conv2_2 = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv2_2")(conv2_1)
    pool2 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2_2)
    # conv3 block
    conv3_1 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv3_1")(pool2)
    conv3_2 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv3_2")(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv3_3")(conv3_2)
    pool3 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3_3)
    # conv4 block
    conv4_1 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv4_1")(pool3)
    conv4_2 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv4_2")(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv4_3")(conv4_2)
    pool4 = MaxPool2D((2, 2), strides=(2, 2), padding="same", name="pool4")(conv4_3)
    # conv5 block
    conv5_1 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv5_1")(pool4)
    conv5_2 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv5_2")(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv5_3")(conv5_2)
    pool5 = MaxPool2D((3, 3), strides=(1, 1), padding="same", name="pool5")(conv5_3)
    # conv6 and conv7 converted from fc6 and fc7 and remove dropouts
    # These layers coming from modified vgg16 model
    # https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6
    conv6 = Conv2D(1024, (3, 3), dilation_rate=6, padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv6")(pool5)
    conv7 = Conv2D(1024, (1, 1), strides=(1, 1), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv7")(conv6)
    ############################ Extra Feature Layers Start ############################
    # conv8 block <=> conv6 block in paper caffe implementation
    conv8_1 = Conv2D(256, (1, 1), strides=(1, 1), padding="valid", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv8_1")(conv7)
    conv8_2 = Conv2D(512, (3, 3), strides=(2, 2), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv8_2")(conv8_1)
    # conv9 block <=> conv7 block in paper caffe implementation
    conv9_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv9_1")(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv9_2")(conv9_1)
    # conv10 block <=> conv8 block in paper caffe implementation
    conv10_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv10_1")(conv9_2)
    conv10_2 = Conv2D(256, (3, 3), strides=(1, 1), padding="valid", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv10_2")(conv10_1)
    # conv11 block <=> conv9 block in paper caffe implementation
    conv11_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv11_1")(conv10_2)
    conv11_2 = Conv2D(256, (3, 3), strides=(1, 1), padding="valid", activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(reg_factor), name="conv11_2")(conv11_1)
    ############################ Extra Feature Layers End ############################
    # l2 normalization for each location in the feature map
    conv4_3_norm = L2Normalization(scale_factor)(conv4_3)
    #
    pred_bbox_deltas, pred_labels = get_head_from_outputs(hyper_params, [conv4_3_norm, conv7, conv8_2, conv9_2, conv10_2, conv11_2])
    #
    return Model(inputs=input, outputs=[pred_bbox_deltas, pred_labels])

def init_model(model):
    """Initializing model with dummy data for load weights with optimizer state and also graph construction.
    inputs:
        model = tf.keras.model

    """
    model(tf.random.uniform((1, 512, 512, 3)))
