import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from .header import get_head_from_outputs

def get_model(hyper_params):
    """Generating ssd model for hyper params.
    inputs:
        hyper_params = dictionary

    outputs:
        ssd_model = tf.keras.model
    """
    img_size = hyper_params["img_size"]
    base_model = MobileNetV2(include_top=False, input_shape=(img_size, img_size, 3))
    input = base_model.input
    first_conv = base_model.get_layer("block_13_expand_relu").output
    second_conv = base_model.output
    #
    ############################ Extra Feature Layers Start ############################
    extra1_1 = Conv2D(256, (1, 1), strides=(1, 1), padding="valid", activation="relu", name="extra1_1")(second_conv)
    extra1_2 = Conv2D(512, (3, 3), strides=(2, 2), padding="same", activation="relu", name="extra1_2")(extra1_1)
    #
    extra2_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", name="extra2_1")(extra1_2)
    extra2_2 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu", name="extra2_2")(extra2_1)
    #
    extra3_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", name="extra3_1")(extra2_2)
    extra3_2 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu", name="extra3_2")(extra3_1)
    #
    extra4_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", name="extra4_1")(extra3_2)
    extra4_2 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu", name="extra4_2")(extra4_1)
    ############################ Extra Feature Layers End ############################
    pred_deltas, pred_labels = get_head_from_outputs(hyper_params, [first_conv, second_conv, extra1_2, extra2_2, extra3_2, extra4_2])
    return Model(inputs=input, outputs=[pred_deltas, pred_labels])

def init_model(model):
    """Initializing model with dummy data for load weights with optimizer state and also graph construction.
    inputs:
        model = tf.keras.model

    """
    model(tf.random.uniform((1, 300, 300, 3)))
