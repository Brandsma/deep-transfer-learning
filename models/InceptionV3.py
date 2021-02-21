import tensorflow as tf
from keras.applications.inception_v3 import preprocess_input
from logger import setup_logger
from tensorflow.keras.applications.inception_v3 import InceptionV3

log = setup_logger(__name__)

# https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4


def build_inception(config, num_classes):
    log.info("Building Inception V3")
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input
    base_model = InceptionV3(include_top=False, weights="imagenet")

    # TODO: Decide on input shape
    inputs = tf.keras.Input(shape=(299, 299, 3))
    x = preprocess_input(x)
    x = base_model(x, training=False)
    #   x = global_average_layer(x)
    #   x = tf.keras.layers.Dropout(0.2)(x)
    # outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def train_inception(train_ds, validation_ds, model, config):
    pass
