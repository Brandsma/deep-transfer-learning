import tensorflow as tf


def get_optimizer(optimizer_type, learning_rate):
    optimizers = {
        "adam": tf.keras.optimizers.Adam,
        "adamax": tf.keras.optimizers.Adamax,
        "RMS": tf.keras.optimizers.SGD,
    }
    return optimizers[optimizer_type](learning_rate=learning_rate)
