import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Dropout
from logger import setup_logger

from models.util import get_optimizer

log = setup_logger(__name__)

# https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4


def build_inception(config, num_classes):
    log.info("Building Inception V3")
    # CONSTANTS
    IMAGE_SHAPE = (299, 299)

    classifier_model = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"
    feature_extractor_layer = hub.KerasLayer(
        classifier_model, input_shape=IMAGE_SHAPE + (3,), trainable=False
    )

    # Add a classification layer, which is a dense layer connected to num_classes nodes
    classification_layer = tf.keras.layers.Dense(num_classes)

    # Build the classifier
    classifier = tf.keras.Sequential()

    classifier.add(feature_extractor_layer)
    if config.with_dropout:
        classifier.add(Dropout(config.dropout_rate))
    classifier.add(classification_layer)

    classifier.summary()

    return classifier


def train_inception(train_ds, validation_ds, model, config):
    log.info("Training Inception V3")

    # We train the model using an optimizer and SparseCategoricalCrossentropy
    model.compile(
        optimizer=get_optimizer(config.optimizer, config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["acc"],
    )

    history = model.fit(train_ds, epochs=config.epochs, validation_data=validation_ds)

    return model, history
