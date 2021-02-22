import tensorflow as tf
from logger import setup_logger
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential

from models.util import get_optimizer

log = setup_logger(__name__)

# https://www.tensorflow.org/tutorials/images/transfer_learning


def build_vgg19(config, num_classes):
    log.info("Building VGG19")

    feature_extractor_layer = VGG19(
        input_shape=(224, 224, 3), weights="imagenet", include_top=False
    )
    feature_extractor_layer.trainable = False

    # Build a classification layer
    classifier = Sequential()

    classifier.add(feature_extractor_layer)
    classifier.add(tf.keras.layers.GlobalAveragePooling2D())

    if config.with_dropout:
        classifier.add(Dropout(config.dropout_rate))

    classifier.add(tf.keras.layers.Dense(num_classes))

    return classifier


def train_vgg19(train_ds, validation_ds, model, config):
    log.info("Training VGG19")

    # We train the model using an optimizer and SparseCategoricalCrossentropy
    model.compile(
        optimizer=get_optimizer(config.optimizer, config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["acc"],
    )

    history = model.fit(train_ds, epochs=config.epochs, validation_data=validation_ds)

    return model, history
