import tensorflow as tf
import tensorflow_hub as hub
from keras.applications import MobileNet
from keras.layers import Dropout
from logger import setup_logger

from models.util import get_callbacks, get_optimizer

log = setup_logger(__name__)


def build_mobilenet(config, num_classes):
    log.info("Building Mobile Net")
    # CONSTANTS
    if config.load_cifar:
        IMAGE_SHAPE = (32, 32, 3)
    else:
        IMAGE_SHAPE = (224, 224, 3)

    feature_extractor_layer = MobileNet(
        include_top=False,
        weights="imagenet",
        input_shape=IMAGE_SHAPE,
        classes=num_classes,
    )
    feature_extractor_layer.trainable = False

    # Add a classification layer, which is a dense layer connected to num_classes nodes
    classification_layer = tf.keras.layers.Dense(num_classes)

    # Build the classifier
    classifier = tf.keras.Sequential()

    classifier.add(feature_extractor_layer)
    classifier.add(tf.keras.layers.GlobalAveragePooling2D())

    if config.with_dropout:
        classifier.add(Dropout(config.dropout_rate))
    classifier.add(classification_layer)

    classifier.summary()

    return classifier


def train_mobilenet(train_ds, validation_ds, model, config):
    log.info("Training Mobile Net")

    # We train the model using an optimizer and SparseCategoricalCrossentropy
    model.compile(
        optimizer=get_optimizer(config.optimizer, config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["acc"],
    )

    history = model.fit(
        train_ds,
        epochs=config.epochs,
        validation_data=validation_ds,
        callbacks=get_callbacks(),
    )

    return model, history
