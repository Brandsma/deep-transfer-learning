import tensorflow as tf
import tensorflow_hub as hub
from logger import setup_logger

log = setup_logger(__name__)


def build_mobilenet(num_classes):
    log.info("Building Mobile Net V2")
    # CONSTANTS
    IMAGE_SHAPE = (224, 224)

    # We download the headless model from tensorflow hub
    classifier_model = (
        "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    )
    feature_extractor_layer = hub.KerasLayer(
        classifier_model, input_shape=IMAGE_SHAPE + (3,), trainable=False
    )

    # Add a classification layer, which is a dense layer connected to num_classes nodes
    classification_layer = tf.keras.layers.Dense(num_classes)

    # Build the classifier
    classifier = tf.keras.Sequential([feature_extractor_layer, classification_layer])

    return classifier


def train_mobilenet(train_ds, validation_ds, model, config):
    log.info("Training Mobile Net V2")

    # We train the model using an optimizer and SparseCategoricalCrossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["acc"],
    )

    history = model.fit(train_ds, epochs=config.epochs, validation_data=validation_ds)

    return model, history
