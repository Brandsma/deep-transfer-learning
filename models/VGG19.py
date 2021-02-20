import tensorflow.keras as keras
from keras.applications.vgg19 import decode_predictions, preprocess_input
from logger import setup_logger
from tensorflow.keras.applications.vgg19 import VGG19

log = setup_logger(__name__)


def build_vgg19(config, num_classes):
    log.info("Building VGG19")

    base_model = VGG19(
        input_shape=(224, 224, 3), weights="imagenet", include_top=False, classes=3
    )

    inputs = keras.Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)
    output = base_model(x)

    model = keras.Model(inputs, output)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=config.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


def train_vgg19(train_ds, validation_ds, model, config):
    pass
