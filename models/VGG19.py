import tensorflow.keras as keras
from keras.applications.vgg19 import decode_predictions, preprocess_input
from logger import setup_logger
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential

log = setup_logger(__name__)


def build_vgg19(config, num_classes):
    log.info("Building VGG19")

    model = VGG19(
        input_shape=(224, 224, 3), weights="imagenet", include_top=False, classes=3
    )

    # check structure and layer names before looping
    model.summary()

    # loop through layers, add Dropout after layers 'fc1' and 'fc2'
    if config.with_dropout:
        updated_model = Sequential()

    model = updated_model

    # check structure
    model.summary()
    return model


def train_vgg19(train_ds, validation_ds, model, config):
    pass
