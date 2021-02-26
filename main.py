import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.datasets import cifar10
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from argument_parser import get_arg_parser
from logger import setup_logger
from models.MobileNetV2 import build_mobilenet, train_mobilenet
from models.ResNet50 import build_resnet, train_resnet
from models.VGG19 import build_vgg19, train_vgg19
from results import save_test_results, save_training_results

log = setup_logger(__name__)


def load_cifar(config):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
        config.batch_size
    )
    validation_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
        config.batch_size
    )
    test_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(
        config.batch_size
    )

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Prefetch the data so we do not have to wait for the disk reading
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return (train_ds, validation_ds, test_ds, np.array(class_names))


def load_dataset(config):
    if config.load_cifar:
        return load_cifar(config)
    # The size of our images is dependent on which network we use
    image_input = {
        "VGG19": (224, 224),
        "MobileNetV2": (224, 224),
        "InceptionV3": (299, 299),
    }
    image_size = image_input[config.model_type]

    # Retrieve the training and validation dataset from the given directories
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.training_dir,
        shuffle=True,
        image_size=image_size,
        batch_size=config.batch_size,
    )
    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.test_dir,
        shuffle=True,
        image_size=image_size,
        batch_size=config.batch_size,
    )

    # Extract the labels of all the possible classes
    class_names = np.array(train_ds.class_names)

    # All pixel values [0,255] to [0,1]
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
        1.0 / 255
    )
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    validation_ds = validation_ds.map(lambda x, y: (normalization_layer(x), y))

    # Create a test dataset from 20% of the validation dataset
    val_batches = tf.data.experimental.cardinality(validation_ds)
    test_ds = validation_ds.take(val_batches // 5)
    validation_ds = validation_ds.skip(val_batches // 5)

    # Prefetch the data so we do not have to wait for the disk reading
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return (train_ds, validation_ds, test_ds, class_names)


# start doing useful stuff


def train_model(model_type, train_ds, validation_ds, model, config):
    models = {
        "VGG19": train_vgg19,
        "MobileNet": train_mobilenet,
        "ResNet50": train_resnet,
    }
    return models[model_type](train_ds, validation_ds, model, config)


def get_model(model_type, num_classes, config):
    if config.load_model:
        return load_model(config)

    models = {
        "VGG19": build_vgg19,
        "MobileNet": build_mobilenet,
        "ResNet50": build_resnet,
    }
    return models[model_type](config, num_classes)


def load_model(config):
    return tf.keras.models.load_model(config.model_path)


def save_model(model, config):
    path = f"{config.model_path}/{config.model_type}-{int(time.time())}"
    model.save(path)


def main(config):
    log.info("Setting up...")
    name = f"{config.model_type}-{config.optimizer}{f'-dropout_{config.dropout_rate}' if config.with_dropout else ''}-epochs_{config.epochs}"
    result_dir = f"./results/{name}/"
    try:
        os.makedirs(result_dir)
    except OSError as error:
        log.warn(error)

    log.info("Setup complete")

    log.info("Loading dataset...")
    train_ds, validation_ds, test_ds, class_names = load_dataset(config)
    log.info("Loading dataset done")

    log.info("Loading model...")
    model = get_model(config.model_type, len(class_names), config)
    log.info("Loading model done")

    log.info("Training...")
    model, history = train_model(
        config.model_type, train_ds, validation_ds, model, config
    )

    save_training_results(result_dir, name, history, config)
    log.info("Training Done")

    if config.skip_saving_model:
        # We save the model by default, but it can be turned off
        log.info("Saving model...")
        save_model(model, config)
        log.info("Saved model")

    log.info("Evaluating...")

    save_test_results(
        result_dir,
        name,
        model,
        test_ds,
        class_names,
        config,
    )

    log.info("Evaluation Done")

    log.info("  --Completed--  ")


if __name__ == "__main__":
    # The program starts here
    config = get_arg_parser()
    main(config)
