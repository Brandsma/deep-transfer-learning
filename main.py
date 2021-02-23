import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from argument_parser import get_arg_parser
from logger import setup_logger
from models.InceptionV3 import build_inception, train_inception
from models.MobileNetV2 import build_mobilenet, train_mobilenet
from models.VGG19 import build_vgg19, train_vgg19

log = setup_logger(__name__)


def load_dataset(config):
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
        "MobileNetV2": train_mobilenet,
        "InceptionV3": train_inception,
    }
    return models[model_type](train_ds, validation_ds, model, config)


def get_model(model_type, num_classes, config):
    if config.load_model:
        return load_model(config)

    models = {
        "VGG19": build_vgg19,
        "MobileNetV2": build_mobilenet,
        "InceptionV3": build_inception,
    }
    return models[model_type](config, num_classes)


def load_model(config):
    return tf.keras.models.load_model(config.model_path)


def save_model(model, config):
    path = f"{config.model_path}/{config.model_type}-{int(time.time())}"
    model.save(path)


def save_training_results(history, config):
    log.info("Saving training results")

    # Save the history to a csv
    hist_df = pd.DataFrame.from_dict(history.history)
    hist_df.to_csv(
        f"./{config.model_type}-{config.optimizer}{f'-dropout_{config.dropout_rate}' if config.with_dropout else ''}-epochs_{config.epochs}-history.csv"
    )

    # Show the training vs validation loss
    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{config.model_type} loss")
    plt.legend()
    plt.savefig(f"./{config.model_type}-loss.png")

    # Show the training vs validation accuracy
    plt.figure()
    plt.plot(history.history["acc"], label="Training Accuracy")
    plt.plot(history.history["val_acc"], label="Validation Accuracy")
    plt.title(f"{config.model_type} accuracy")
    plt.legend()
    plt.savefig(f"./{config.model_type}-accuracy.png")


def main(config):
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

    save_training_results(history, config)
    log.info("Training Done")

    if config.skip_saving_model:
        # We save the model by default, but it can be turned off
        log.info("Saving model...")
        save_model(model, config)
        log.info("Saved model")

    log.info("Predicting after training...")
    # We take one batch of (16) images and show 9 of them
    # together with their predicted labels

    prediction_batch = model.predict(validation_ds, verbose=1)
    prediction_id = np.argmax(prediction_batch, axis=-1)
    predicted_label_batch = class_names[prediction_id]

    for image_batch, label_set in validation_ds.take(1):
        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(hspace=0.5)
        for n in range(min(9, config.batch_size)):
            plt.subplot(3, 3, n + 1)
            plt.imshow(image_batch[n])
            plt.title(predicted_label_batch[n].title())
            plt.axis("off")
            _ = plt.suptitle(f"{config.model_type} predictions")
            plt.savefig(f"./{config.model_type}-predictions.png")

    log.info("Prediction Done")

    log.info("  --Completed--  ")


if __name__ == "__main__":
    # The program starts here
    config = get_arg_parser()
    main(config)
