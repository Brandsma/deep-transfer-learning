import tensorflow as tf
import numpy as np
import argparse


from logger import setup_logger

log = setup_logger(__name__)


def load_dataset(train_location, validation_location, batch_size=32, image_size=(100, 100)):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_location,
        seed=42,
        image_size=image_size,
        batch_size=batch_size
    )

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        validation_location,
        seed=42,
        image_size=image_size,
        batch_size=batch_size
    )

    # All pixel values [0,255] to [0,1]
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
        1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    validation_ds = validation_ds.map(lambda x, y: (normalization_layer(x), y))

    # train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    # validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return (train_ds, validation_ds)

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Input

def get_model():
    model = VGG19(weights='imagenet', input_shape=(100,100,3), include_top=False)
    return model

def main(config):
    log.info("Loading dataset...")
    train_ds, validation_ds = load_dataset(config.training_dir, config.test_dir)
    log.info("Loading dataset done")

    log.info("Loading model...")
    model = get_model()
    log.info("Loading model done")

    log.info("Predicting...")
    predictions = model.predict(validation_ds, verbose=1)
    print(predictions)
    log.info("Prediction Done")


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("training_dir", help="input directory where all training images are found")
    parser.add_argument("test_dir", help="input directory where all test images are found")
    parser.add_argument("--model", help="Which base model to transfer learning from", default="VGG19", choices=["VGG19"])

    return parser.parse_args()



if __name__ == "__main__":
    config = get_arg_parser()
    main(config)
