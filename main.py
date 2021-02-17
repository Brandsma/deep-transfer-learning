import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt

from models.MobileNetV2 import build_mobilenet
from models.InceptionV3 import build_inception
from models.VGG19 import build_vgg19


from logger import setup_logger

log = setup_logger(__name__)


def load_dataset(train_location, validation_location, model_type, batch_size=16):
    image_input = {
        "VGG19": (224, 224),
        "MobileNetV2": (299, 299),
        "InceptionV3": (100, 100),
    }
    image_size = image_input[model_type]

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_location,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size
    )

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        validation_location,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(4):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
            plt.savefig("./input.png")



    # All pixel values [0,255] to [-1,1]
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
        1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    validation_ds = validation_ds.map(lambda x, y: (normalization_layer(x), y))

    # train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    # validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


    return (train_ds, validation_ds, class_names)

# start doing usefull stuff

def get_model(model_type):
    models = {
        "VGG19": build_vgg19(config),
        "MobileNetV2": build_mobilenet(config),
        "InceptionV3": build_inception(config),
    }
    return models[model_type]

def decode_predictions(validation_ds, class_names):
    #Retrieve a batch of images from the test set
    image_batch, label_batch = validation_ds.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

    plt.figure(figsize=(10, 10))
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(image_batch[i].astype("uint8"))
      plt.title(class_names[predictions[i]])
      plt.axis("off")
      plt.savefig("./input.png")


def main(config):
    log.info("Loading dataset...")
    train_ds, validation_ds, class_names = load_dataset(config.training_dir, config.test_dir, config.model_type)
    log.info("Loading dataset done")

    log.info("Loading model...")
    model = get_model(config.model_type)
    log.info("Loading model done")

    log.info("Predicting...")
    # predictions = model.predict(validation_ds, verbose=1)
    log.info("Prediction Done")

    log.info("Decode, aka convert the probabilities to class labels")
    decode_predictions(class_names)
    log.info("Decode Done")


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("training_dir", help="input directory where all training images are found")
    parser.add_argument("test_dir", help="input directory where all test images are found")
    parser.add_argument("--model_type", help="Which base model to transfer learning from", default="MobileNetV2", choices=["VGG19", "MobileNetV2", "InceptionV3"])
    parser.add_argument("--learning_rate", help="Base learning rate for optimizers", default=0.0001, type=float)
    parser.add_argument("--dropout_rate", help="Dropout rate for the dropout layers", default=0.2, type=float)
    parser.add_argument("--with_dropout", help="Determines if we are going to use a dropout layer in the models", store_action=True)
    parser.add_argument("--epochs", help="Number of epochs to run the model for", default=10, type=int)

    return parser.parse_args()



if __name__ == "__main__":
    config = get_arg_parser()
    main(config)
