import tensorflow as tf
import numpy as np


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

    return (train_ds, validation_ds)


def main():
    # train_ds, validation_ds = load_dataset("./Training", "./Test")
    # class_names = np.array(train_ds.class_names)
    # print(class_names)

    a = 1
    b = 2
    print(a + b)


if __name__ == "__main__":
    main()
