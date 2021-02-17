import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

def build_mobilenet(config):
    # Pre-create all the required layers
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    prediction_layer = tf.keras.layers.Dense(1)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    base_model = MobileNetV2(include_top=False, weights='imagenet')

    # Set some model specific settings
    base_model.trainable = False

    # Build up the model
    inputs = tf.keras.Input(shape=(299, 299, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    if config.with_dropout:
        x = tf.keras.layers.Dropout(config.dropout_rate)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=config.learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    print(model.summary())
    return model
