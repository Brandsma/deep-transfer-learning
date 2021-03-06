from logger import setup_logger
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam, Adamax

log = setup_logger(__name__)


def get_callbacks():
    # return [EarlyStopping(patience=20), ReduceLROnPlateau()]
    return [ReduceLROnPlateau()]


def get_optimizer(optimizer_type, learning_rate):
    optimizers = {
        "adam": Adam,
        "adamax": Adamax,
        "sgd": SGD,
    }
    log.info(f"Using optimizer {optimizers[optimizer_type]}")
    return optimizers[optimizer_type](learning_rate=learning_rate)
