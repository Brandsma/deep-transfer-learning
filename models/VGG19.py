from tensorflow.keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

def build_vgg19(config):
    VGG19(weights='imagenet', include_top=False)
