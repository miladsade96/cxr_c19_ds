"""
    Chest X-Ray Covid-19 Detection Model Based On Transfer Learning(VGG-16)

    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""

from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout


# Instantiating the model with passing default parameters values
# Weights parameter may be either "imagenet" string or direct path to .h5 file downloaded from googleapis
vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 244, 3))

# Freezing all vgg-16 model layers except last convolutional block
for layer in vgg_model.layers[:15]:
    layer.trainable = False
