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

# Adding flatten and dense layers to vgg-16
output = vgg_model.output
output = Flatten()(output)
output = Dense(512, activation="relu")(output)
output = Dropout(0.2)(output)
output = Dense(4, activation="softmax")(output)
transferred_model = Model(inputs=vgg_model.inputs, outputs=output, name="Transferred Model")
