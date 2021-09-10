"""
    Chest X-Ray Covid-19 Detection Model Based On Parallel Conv Layers

    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.layers import (Conv2D, Dense, BatchNormalization, Concatenate,
                                     Flatten, MaxPooling2D, AveragePooling2D, Input)


# Defining model input
input_ = Input(shape=(224, 224, 3))

# Defining first parallel layer
conv_1 = Conv2D(filters=16, kernel_size=(3, 3), activation=relu)(input_)
conv_1 = BatchNormalization()(conv_1)
conv_1 = AveragePooling2D(pool_size=(2, 2), strides=(3, 3))(conv_1)

# Defining second parallel layer
conv_2 = Conv2D(filters=16, kernel_size=(5, 5), activation=relu)(input_)
conv_2 = BatchNormalization()(conv_2)
conv_2 = AveragePooling2D(pool_size=(2, 2), strides=(3, 3))(conv_2)
