"""
    Chest X-Ray Covid-19 Detection Model Based On Parallel Conv Layers

    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, Dense, BatchNormalization, Concatenate,
                                     Flatten, MaxPooling2D, AveragePooling2D, Input)


# Defining model input
input_ = Input(shape=(224, 224, 3))

# Defining first parallel layer
in_1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(input_)
conv_1 = BatchNormalization()(in_1)
conv_1 = AveragePooling2D(pool_size=(2, 2), strides=(3, 3))(conv_1)

# Defining second parallel layer
in_2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(input_)
conv_2 = BatchNormalization()(in_2)
conv_2 = AveragePooling2D(pool_size=(2, 2), strides=(3, 3))(conv_2)

# Defining third parallel layer
in_3 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(input_)
conv_3 = BatchNormalization()(in_3)
conv_3 = MaxPooling2D(pool_size=(2, 2), strides=(3, 3))(conv_3)

# Defining fourth parallel layer
in_4 = Conv2D(filters=16, kernel_size=(9, 9), activation='relu', padding='same')(input_)
conv_4 = BatchNormalization()(in_4)
conv_4 = MaxPooling2D(pool_size=(2, 2), strides=(3, 3))(conv_4)

# Concatenating layers
concat = Concatenate()([conv_1, conv_2, conv_3, conv_4])
flat = Flatten()(concat)
out = Dense(units=4, activation='softmax')(flat)
# Creating model
parallel_model = Model(inputs=[input_], outputs=[out], name="Parallel Model")


if __name__ == '__main__':
    parallel_model.summary()
