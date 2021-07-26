"""
    Chest X-Ray Covid-19 Detection System
"""


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

# Setting image data generator parameters
# These are will used for image augmentation
data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255,
    zoom_range=0.3,
    validation_split=0.3
)

# Instantiating image data generator
data_gen = image.ImageDataGenerator(**data_gen_args)

# Preparing data for training process
train_data = data_gen.flow_from_directory(
    directory="../dataset",
    target_size=(256, 256),
    batch_size=16,
    subset="training",
    class_mode="binary"
)

# Preparing data for validation process
valid_data = data_gen.flow_from_directory(
    directory="../dataset",
    target_size=(256, 256),
    batch_size=16,
    subset="validation",
    class_mode="binary"
)
