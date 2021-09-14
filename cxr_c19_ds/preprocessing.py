"""
    Preprocessing settings and data generator configurations

    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping


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
    validation_split=0.2
)

# Instantiating image data generator
data_gen = image.ImageDataGenerator(**data_gen_args)

# Preparing data for training process
train_data = data_gen.flow_from_directory(
    directory="path to kaggle dataset",
    target_size=(224, 224),
    batch_size=32,
    subset="training"
)

# Preparing data for validation process
valid_data = data_gen.flow_from_directory(
    directory="path to kaggle dataset",
    target_size=(224, 224),
    batch_size=32,
    subset="validation"
)

# Early Stopping
early_stop_rule = EarlyStopping(monitor="val_acc", mode="max", baseline=0.95)
cb = [early_stop_rule]
