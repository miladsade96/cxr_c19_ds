"""
    Retrain Saved Model
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


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
)

# Instantiating image data generator
data_gen = image.ImageDataGenerator(**data_gen_args)
# Preparing data for training process
train_data = data_gen.flow_from_directory(
    directory="/dataset/",
    target_size=(224, 224),
    batch_size=32,
    subset="training"
)
