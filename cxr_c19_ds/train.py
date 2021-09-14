"""
    Training the models

    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""

from os import getcwd
from cxr_c19_ds.parallel_model import parallel_model
from cxr_c19_ds.transferred_model import transferred_model
from cxr_c19_ds.preprocessing import train_data, valid_data, cb
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import categorical_crossentropy


# Defining models
models = {"VGG-16": transferred_model, "Parallel": parallel_model}
