"""
    Training the models

    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""

from cxr_c19_ds.parallel_model import parallel_model
from cxr_c19_ds.transferred_model import transferred_model
from cxr_c19_ds.preprocessing import train_data, valid_data, cb
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import categorical_crossentropy
