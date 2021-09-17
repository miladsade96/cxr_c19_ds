"""
    Preprocessing the data and training the models

    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


import argparse
from os import getcwd
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
from cxr_c19_ds.transferred_model import transferred_model
from tensorflow.keras.losses import categorical_crossentropy


# Defining supported models
supported_models = {"VGG-16": transferred_model, "Parallel": parallel_model}


while True:
    print("  CXR_C19_DS  Training".center(30, "*"))
    print(f"Supported models are: {[keys for keys in supported_models.keys()]}")
    value = input("Please select the model to train:")
    if value not in supported_models.keys():
        print(f"You selected unsupported model: {value}. Please try again!")
        continue
    else:
        if value == "VGG-16":
            model = supported_models.__getitem__("VGG-16")
            # Displaying model details
            model.summary()

        else:
            model = supported_models.__getitem__("Parallel")
            # Displaying model details
            model.summary()

        # Model compilation
        model.compile(
            optimizer=RMSprop(learning_rate=0.00001),
            loss=categorical_crossentropy,
            metrics=["accuracy"]
        )

        # Training model
        model.fit(
            train_data,
            validation_data=valid_data,
            verbose=1,
            workers=32,
            epochs=100,
            validation_steps=8,
            callbacks=cb
        )
        print("TRaining process finished successfully.")

        # Saving trained model
        model.save(filepath="./")

        print(f"Model Saved at {getcwd()}")
