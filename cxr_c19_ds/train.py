"""
    Preprocessing the data and training the models

    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


import argparse
from os import getcwd
from transferred_model import transferred_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import categorical_crossentropy


# Initializing argument parser
parser = argparse.ArgumentParser()
# Adding optional arguments
parser.add_argument("-d", "--dataset", help="Path to dataset directory")
parser.add_argument("-b", "--batch-size", help="Batch size", type=int)
parser.add_argument("-l", "--learning-rate", help="Learning rate", type=float)
parser.add_argument("-v", "--verbose", help="Level of verbosity", action="store_true")
parser.add_argument("-w", "--workers", help="Number of workers", type=int)
parser.add_argument("-n", "--epochs", help="Number od epochs", type=int)
parser.add_argument("-s", "--save", help="Path to save trained model", default=getcwd())
# Parsing the arguments
args = parser.parse_args()


# Setting image data generator parameters
# These are will be used for image augmentation
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
    directory=args.dataset,
    target_size=(224, 224),
    batch_size=args.batch_size,
    subset="training"
)

# Preparing data for validation process
valid_data = data_gen.flow_from_directory(
    directory=args.dataset,
    target_size=(224, 224),
    batch_size=args.batch_size,
    subset="validation"
)

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
