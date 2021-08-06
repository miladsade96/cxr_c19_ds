"""
    Chest X-Ray Covid-19 Detection System
"""


from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.losses import categorical_crossentropy


# Instantiating the model with passing default parameters values
vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 244, 3))

# Freezing vgg model layers
for layer in vgg_model.layers[:18]:
    layer.trainable = False

# Adding flatten and dense layers to vgg-16
output = vgg_model.output
output = Flatten()(output)
output = Dense(512, activation="relu")(output)
output = Dropout(0.5)(output)
output = Dense(3, activation="softmax")(output)
model = Model(inputs=vgg_model.inputs, outputs=output)

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
    target_size=(224, 224),
    batch_size=16,
    subset="training",
    class_mode="binary"
)

# Preparing data for validation process
valid_data = data_gen.flow_from_directory(
    directory="../dataset",
    target_size=(224, 224),
    batch_size=16,
    subset="validation",
    class_mode="binary"
)


# Compiling the model
model.compile(
    loss=sparse_categorical_crossentropy,
    optimizer=RMSprop(),
    metrics=['accuracy']
)

# Model architecture summary
model.summary()


class TerminateOnBaseLine(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline"""
    def __init__(self, monitor='accuracy', baseline=0.9):
        super(TerminateOnBaseLine, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % epoch)
                self.model.stop_training = True


# Terminating training process when validation accuracy reaches the specified baseline
cb = [TerminateOnBaseLine(monitor='val_accuracy', baseline=0.97)]

# Training section
history = model.fit_generator(
    train_data,
    epochs=50,
    validation_data=valid_data,
    verbose=1,
    workers=32,
    validation_steps=8,
    callbacks=cb
)

# Saving trained model
path = "../models"
model.save(filepath=path)
