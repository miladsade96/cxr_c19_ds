"""
    Retrain Saved Model
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.callbacks import Callback


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
