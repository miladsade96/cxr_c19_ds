"""
    Loading saved model and testing prediction accuracy
"""


import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Defining class names
classes = {0: "Covid", 1: "Lung Opacity", 2: "Normal", 3: "Viral Pneumonia"}
