"""
    Loading saved model and testing prediction accuracy
"""


import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Defining class names
classes = {0: "Covid", 1: "Lung Opacity", 2: "Normal", 3: "Viral Pneumonia"}

# Instantiating cli argument parser
parser = argparse.ArgumentParser()
# Adding optional arguments
parser.add_argument("-m", "--model", help="Path to saved model directory")
parser.add_argument("-i", "--image", help="Path to image file")
parser.add_argument("-v", "--verbose", action="store_true", help="Level of verbosity")
# Parsing the arguments
args = parser.parse_args()
