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

model_path = args.model
# Loading saved model
model = load_model(filepath=model_path)
if args.verbose:
    # Displaying model architecture and details
    print(model.summary())

# Loading the image
image_path = args.image
img = load_img(image_path, target_size=(224, 224))
if args.verbose:
    print("Image is loaded and resized to 224x224")
img_array = img_to_array(img)
if args.verbose:
    print("Image converted to the array.")
img_batch = np.expand_dims(img_array, axis=0)
if args.verbose:
    print("Image batch created.")

# Predicting the class
prediction = model.predict(img_batch)
if args.verbose:
    print(f"Predicted class is: {classes.get(int(np.argmax(prediction)))}")
else:
    print(classes.get(int(np.argmax(prediction))))
