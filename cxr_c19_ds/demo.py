"""
    Loading saved model and testing prediction accuracy
"""


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Loading saved model and displaying its architecture
path = "../models/"
model = load_model(filepath=path)
model.summary()

# Defining class names
classes = {0: "Covid", 1: "Lung Opacity", 2: "Normal", 3: "Viral Pneumonia"}

# Analyzing model prediction
image_path = "../For Local Test/COVID/COVID-2893.png"
img = load_img(image_path, target_size=(224, 224))
img_array = img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_batch)
print(classes.get(int(np.argmax(prediction))))
