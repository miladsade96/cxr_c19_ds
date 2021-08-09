"""
    Loading saved model and testing prediction accuracy
"""


import numpy as np
from PIL import Image
import streamlit as st
from tensorflow import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

