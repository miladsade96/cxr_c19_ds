"""
    Chest X-Ray Covid-19 Detection Model Based On Transfer Learning(VGG-16)

    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""

from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
