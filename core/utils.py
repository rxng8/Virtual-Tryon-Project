import sys
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import math
import os
import re

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2

def conv(batch_input, out_channels):

    padded_input = tf.pad(
        batch_input, 
        [[0, 0], [1, 1], [1, 1], [0, 0]], 
        mode="CONSTANT"
    )

    return tf.keras.layers.Conv2D(
        filters=out_channels, 
        kernel_size=(4, 4), 
        activation='relu', 
        padding='valid'
    )(padded_input)

def max_pool(batch_input):
    return tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    ) (batch_input)

def final_deconv(batch_input, out_channels):
    return tf.keras.layers.Conv2DTranspose(
        out_channels, 4, strides=2,
        padding='same',
        activation='sigmoid'
    ) (batch_input)

def deconv(batch_input, out_channels):
    return tf.keras.layers.Conv2DTranspose(
        out_channels, 4, strides=2,
        padding='same',
        activation='relu'
    ) (batch_input)

def upsampling(batch_input):
    pass
