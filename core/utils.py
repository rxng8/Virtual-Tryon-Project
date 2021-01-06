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

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

IMG_SHAPE = (256, 192, 3)
LABEL_NAME = {
    'Background': 0,
    'Hat': 1,
    'Hair': 2,
    'Glove': 3,
    'Sunglasses': 4,
    'UpperClothes': 5,
    'Dress': 6,
    'Coat': 7,
    'Socks': 8,
    'Pants': 9,
    'Jumpsuits': 10,
    'Scarf': 11,
    'Skirt': 12,
    'Face': 13,
    'Left-arm': 14,
    'Right-arm': 15,
    'Left-leg': 16,
    'Right-leg': 17,
    'Left-shoe': 18,
    'Right-shoe': 19
}


def conv(batch_input, out_channels, strides=1, activation='relu'):

    # padded_input = tf.pad(
    #     batch_input, 
    #     [[0, 0], [1, 1], [1, 1], [0, 0]], 
    #     mode="CONSTANT"
    # )

    out = tf.keras.layers.Conv2D(
        filters=out_channels, 
        kernel_size=(4, 4),
        activation=activation, 
        padding='same'
    )(batch_input)
    # print(out.shape)
    return out

def dropout(batch_input, rate=0.5):
    return tf.keras.layers.Dropout(
        rate
    ) (batch_input)

def max_pool(batch_input):
    return tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    ) (batch_input)

def final_conv(batch_input, out_channels):
    # out = tf.keras.layers.Conv2D(
    #     filters=out_channels, 
    #     kernel_size=(4, 4),
    #     activation='sigmoid', 
    #     padding='same'
    # )(batch_input)
    # return out
    return conv(batch_input, out_channels, activation='sigmoid')

def final_deconv(batch_input, out_channels):
    return tf.keras.layers.Conv2DTranspose(
        out_channels, 4, strides=2,
        padding='same',
        activation='sigmoid'
    ) (batch_input)

def deconv(batch_input, out_channels, activation='relu'):
    return tf.keras.layers.Conv2DTranspose(
        out_channels, 4, strides=2,
        padding='same',
        activation=activation
    ) (batch_input)

def upsampling(batch_input):
    pass

def create_mask(pred_mask):
    # pred_mask shape 3d, not 4d
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def preprocess_image(
        img: np.ndarray, 
        shape=(256, 192),
        resize_method=tf.image.ResizeMethod.BILINEAR,
        tanh_range=True
    ) -> tf.Tensor:
    if len(img.shape) == 2:
        img = tf.expand_dims(img, axis=-1)
    # Expect range 0 - 255
    resized = tf.image.resize(
        img, 
        shape,
        method=resize_method
    )
    rescaled = tf.cast(resized, dtype=float) / 255.0
    if tanh_range:
        rescaled = (rescaled - 0.5) * 2 # range [-1, 1]
    # Convert to BGR
    bgr = rescaled[..., ::-1]
    return bgr

def preprocess_mask(
    img: np.ndarray, 
    shape=(256, 192),
    resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    sigmoid_range=True
) -> tf.Tensor:
    # Expect range 0 - 255
    resized = tf.image.resize(
        img, 
        shape,
        method=resize_method
    )
    if sigmoid_range and tf.reduce_max(resized) > 1:
        resized = tf.cast(resized, dtype=float) / 255.0
    return resized

def deprocess_img(img):
    # Expect img range [-1, 1]
    # Do the rescale back to 0, 1 range, and convert from bgr back to rgb
    return (img / 2.0 + 0.5)[..., ::-1]

def show_img(img):
    if len(img.shape) == 3:
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    elif len(img.shape) == 2:
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()

def load_image(path):
    return np.asarray(Image.open(path))


def plot_parsing_map(test_predict, label2int=LABEL_NAME):
    # Test pred shape 3d or 4d
    # mask is shape (256, 192, 1). Range[0, 19]
    mask = None
    if len(test_predict.shape) == 4:
        mask = create_mask(test_predict)[0]
    else:
        mask = create_mask(test_predict)
    width=15
    height=10
    rows = 4
    cols = 5
    axes=[]
    fig=plt.figure(figsize=(width, height))

    for i, (label_name, label_channel) in enumerate(label2int.items()):
        
        axes.append(fig.add_subplot(rows, cols, i+1))
        subplot_title=(label_name)
        axes[-1].set_title(subplot_title)
        axes[-1].axis('off')
        plt.imshow(mask == label_channel, cmap='gray')
    fig.tight_layout()
    plt.show()

def get_pose_map(path) -> np.ndarray:
    """ given a path, return a pose map

    Args:
        img (np.ndarray): 

    Returns:
        np.ndarray: [description]
    """
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.imread(path) # range 0 - 255. 3d
    image = cv2.resize(image, IMG_SHAPE[:2][::-1],fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height, image_width, channels = image.shape
    annotated_image = np.zeros((image_height, image_width, channels))
    if not results.pose_landmarks:
        return annotated_image
    mp_drawing.draw_landmarks(
        annotated_image, 
        results.pose_landmarks)
    pose.close()
    # return preprocess_image(np.asarray(annotated_image))
    return tf.cast(annotated_image, dtype=tf.int32)


def get_hand_pose_map(path) -> np.ndarray:
    """ given a path, return a pose map

    Args:
        img (np.ndarray): 

    Returns:
        np.ndarray: [description]
    """
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.imread(path) # range 0 - 255. 3d
    image = cv2.resize(image, IMG_SHAPE[:2][::-1],fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    # show_img(image)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    # print('Handedness:', results.multi_handedness)
    
    image_height, image_width, channels = image.shape
    annotated_image = image.copy()
    empty_image = np.zeros((image_height, image_width, channels))
    if not results.multi_hand_landmarks:
        return annotated_image
    for hand_landmarks in results.multi_hand_landmarks:
        # print('hand_landmarks:', hand_landmarks)
        # print(
        #     f'Index finger tip coordinates: (',
        #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
        #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
        # )
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            empty_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    # cv2.imwrite(
    #     '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    hands.close()
    # return preprocess_image(np.asarray(annotated_image))
    return annotated_image, empty_image

def compute_mae_loss(real, pred):
    return tf.reduce_mean(tf.math.abs(real - pred))

def compute_mse_loss(real, pred):
    return tf.reduce_mean(tf.math.abs(real - pred) ** 2)