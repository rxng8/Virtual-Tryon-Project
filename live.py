#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College
"""

# %%

# Import

from pathlib import Path
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# config
PARSING_MODEL_PATH = "./models/human_parsing_mbv2-50epochs"
parsing_model = tf.keras.models.load_model(PARSING_MODEL_PATH)

COMPUTING_IMG_SHAPE = (256, 192, 3)

# Label name, the actual label in the parsing image. note that it is 1 channel
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

ORANGE_RED = (255,69,0)

COLOR_LABEL = {
    'Background': ORANGE_RED,
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

# %%

# Method definition



def parse_human(img):
    # Parse the human
    resized = tf.image.resize(img, COMPUTING_IMG_SHAPE[:2])
    rescaled = resized / 255.0
    prediction = parsing_model.predict(tf.expand_dims(rescaled, axis=0))[0]
    return prediction # 20 channel

def create_mask(pred_mask):
    # pred_mask shape (256, 192, 20)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def draw_simple_parsing(original_img, prediction, threshold: float=0.5):
    # Original image have the shape of the cam, 3 channel (height, width, 3). Scale 0 - 255
    # prediction have the shape of the computing shape, 20 channels (height, width, 20). Scale 0 - 1
    assert original_img.shape[:2] == prediction.shape[:2], "image width and height not matching"
    
    # sample_body_mask shape (256, 192, 1). Range [0, 19]. Representing classes.
    sample_mask = create_mask(prediction)

    body_masking_channels = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
    sample_body_mask = [sample_mask == c for c in body_masking_channels]
    # sample_body_mask shape (len(body_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
    sample_body_mask = tf.reduce_any(sample_body_mask, axis=0)
    # Now sample_body_mask have shape (256, 192, 1). Range [False, True]
    sample_body_mask = tf.squeeze(sample_body_mask)
    # Now sample_body_mask have shape (256, 192). Range [False, True]

    # set the value to white
    original_img[sample_body_mask, :] = ORANGE_RED

def draw_parsing(original_img, prediction, threshold: float=0.5):
    # Original image have the shape of the cam, 3 channel (height, width, 3). Scale 0 - 255
    # prediction have the shape of the computing shape, 20 channels (height, width, 20). Scale 0 - 1
    assert original_img.shape[:2] == prediction.shape[:2], "image width and height not matching"
    
    # Dont take the background channels but take all others channels. Take the last 19 channels.
    bool_mat = prediction[:, :, 1:] > threshold

    # For each channel, draw a different color.
    # TODO: 


    # set the value to white
    original_img[bool_mat, :] = ORANGE_RED


def show_img(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

def plot_parsing_map(test_predict):
    width=15
    height=10
    rows = 4
    cols = 5
    axes=[]
    fig=plt.figure(figsize=(width, height))

    for i, (label_name, label_channel) in enumerate(LABEL_NAME.items()):
        
        axes.append(fig.add_subplot(rows, cols, i+1))
        subplot_title=(label_name)
        axes[-1].set_title(subplot_title)
        axes[-1].axis('off')
        plt.imshow(test_predict[:, :, label_channel] > 0.5, cmap='gray')
    fig.tight_layout()
    plt.show()

# %%

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    image_height, image_width, _ = image.shape

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False

    # Prediction
    results = parse_human(image)

    # resize
    results = tf.image.resize(results, (image_height, image_width))
    # results = tf.image.flip_left_right(results)

    # Draw human parsing on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    draw_simple_parsing(image, results)
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()

# %%

file_path = "./dataset/pose/1.jpg"

image = cv2.imread(file_path)
image_height, image_width, _ = image.shape
# Convert the BGR image to RGB before processing.
prep = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = parse_human(prep)
results = tf.image.resize(results, (image_height, image_width))
# # Draw thingy on the image.
annotated_image = image.copy()
draw_parsing(annotated_image, results)
show_img(annotated_image)

