# %%

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


# config
# Please config the path as accurate as possible

# this is the path to the root of the dataset
DATASET_PATH = Path("./dataset/lip_mpv_dataset/")

# This is the path to the folder contain actual data
DATASET_SRC = DATASET_PATH / "MPV_192_256"

# This is the name of the file where it contains path to data
DATASET_FILE = "all_poseA_poseB_clothes.txt"

BATCH_SIZE = 32
STEP_PER_EPOCHS = 20
IMG_SHAPE = (256, 192, 3)

MASK_THRESHOLD = 0.9

# load human parsing model that has been trained in human_parsing notebook.
parsing_model = tf.keras.models.load_model('models/human_parsing_mbv2-50epochs')

def get_data_path_raw():
    train_half_front = []
    test_half_front = []

    with open(DATASET_PATH / DATASET_FILE, 'r') as f:
        for line in f:
            elems = line.split("\t")
            assert len(elems) == 4, "Unexpected readline!"
            if "train" in line:
                if "person_half_front.jpg" in line and "cloth_front.jpg" in line:
                    tmp_person = ""
                    tmp_cloth = ""
                    for elem in elems:
                        if "person_half_front.jpg" in elem:
                            tmp_person = str(elem)
                        if "cloth_front.jpg" in elem:
                            tmp_cloth = str(elem)
                    train_half_front.append([tmp_person, tmp_cloth])
                else:
                    continue
            elif "test" in line:
                if "person_half_front.jpg" in line and "cloth_front.jpg" in line:
                    tmp_person = ""
                    tmp_cloth = ""
                    for elem in elems:
                        if "person_half_front.jpg" in elem:
                            tmp_person = str(elem)
                        if "cloth_front.jpg" in elem:
                            tmp_cloth = str(elem)
                    test_half_front.append([tmp_person, tmp_cloth])
                else:
                    continue
            else:
                print("Unexpected behavior!")

    return np.asarray(train_half_front), np.asarray(test_half_front)

def get_data_path():
    train_half_front = []
    test_half_front = []

    with open(DATASET_PATH / DATASET_FILE, 'r') as f:
        for line in f:
            elems = line.split("\t")
            assert len(elems) == 4, "Unexpected readline!"
            if "train" in line:
                if "person_half_front.jpg" in line and "cloth_front.jpg" in line:
                    tmp_person = ""
                    tmp_cloth = ""
                    for elem in elems:
                        if "person_half_front.jpg" in elem:
                            tmp_person = str(DATASET_SRC / elem)
                        if "cloth_front.jpg" in elem:
                            tmp_cloth = str(DATASET_SRC / elem)
                    train_half_front.append([tmp_person, tmp_cloth])
                else:
                    continue
            elif "test" in line:
                if "person_half_front.jpg" in line and "cloth_front.jpg" in line:
                    tmp_person = ""
                    tmp_cloth = ""
                    for elem in elems:
                        if "person_half_front.jpg" in elem:
                            tmp_person = str(DATASET_SRC / elem)
                        if "cloth_front.jpg" in elem:
                            tmp_cloth = str(DATASET_SRC / elem)
                    test_half_front.append([tmp_person, tmp_cloth])
                else:
                    continue
            else:
                print("Unexpected behavior!")

    return np.asarray(train_half_front), np.asarray(test_half_front)

TRAIN_PATH, TEST_PATH = get_data_path()

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


# %%

# Some definition



def get_pose_map_generator(path_list: np.ndarray) -> None:
    """ given a path, return a pose map

    Args:
        img (np.ndarray): 

    Returns:
        np.ndarray: [description]
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    pose = mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5)
    for idx, f in enumerate(path_list):
        image = cv2.imread(f)
        image_height, image_width, n_channels = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            continue
        # Draw pose landmarks on the image.
        annotated_image = np.zeros(shape=(image_height, image_width, n_channels))
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks)
        yield np.asarray(annotated_image) / 255.0
    pose.close()

def get_pose_map(path) -> np.ndarray:
    """ given a path, return a pose map

    Args:
        img (np.ndarray): 

    Returns:
        np.ndarray: [description]
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5)
    image = cv2.imread(path)
    image_height, image_width, n_channels = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    RED_COLOR = (0, 0, 255)
    if results.pose_landmarks:
        annotated_image = np.zeros(shape=(image_height, image_width, n_channels))
        
        mp_drawing.draw_landmarks(
            annotated_image, 
            results.pose_landmarks,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                # color=RED_COLOR,
                thickness=4,
                circle_radius=1
            ))
        pose.close()
        return tf.image.resize(np.asarray(annotated_image) / 255.0, IMG_SHAPE[:2])
    pose.close()
    return np.zeros(shape=IMG_SHAPE)

def get_human_parsing(img):
    assert img.shape == IMG_SHAPE, "Wrong image shape"
    prediction = parsing_model.predict(tf.expand_dims(img, axis=0))[0]
    return prediction

def create_mask(pred_mask):
    # pred_mask shape 3d, not 4d
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def show_img(img):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# %%

# Sample data

r = np.random.randint(0, TRAIN_PATH.shape[0] - 1)

# sample_img shape (256, 192, 3). Range [0, 1]
sample_cloth = tf.image.resize(
    np.asarray(
        Image.open(
            TRAIN_PATH[r, 1]
        )
    ), IMG_SHAPE[:2]
) / 255.0
show_img(sample_cloth)

# sample_img shape (256, 192, 3). Range [0, 1]
sample_img = tf.image.resize(
    np.asarray(
        Image.open(
            TRAIN_PATH[r, 0]
        )
    ), IMG_SHAPE[:2]
) / 255.0

show_img(sample_img)

# sample_pose shape (256, 192, 3). Range [0, 1]
sample_pose = get_pose_map(TRAIN_PATH[r, 0])
show_img(sample_pose)

# sample_pose shape (256, 192, 20). Range [0, 1]
sample_parsing = get_human_parsing(sample_img)
# sample_body_mask shape (256, 192, 1). Range [0, 19]. Representing classes.
sample_mask = create_mask(sample_parsing)

# Take everything except for the background, face, and hair
body_masking_channels = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
sample_body_mask = [sample_mask == c for c in body_masking_channels]
# sample_body_mask shape (len(body_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
sample_body_mask = tf.reduce_any(sample_body_mask, axis=0)
sample_body_mask = tf.cast(sample_body_mask, dtype=tf.float32)
show_img(sample_body_mask)

# Take everything except for the background, face, and hair
face_hair_masking_channels = [1, 2, 13]
sample_face_hair_mask = [sample_mask == c for c in face_hair_masking_channels]
# sample_face_hair_mask shape (len(face_hair_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
sample_face_hair_mask = tf.reduce_any(sample_face_hair_mask, axis=0)
sample_face_hair_mask = tf.cast(sample_face_hair_mask, dtype=tf.float32)
# show_img(sample_face_hair_mask)

sample_face_hair = sample_img * sample_face_hair_mask
show_img(sample_face_hair)

# Take everything except for the background, face, and hair
clothing_masking_channels = [5, 6, 7, 12]
sample_clothing_mask = [sample_mask == c for c in clothing_masking_channels]
# sample_clothing_mask shape (len(face_hair_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
sample_clothing_mask = tf.reduce_any(sample_clothing_mask, axis=0)
sample_clothing_mask = tf.cast(sample_clothing_mask, dtype=tf.float32)
show_img(sample_clothing_mask)

# %%

# After sampling data, we now can build the dataset

def train_generator():
    for (idx, [img_path, cloth_path]) in enumerate(TRAIN_PATH):
        sample_cloth = tf.image.resize(
            np.asarray(
                Image.open(
                    cloth_path
                )
            ), IMG_SHAPE[:2]
        ) / 255.0

        # sample_img shape (256, 192, 3). Range [0, 1]
        sample_img = tf.image.resize(
            np.asarray(
                Image.open(
                    img_path
                )
            ), IMG_SHAPE[:2]
        ) / 255.0

        # sample_pose shape (256, 192, 3). Range [0, 1]
        sample_pose = get_pose_map(img_path)

        # sample_pose shape (256, 192, 20). Range [0, 1]
        sample_parsing = get_human_parsing(sample_img)
        # sample_body_mask shape (256, 192, 1). Range [0, 19]. Representing classes.
        sample_mask = create_mask(sample_parsing)

        # Take everything except for the background, face, and hair
        body_masking_channels = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
        sample_body_mask = [sample_mask == c for c in body_masking_channels]
        # sample_body_mask shape (len(body_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
        sample_body_mask = tf.reduce_any(sample_body_mask, axis=0)
        sample_body_mask = tf.cast(sample_body_mask, dtype=tf.float32)

        # Take everything except for the background, face, and hair
        face_hair_masking_channels = [1, 2, 13]
        sample_face_hair_mask = [sample_mask == c for c in face_hair_masking_channels]
        # sample_face_hair_mask shape (len(face_hair_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
        sample_face_hair_mask = tf.reduce_any(sample_face_hair_mask, axis=0)
        sample_face_hair_mask = tf.cast(sample_face_hair_mask, dtype=tf.float32)
        # show_img(sample_face_hair_mask)

        sample_face_hair = sample_img * sample_face_hair_mask

        # Take everything except for the background, face, and hair
        clothing_masking_channels = [5, 6, 7, 12]
        sample_clothing_mask = [sample_mask == c for c in clothing_masking_channels]
        # sample_clothing_mask shape (len(face_hair_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
        sample_clothing_mask = tf.reduce_any(sample_clothing_mask, axis=0)
        sample_clothing_mask = tf.cast(sample_clothing_mask, dtype=tf.float32)
        
        yield tf.concat([sample_pose, sample_body_mask, sample_face_hair, sample_cloth], axis=2), \
            tf.concat([sample_img, sample_clothing_mask], axis=2)

train_ds = tf.data.Dataset.from_generator(
    train_generator,
    output_signature=(
        tf.TensorSpec(shape=(*IMG_SHAPE[:2], 10), dtype=tf.float32),
        tf.TensorSpec(shape=(*IMG_SHAPE[:2], 4), dtype=tf.float32)
    )
)
train_batch_ds = train_ds.batch(BATCH_SIZE)
it = iter(train_ds)
# %%

# test dataset
sample_input, sample_output = next(it)
print(sample_input.shape)
print(sample_output.shape)

# %%



# %%
# from shutil import copyfile
# train_path_raw, test_path_raw = get_data_path_raw()

# # Copy the segmentation model to input folder
# r_str = r"\/.*\.jpg$"

# for i, line in enumerate(train_path_raw):
#     # each line is 2 links, one is the person, 1 ius the cloth
#     # copy only th cloth to the input folder to segment.
#     url = line[0]
#     name_list = re.findall(r_str, url)
#     if len(name_list) == 0:
#         print(url)
#     else:
#         name = name_list[0][1:]
#         copyfile(DATASET_SRC / url, Path("./reference/inputs") / name)

# for i, line in enumerate(test_path_raw):
#     # each line is 2 links, one is the person, 1 ius the cloth
#     # copy only th cloth to the input folder to segment.
#     url = line[0]
#     name_list = re.findall(r_str, url)
#     if len(name_list) == 0:
#         print(url)
#     else:
#         name = name_list[0][1:]
#         copyfile(DATASET_SRC / url, Path("./reference/inputs") / name)

# %%

# Test Human pose




# %%

# Personal representation:
#   - Pose heatmap (18 channels) Check (3 channels)
#   - Human segmentation (1 channel) Check (3 channels)
#   - Face and hair segmentation (3 channels). Human parser prediction
# 

# - Labels:
#       - Clothing mask: human parser prediction.
#       - original person with clothes image.

# - Predict:
#       - The course agostic of the person.
#       - The clothing mask.


# %%
 
 # This is the U-net model

"""
Read this u-net article with the cute meow:
    https://towardsdatascience.com/u-net-b229b32b4a71
"""

from tensorflow_examples.models.pix2pix import pix2pix
mobile_net_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, 
    include_top=False)
mobile_net_model.trainable = False
# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 128x96
    'block_3_expand_relu',   # 64x48
    'block_6_expand_relu',   # 32x24
    'block_13_expand_relu',  # 16x12
    'block_16_project',      # 8x6
]
layers = [mobile_net_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
wrap_mobile_net_model = tf.keras.Model(inputs=mobile_net_model.input, outputs=layers)
wrap_mobile_net_model.trainable = False


inputs = tf.keras.Input(shape=(*IMG_SHAPE[:2], 7))

pre_conv = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(inputs)

out4, out3, out2, out1, out0 = wrap_mobile_net_model(pre_conv, training=False)

up1_tensor = pix2pix.upsample(512, 3)(out0)

cat1_tensor = tf.keras.layers.concatenate([up1_tensor, out1])
up2_tensor = pix2pix.upsample(256, 3)(cat1_tensor)

cat2_tensor = tf.keras.layers.concatenate([up2_tensor, out2])
up3_tensor = pix2pix.upsample(128, 3)(cat2_tensor)

cat3_tensor = tf.keras.layers.concatenate([up3_tensor, out3])
up4_tensor = pix2pix.upsample(64, 3)(cat3_tensor)

cat4_tensor = tf.keras.layers.concatenate([up4_tensor, out4])

# n channels (or neurons, or feature vectors) is 4 because we are predicting 2 things:
#       - course human image
#       - clothing mask on the person

out = tf.keras.layers.Conv2DTranspose(
    4, 3, strides=2,
    padding='same',
    activation='sigmoid'
) (cat4_tensor)

# We will not use model, we will just use it to see the summary!
model = tf.keras.Model(inputs, out)
model.summary()

# %%

# Definition of losses and train step

def loss_function():
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    pass

optimizer = tf.keras.optimizers.Adam(lr=2e-3)

def train_step(person_reprs, clothings, labels):
    # Use gradient tape
    pass

# %%

# Training the model

EPOCHS = 1

for epoch in EPOCHS:
    # Train train train
    pass


