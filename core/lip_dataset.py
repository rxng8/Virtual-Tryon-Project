#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College
"""

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
from IPython import display

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2

import mediapipe as mp

from .utils import *
from .models import *

# config

# Please config the path as accurate as possible

# this is the path to the root of the dataset
DATASET_PATH = Path("./dataset/lip_mpv_dataset/")

# This is the path to the folder contain actual data
DATASET_SRC = DATASET_PATH / "MPV_192_256"

# This is the name of the file where it contains path to data
DATASET_FILE = "all_poseA_poseB_clothes.txt"

DATASET_OUT_PATH = DATASET_PATH / "preprocessed"

LABEL_NAME_LIST = ['body_mask', 'face_hair', 'clothing_mask', 'pose']

LABEL_FOLDER_PATH = [DATASET_OUT_PATH / d for d in LABEL_NAME_LIST]

BATCH_SIZE = 8

IMG_SHAPE = (256, 192, 3)

MASK_THRESHOLD = 0.9

# Deprecated approach
# load human parsing model that has been trained in human_parsing notebook.
# parsing_model = tf.keras.models.load_model('models/human_parsing_mbv2-50epochs')

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

TRAIN_PATH, TEST_PATH = get_data_path_raw()

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

def get_human_parsing(img):
    assert img.shape == IMG_SHAPE, "Wrong image shape"
    prediction = parsing_model.predict(tf.expand_dims(img, axis=0))[0]
    return prediction

def test():
    r = np.random.randint(0, TRAIN_PATH.shape[0] - 1)

    sample_cloth = preprocess_image(
        np.asarray(
            Image.open(
                DATASET_SRC / TRAIN_PATH[r,1]
            )
        ), IMG_SHAPE[:2]
    )
    print(f"Min val: {tf.reduce_min(sample_cloth)}, max val: {tf.reduce_max(sample_cloth)}")
    show_img(deprocess_img(sample_cloth))

    # sample_img shape (256, 192, 3). Range [0, 1]
    sample_img = preprocess_image(
        np.asarray(
            Image.open(
                    DATASET_SRC / TRAIN_PATH[r,0]
            )
        ), IMG_SHAPE[:2]
    )
    print(f"Min val: {tf.reduce_min(sample_img)}, max val: {tf.reduce_max(sample_img)}")
    show_img(deprocess_img(sample_img))

    # sample_pose shape (256, 192, 3). Range [0, 1].
    sample_pose = preprocess_image (
        np.asarray(Image.open(LABEL_FOLDER_PATH[3] / TRAIN_PATH[r,0])),
        IMG_SHAPE[:2]
    )
    print(f"Min val: {tf.reduce_min(sample_pose)}, max val: {tf.reduce_max(sample_pose)}")
    show_img(deprocess_img(sample_pose))

    # sample_body_mask shape (256, 192, 1).
    sample_body_mask =  preprocess_image(
        tf.expand_dims(np.asarray(Image.open(LABEL_FOLDER_PATH[0] / TRAIN_PATH[r,0])), axis=2),
        IMG_SHAPE[:2],
        resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        tanh_range=False
    )
    print(f"Min val: {tf.reduce_min(sample_body_mask)}, max val: {tf.reduce_max(sample_body_mask)}")
    show_img(deprocess_img(sample_body_mask))

    # sample_face_hair shape (256, 192, 1).
    sample_face_hair = preprocess_image(
        np.asarray(Image.open(LABEL_FOLDER_PATH[1] / TRAIN_PATH[r,0])),
        IMG_SHAPE[:2]
    )
    print(f"Min val: {tf.reduce_min(sample_face_hair)}, max val: {tf.reduce_max(sample_face_hair)}")
    show_img(deprocess_img(sample_face_hair))

    # sample_clothing_mask shape (256, 192, 1).
    sample_clothing_mask = preprocess_image(
        tf.expand_dims(np.asarray(Image.open(LABEL_FOLDER_PATH[2] / TRAIN_PATH[r,1])), axis=2),
        IMG_SHAPE[:2],
        resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        tanh_range=False
    )
    print(f"Min val: {tf.reduce_min(sample_clothing_mask)}, max val: {tf.reduce_max(sample_clothing_mask)}")
    show_img(deprocess_img(sample_clothing_mask))

# After sampling data, we now can build the dataset
# dEPRECATED GENErator
def train_generator_deprecated():
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

        body_masking_channels = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
        sample_body_mask = [sample_mask == c for c in body_masking_channels]
        # sample_body_mask shape (len(body_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
        sample_body_mask = tf.reduce_any(sample_body_mask, axis=0)
        sample_body_mask = tf.cast(sample_body_mask, dtype=tf.float32)

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

def train_generator():
    for (idx, [img_path, cloth_path]) in enumerate(TRAIN_PATH):
        sample_cloth = preprocess_image(
            np.asarray(
                Image.open(
                    DATASET_SRC / cloth_path
                )
            ), IMG_SHAPE[:2]
        )
        # show_img(sample_cloth)

        # sample_img shape (256, 192, 3). Range [0, 1]
        sample_img = preprocess_image(
            np.asarray(
                Image.open(
                     DATASET_SRC / img_path
                )
            ), IMG_SHAPE[:2]
        )
        # show_img(sample_img)

        # sample_pose shape (256, 192, 3). Range [0, 1].
        sample_pose = preprocess_image (
            np.asarray(Image.open(LABEL_FOLDER_PATH[3] / img_path)),
            IMG_SHAPE[:2]
        )
        if tf.reduce_max(sample_pose) == -1.0:
            continue
        # show_img(sample_pose)

        # sample_body_mask shape (256, 192, 1).
        sample_body_mask =  preprocess_image(
            tf.expand_dims(np.asarray(Image.open(LABEL_FOLDER_PATH[0] / img_path)), axis=2),
            IMG_SHAPE[:2],
            resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            tanh_range=False
        )
        # show_img(sample_body_mask)

        # sample_face_hair shape (256, 192, 1).
        sample_face_hair = preprocess_image(
            np.asarray(Image.open(LABEL_FOLDER_PATH[1] / img_path)),
            IMG_SHAPE[:2]
        )
        # show_img(sample_face_hair)

        # sample_clothing_mask shape (256, 192, 1).
        sample_clothing_mask = preprocess_image(
            tf.expand_dims(np.asarray(Image.open(LABEL_FOLDER_PATH[2] / cloth_path)), axis=2),
            IMG_SHAPE[:2],
            resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            tanh_range=False
        )
        # show_img(sample_clothing_mask)

        yield {
                "input_pose": sample_pose,
                "input_body_mask": sample_body_mask,
                "input_face_hair":sample_face_hair,
                "input_cloth": sample_cloth
            }, \
            {
                "output_image": sample_img,
                "output_cloth_mask": sample_clothing_mask
            }
