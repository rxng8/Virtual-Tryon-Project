#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College
"""

# %%

"""
This notebook is for the atr dataset training.

"""

from PIL import Image
import tensorflow as tf
import random
import os
from pathlib import Path
import numpy as np


from core.dataset import ATRDataset
from core.utils import show_img, deprocess_img, plot_parsing_map, create_mask, preprocess_image
from core.models import get_unet_model_for_human_parsing

# Config
IMG_SHAPE = (256, 192, 3)
BATCH_SIZE = 16
OUTPUT_CHANNELS = len(ATRDataset.LABEL2INT)
n_epochs = 2
# %%

# At 6 epoch the model achieve its best. Overfit when comes to epoch 8

checkpoint_path = "models/checkpoints/human_parse_unet_{epoch:04d}_epoch/\
human_parse_unet_{epoch:04d}_epoch.ckpt".format(epoch=6)
model_path = "models/human_parse_unet_{epoch:04d}_epoch".format(epoch=6)

# %%

# Set up pipeline

base = "./dataset/ICCV15_fashion_dataset(ATR)/humanparsing"
ds = ATRDataset(base)
steps_per_epoch = ds.steps_per_epoch
model = get_unet_model_for_human_parsing(IMG_SHAPE, OUTPUT_CHANNELS)
model.load_weights(checkpoint_path)
train_ds = ds.get_tf_train_batch_dataset()
test_ds = ds.get_tf_test_batch_dataset()
test_it = ds.get_tf_test_dataset_iter()

# %%

# Training and save models

history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch, 
    epochs=n_epochs,
    validation_data=test_ds,
    validation_steps=10
)

# %%

# # Save checkpoint
# model.save_weights(checkpoint_path)

# # Save model
# model.save(model_path)

# %%

# Load model

model = tf.keras.models.load_model(model_path)
model.summary()

# %%

def test_random_prediction(model, test_it, batch_size, label2int):
    img, label = next(test_it)
    prediction = model.predict(tf.expand_dims(img, axis=0))[0]
    print("Original image:")
    show_img(deprocess_img(img))
    print("Label")
    show_img(label)
    plot_parsing_map(prediction, label2int=label2int)
    mask = create_mask(prediction)
    show_img(mask)


# %%

test_random_prediction(model, test_it, BATCH_SIZE, ATRDataset.LABEL2INT)





# %%

"""
Next, we try to improve our models with this paper:
    https://arxiv.org/pdf/1910.09777v1.pdf
"""




# %%


# -------------------  Mass Build VITON Pipeline ----------------------#

# Mass Build VITON pipeline



# this is the path to the root of the dataset
LIP_DATASET_PATH = Path("./dataset/lip_mpv_dataset/")

# This is the path to the folder contain actual data
LIP_DATASET_SRC = LIP_DATASET_PATH / "MPV_192_256"

# This is the name of the file where it contains path to data
LIP_DATASET_FILE = "all_poseA_poseB_clothes.txt"

DATASET_OUT_PATH = LIP_DATASET_PATH / "preprocessed"

LABEL_NAME_LIST = ['body_mask', 'face_hair', 'clothing_mask']

LABEL_FOLDER_PATH = [DATASET_OUT_PATH / d for d in LABEL_NAME_LIST]
for i, name in enumerate(LABEL_FOLDER_PATH):
    if not os.path.exists(name):
        os.mkdir(name)

def get_data_path_raw():
    train_half_front = []
    test_half_front = []

    with open(LIP_DATASET_PATH / LIP_DATASET_FILE, 'r') as f:
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

    with open(LIP_DATASET_PATH / LIP_DATASET_FILE, 'r') as f:
        for line in f:
            elems = line.split("\t")
            assert len(elems) == 4, "Unexpected readline!"
            if "train" in line:
                if "person_half_front.jpg" in line and "cloth_front.jpg" in line:
                    tmp_person = ""
                    tmp_cloth = ""
                    for elem in elems:
                        if "person_half_front.jpg" in elem:
                            tmp_person = str(LIP_DATASET_SRC / elem)
                        if "cloth_front.jpg" in elem:
                            tmp_cloth = str(LIP_DATASET_SRC / elem)
                    train_half_front.append([tmp_person, tmp_cloth])
                else:
                    continue
            elif "test" in line:
                if "person_half_front.jpg" in line and "cloth_front.jpg" in line:
                    tmp_person = ""
                    tmp_cloth = ""
                    for elem in elems:
                        if "person_half_front.jpg" in elem:
                            tmp_person = str(LIP_DATASET_SRC / elem)
                        if "cloth_front.jpg" in elem:
                            tmp_cloth = str(LIP_DATASET_SRC / elem)
                    test_half_front.append([tmp_person, tmp_cloth])
                else:
                    continue
            else:
                print("Unexpected behavior!")

    return np.asarray(train_half_front), np.asarray(test_half_front)

TRAIN_PATH, TEST_PATH = get_data_path_raw()

LABEL_NAME = ATRDataset.LABEL2INT

def get_human_parsing(img):
    assert img.shape == IMG_SHAPE, "Wrong image shape"
    prediction = model.predict(tf.expand_dims(img, axis=0))[0]
    return prediction

def random_prediction():
    r = random.randint(0, TRAIN_PATH.shape[0] - 1)
    img = np.asarray(
        Image.open(
            LIP_DATASET_SRC / TRAIN_PATH[r, 0]
        )
    )
    print("Original:")
    show_img(img)
    rescaled_img = preprocess_image(img)
    test_predict = model.predict(tf.expand_dims(rescaled_img, axis=0))
    plot_parsing_map(test_predict, label2int=ATRDataset.LABEL2INT)
    pred_mask = create_mask(test_predict)[0]
    show_img(pred_mask)

# %%

# LIP Random prediction

random_prediction()


# %%
""" Write files

Writw a lot of files

"""

import re
from tqdm import tqdm
import time

r_str = r"\/.*\.jpg$"
with tqdm(total=TRAIN_PATH.shape[0]) as pbar:
    for (idx, [img_path, cloth_path]) in enumerate(TRAIN_PATH):

        # time.sleep(0.1)

        # Eg: img path is raw. I.e: does not contains the full path (dataset/lip_mpv_dataset/preprocessed)
        # img_path: RE321D05M\RE321D05M-Q11@8=person_half_front.jpg
        # img_path_name: RE321D05M-Q11@8=person_half_front.jpg
        # img_folder_name: RE321D05M
        img_path_name = re.findall(r_str, img_path)[0]
        img_folder_name = img_path[:(len(img_path) - len(img_path_name))]
        cloth_path_name = re.findall(r_str, cloth_path)[0]
        cloth_folder_name = cloth_path[:(len(cloth_path) - len(cloth_path_name))]

        assert img_folder_name == cloth_folder_name, "Unexpected behavior"

        for i, name in enumerate(LABEL_FOLDER_PATH):
            if not os.path.exists(name / img_folder_name):
                os.mkdir(name / img_folder_name)
            if not os.path.exists(name / cloth_folder_name):
                os.mkdir(name / cloth_folder_name)

        sample_cloth = tf.image.resize(
            np.asarray(
                Image.open(
                    LIP_DATASET_SRC / cloth_path
                )
            ), IMG_SHAPE[:2]
        ) / 255.0

        # sample_img shape (256, 192, 3). Range [0, 1]
        sample_img = tf.image.resize(
            np.asarray(
                Image.open(
                    LIP_DATASET_SRC / img_path
                )
            ), IMG_SHAPE[:2]
        ) / 255.0

        # sample_pose shape (256, 192, 20). Range [0, 1]
        sample_parsing = get_human_parsing(sample_img)
        # sample_body_mask shape (256, 192, 1). Range [0, 19]. Representing classes.
        sample_mask = create_mask(sample_parsing)

        if not os.path.exists(LABEL_FOLDER_PATH[0] / img_path):
            body_masking_channels = [4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
            sample_body_mask = [sample_mask == c for c in body_masking_channels]
            # sample_body_mask shape (len(body_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
            sample_body_mask = tf.reduce_any(sample_body_mask, axis=0)
            sample_body_mask = tf.cast(sample_body_mask, dtype=tf.float32)
            # print(sample_body_mask.shape)
            # show_img(sample_body_mask)
            # print(os.path.join(LABEL_FOLDER_PATH[0], img_path_name))
            tf.keras.preprocessing.image.save_img(
                LABEL_FOLDER_PATH[0] / img_path, sample_body_mask
            )

        if not os.path.exists(LABEL_FOLDER_PATH[1] / img_path):
            face_hair_masking_channels = [1, 2, 3, 11]
            sample_face_hair_mask = [sample_mask == c for c in face_hair_masking_channels]
            # sample_face_hair_mask shape (len(face_hair_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
            sample_face_hair_mask = tf.reduce_any(sample_face_hair_mask, axis=0)
            sample_face_hair_mask = tf.cast(sample_face_hair_mask, dtype=tf.float32)

            sample_face_hair = sample_img * sample_face_hair_mask
            # print(sample_face_hair.shape)
            # show_img(sample_face_hair)
            tf.keras.preprocessing.image.save_img(
                LABEL_FOLDER_PATH[1] / img_path, sample_face_hair
            )

        if not os.path.exists(LABEL_FOLDER_PATH[2] / img_path):
            # Take everything except for the background, face, and hair
            clothing_masking_channels = [4, 7]
            sample_clothing_mask = [sample_mask == c for c in clothing_masking_channels]
            # sample_clothing_mask shape (len(face_hair_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
            sample_clothing_mask = tf.reduce_any(sample_clothing_mask, axis=0)
            sample_clothing_mask = tf.cast(sample_clothing_mask, dtype=tf.float32)
            # print(sample_clothing_mask.shape)
            # show_img(sample_clothing_mask)
            tf.keras.preprocessing.image.save_img(
                LABEL_FOLDER_PATH[2] / cloth_path, sample_clothing_mask
            )

        pbar.update(1)


