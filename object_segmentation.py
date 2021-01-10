#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  Author: Alex Nguyen
  Gettysburg College

Refer to this link:
    https://towardsdatascience.com/image-segmentation-with-six-lines-0f-code-acb870a462e8
"""
# %%

import pixellib
from pixellib.semantic import semantic_segmentation

from core.utils import *

segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("./models/object_segment/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5") 


# %%
test_path = "./dataset/lip_mpv_dataset/MPV_192_256/1LJ21D002/1LJ21D002-G11@10=cloth_front.jpg"
img = segment_image.segmentAsPascalvoc(test_path)
# %%

show_img(np.asarray(img[0]))
img[1].shape


# %%

# this is the path to the root of the dataset
LIP_DATASET_PATH = Path("./dataset/lip_mpv_dataset/")

# This is the path to the folder contain actual data
LIP_DATASET_SRC = LIP_DATASET_PATH / "MPV_192_256"

# This is the name of the file where it contains path to data
LIP_DATASET_FILE = "all_poseA_poseB_clothes.txt"

DATASET_OUT_PATH = LIP_DATASET_PATH / "preprocessed"

LABEL_NAME_LIST = ['cloth_mask']

LABEL_FOLDER_PATH = [DATASET_OUT_PATH / d for d in LABEL_NAME_LIST]
for i, name in enumerate(LABEL_FOLDER_PATH):
    if not os.path.exists(name):
        os.mkdir(name)

# print(LABEL_FOLDER_PATH[0])
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
        cloth_path_name = re.findall(r_str, cloth_path)[0]
        cloth_folder_name = cloth_path[:(len(cloth_path) - len(cloth_path_name))]

        # assert img_folder_name == cloth_folder_name, "Unexpected behavior"

        for i, name in enumerate(LABEL_FOLDER_PATH):
            if not os.path.exists(name / cloth_folder_name):
                os.mkdir(name / cloth_folder_name)

        if not os.path.exists(LABEL_FOLDER_PATH[0] / img_path):
            mask, _ = segment_image.segmentAsPascalvoc(LIP_DATASET_SRC / cloth_path)
            tf.keras.preprocessing.image.save_img(
                LABEL_FOLDER_PATH[0] / cloth_path, mask
            )
        pbar.update(1)
    


