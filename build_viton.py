#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College
"""

# %%

import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import os
from tqdm import tqdm
import time

from core.dataset import VtonPrep
from core.utils import get_pose_map, show_img 

IMG_SHAPE = (256, 192, 3)

base = "./dataset/lip_mpv_dataset/preprocessed/viton_preprocessed"
train_pose_dir = os.path.join(base, "train", "pose")
test_pose_dir = os.path.join(base, "test", "pose")
if not os.path.exists(train_pose_dir):
    os.mkdir(train_pose_dir)
if not os.path.exists(test_pose_dir):
    os.mkdir(test_pose_dir)

train_image_dir = os.path.join(base, "train", "image")
test_image_dir = os.path.join(base, "test", "image")

ds = VtonPrep(base)

# %%

train_names = ds.__get_image_name_only("train")
test_names = ds.__get_image_name_only("test")

def export_pose(file_names, input_dir_path, output_dir_path):
    with tqdm(total=file_names.shape[0]) as pbar:
        for img_path in file_names:
            src_file_path = os.path.join(input_dir_path, img_path)
            dest_file_path = os.path.join(output_dir_path, img_path)
            if not os.path.exists(dest_file_path):
                # sample_pose shape (256, 192, 3). Range [0, 1]
                sample_pose = get_pose_map(src_file_path)
                tf.keras.preprocessing.image.save_img(
                    dest_file_path, sample_pose
                )
                # show_img(sample_pose)
            pbar.update(1)
    time.sleep(0.1)
    print("Done!")

export_pose(train_names, train_image_dir, train_pose_dir)
export_pose(test_names, test_image_dir, test_pose_dir)

# %%

# rename utils: Rename every file to the same form, originally _1.jpg, now
# rename it to _0.jpg

def rename(src, dst):
    os.rename(src, dst)

def rename_all(dir_path):
    for img_name in os.listdir(dir_path):
        new_name = img_name[:7] + '0.jpg'
        rename(os.path.join(dir_path, img_name), os.path.join(dir_path, new_name))

dir_path_train_cloth = os.path.join(base, "train", "cloth")
dir_path_train_mask = os.path.join(base, "train", "cloth-mask")
dir_path_test_cloth = os.path.join(base, "test", "cloth")
dir_path_test_mask = os.path.join(base, "test", "cloth-mask")
rename_all(dir_path_train_cloth)
rename_all(dir_path_train_mask)
rename_all(dir_path_test_cloth)
rename_all(dir_path_test_mask)

