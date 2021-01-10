#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College
"""

from .lip_dataset import *
from shutil import copyfile

train_path_raw, test_path_raw = get_data_path_raw()

# Copy the segmentation model to input folder
r_str = r"\/.*\.jpg$"

for i, line in enumerate(train_path_raw):
    # each line is 2 links, one is the person, 1 ius the cloth
    # copy only th cloth to the input folder to segment.
    url = line[0]
    name_list = re.findall(r_str, url)
    if len(name_list) == 0:
        print(url)
    else:
        name = name_list[0][1:]
        copyfile(DATASET_SRC / url, Path("./reference/inputs") / name)

for i, line in enumerate(test_path_raw):
    # each line is 2 links, one is the person, 1 ius the cloth
    # copy only th cloth to the input folder to segment.
    url = line[0]
    name_list = re.findall(r_str, url)
    if len(name_list) == 0:
        print(url)
    else:
        name = name_list[0][1:]
        copyfile(DATASET_SRC / url, Path("./reference/inputs") / name)