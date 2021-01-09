"""
"Finally, the experiments with existing methods reveal
that warped clothing is often severely distorted. We could
not clearly determine the reason, but can conclude that the
estimation of the TPS parameters needs regularization."
    From CP-VTON+ paper:
    https://minar09.github.io/cpvtonplus/cvprw20_cpvtonplus.pdf

Reference also this paper CP-VTON:
    https://arxiv.org/pdf/1807.07688.pdf

The preprocessed dataset is download here:
    https://github.com/sergeywong/cp-vton

Read about this paper about CNN in Geometric Matching, especially the "theta" param:
    https://arxiv.org/pdf/1703.05593.pdf

Read about Spatial Transformer Network
    https://arxiv.org/abs/1506.02025

Read about this STN tutorial to more understand the STN:
    https://kevinzakka.github.io/2017/01/10/stn-part1/
    https://kevinzakka.github.io/2017/01/18/stn-part2/

Implementation of STN on this git:
    https://github.com/kevinzakka/spatial-transformer-network

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
from core.utils import get_pose_map, show_img, preprocess_image

IMG_SHAPE = (256, 192, 3)
BATCH_SIZE = 16


base = "./dataset/lip_mpv_dataset/preprocessed/viton_preprocessed"

ds = VtonPrep(base)
# ds.visualize_random_data()


# %%

tfds_train = ds.get_tf_train_dataset()
tfds_train = tfds_train.batch(BATCH_SIZE).repeat()

# %%



# %%


def step(batch_data):
    pass


def train(n_epochs):
    pass


# %%


N_EPOCHS = 1

train(N_EPOCHS)




