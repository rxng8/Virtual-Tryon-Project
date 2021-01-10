#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College

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
from core.network.gmm import GMM

from IPython import display

IMG_SHAPE = (256, 192, 3)
BATCH_SIZE = 16


base = "./dataset/lip_mpv_dataset/preprocessed/viton_preprocessed"

ds = VtonPrep(base)
# ds.visualize_random_data()


# %%

tfds_train = ds.get_tf_train_dataset()
steps_per_epoch = 10000 // BATCH_SIZE
tfds_train = tfds_train.batch(BATCH_SIZE).repeat()

# %%



# %%

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(model, data, step):
    with tf.device('/device:GPU:0'):
        with tf.GradientTape() as tape:
            # Prepare data

            # Each tensor has the shape (B, H, W, C)
            # 'image': sample_img,
            # 'cloth': sample_cloth,
            # 'cloth-mask': sample_cloth_mask,
            # 'body-mask': sample_body_mask,
            # 'face-hair': sample_face_hair,
            # 'actual-cloth-mask': sample_clothing_mask,
            # 'pose': sample_pose

            
            agnostic = tf.concat([data['body-mask'], data['face-hair'], data['pose']], axis=3)

            # Eval
            theta, grid, output = model(agnostic, data['cloth-mask'], training=False)

            # Compute the loss value for this minibatch.
            loss = None
            
            print(f"loss for this batch at step: {step + 1}: {loss }")

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        # grads = tape.gradient(loss, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        # optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return theta, grid, output

def train(model, n_epochs, steps_per_epoch):
    with tf.device('/device:CPU:0'):
        for epoch in range(n_epochs):
            print("\nStart of epoch %d" % (epoch + 1,))
            # Iterate over the batches of the dataset.
            for step, data in enumerate(tfds_train.take(steps_per_epoch)):
                theta, grid, output = train_step(model, data, step)

                # For visualization
                if step % 20 == 0:
                    display.clear_output(wait=True)
                    # print(f"Epoch {epoch + 1}, step {step + 1}:")
                    # print("Input cloth:")
                    # print("Predictions:")

                    show_img(output[0])

                break
            break


# %%


N_EPOCHS = 1
model = GMM()


# %%

train(model, N_EPOCHS, steps_per_epoch)




