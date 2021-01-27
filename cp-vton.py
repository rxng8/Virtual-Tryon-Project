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

Implementation tutorial of STN using pytorch:
    https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
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
from core.network.gmm import GMMTPS, SimpleGMM, GMMSTN

from IPython import display

IMG_SHAPE = (256, 192, 3)
BATCH_SIZE = 2


base = "./dataset/lip_mpv_dataset/preprocessed/viton_preprocessed"

ds = VtonPrep(base)
# ds.visualize_random_data()


# %%

tfds_train = ds.get_tf_train_dataset()
steps_per_epoch = 1000 // BATCH_SIZE
tfds_train = tfds_train.batch(BATCH_SIZE).repeat()


# %%

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.999)

# def l1_loss(real, pred):
#     return tf.reduce_sum(tf.keras.losses.MAE(real, pred))

def l1_loss(real, pred):
    return tf.keras.losses.MeanAbsoluteError()(real, pred)

def bin_cross_loss(real, pred):
    return tf.keras.losses.BinaryCrossentropy()(real, pred)

# @tf.function
def train_step(model, data, epoch, step):
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
            theta, grid, output = model(agnostic, data['cloth-mask'], training=True)

            # Compute the loss value for this minibatch.
            loss = l1_loss(data['actual-cloth-mask'], output[:,:,:,0])
            # loss = bin_cross_loss(data['actual-cloth-mask'], output)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # For visualization
        if step % 20 == 0:
            display.clear_output(wait=True)
            print(f"Epoch {epoch + 1}, step {step + 1}:")
            # print("Input cloth:")
            # print("Predictions:")

            # show_img(data['body-mask'][0])
            # show_img(data['face-hair'][0])
            # show_img(data['pose'][0])
            # show_img(data['cloth-mask'][0])
            show_img(data['actual-cloth-mask'][0])
            show_img(output[0])
            print(f"loss for this batch at step: {step + 1}: {loss }")

    return output


    


# %%


N_EPOCHS = 1
# steps_per_epoch = 1000
model = GMMTPS(batch_size=BATCH_SIZE)


# %%

with tf.device('/device:CPU:0'):
    for epoch in range(N_EPOCHS):
        print("\nStart of epoch %d" % (epoch + 1,))
        # Iterate over the batches of the dataset.
        for step, data in enumerate(tfds_train.take(10)):
            output = train_step(model, data, epoch, step)

        


# %%



data = list(tfds_train.take(1))[0]
agnostic = tf.concat([data['body-mask'], data['face-hair'], data['pose']], axis=3)
# Eval
with tf.device('/device:CPU:0'):
    theta, grid, output = model(agnostic, data['cloth-mask'], training=False)

# %%

show_img(output[0,:,:,:])


# %%


output[0,:,:,0]


# %%


a = tf.constant([[4,0],[9,6]])
# a [a==5] = 1

# %%

a + tf.cast(a == 0, a.dtype)
