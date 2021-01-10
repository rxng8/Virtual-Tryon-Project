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
import cv2

import mediapipe as mp

from core.utils import *
from core.models import *
from lip_dataset import train_generator

# GPU config first
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        # Set auto scale memory
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Un comment this to manually set memory limit
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# %%

train_ds = tf.data.Dataset.from_generator(
    train_generator,
    output_signature=(
        {
            "input_pose": tf.TensorSpec(shape=(*IMG_SHAPE[:2], 3), dtype=tf.float32),
            "input_body_mask": tf.TensorSpec(shape=(*IMG_SHAPE[:2], 1), dtype=tf.float32),
            "input_face_hair": tf.TensorSpec(shape=(*IMG_SHAPE[:2], 3), dtype=tf.float32),
            "input_cloth": tf.TensorSpec(shape=(*IMG_SHAPE[:2], 3), dtype=tf.float32)
        }, 
        {
            "output_image": tf.TensorSpec(shape=(*IMG_SHAPE[:2], 3), dtype=tf.float32),
            "output_cloth_mask": tf.TensorSpec(shape=(*IMG_SHAPE[:2], 1), dtype=tf.float32)
        }
    )
)
train_batch_ds = train_ds.shuffle(1000).batch(BATCH_SIZE)
it = iter(train_ds)
it_batch = iter(train_batch_ds)

# %%

# test dataset
sample_input, sample_output = next(it)
print(sample_input["input_pose"].shape)
print(sample_input["input_body_mask"].shape)
print(sample_input["input_face_hair"].shape)
print(sample_input["input_cloth"].shape)
print(sample_output["output_image"].shape)
print(sample_output["output_cloth_mask"].shape)


# %%

vgg19 = tf.keras.applications.VGG19(
    include_top=False, 
    weights='imagenet',
    input_shape=IMG_SHAPE
)
vgg19.trainable = False
# vgg19.summary()
layer_names = [
    'block1_conv2', # 256 x 192 x 64
    'block2_conv2', # 128 x 96 x 128
    'block3_conv2', # 64 x 48 x 256
    'block4_conv2', # 32 x 24 x 512
    'block5_conv2'  # 16 x 12 x 512
]
layers = [vgg19.get_layer(name).output for name in layer_names]

# Create the feature extraction model
wrap_vgg19_model = tf.keras.Model(inputs=vgg19.input, outputs=layers)
wrap_vgg19_model.trainable = False

# Copied from xthan github
def compute_error(real, fake, mask=None):
    if mask == None:
        return tf.reduce_mean(tf.abs(fake - real))  # simple loss
    else:
        _, h, w, _ = real.get_shape().as_list()
        sampled_mask = tf.image.resize_images(mask, (h, w),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.reduce_mean(tf.abs(fake - real) * sampled_mask)  # simple loss

def loss_function(real, pred):
    # Read about perceptual loss here:
    # https://towardsdatascience.com/perceptual-losses-for-real-time-style-transfer-and-super-resolution-637b5d93fa6d
    # Also, tensorflow losses only compute loss across the last dimension. so we 
    # have to reduce mean to a constant

    # mask the real and pred first. TODO: Do we need to?
    # # Convert RGB to BGR. Deprecated because the image originally is converted
    # to BGR through preprocessing state
    # bgr_real = real[..., ::-1]
    # bgr_pred = pred[..., ::-1]

    # Perceptual loss eval
    out_real = wrap_vgg19_model(real, training=False)
    out_pred = wrap_vgg19_model(pred, training=False)

    # pixel-pise loss, RGB predicted value
    pixel_loss = compute_mse_loss(real, pred)

    # Perceptual loss
    # for real_features, pred_features in zip(out_real, out_pred):
    #     perceptual_loss += tf.reduce_mean(tf.math.abs(real_features - pred_features))
    # perceptual_loss /= len(out_real)
    # Compute perceptual loss manually
    p1 = compute_mse_loss(out_real[0], out_pred[0]) / 5.3 * 2.5
    p2 = compute_mse_loss(out_real[1], out_pred[1]) / 2.7  / 1.2
    p3 = compute_mse_loss(out_real[2], out_pred[2]) / 1.35 / 2.3
    p4 = compute_mse_loss(out_real[3], out_pred[3]) / 0.67 / 8.2
    p5 = compute_mse_loss(out_real[4], out_pred[4]) / 0.16 

    perceptual_loss = (p1 + p2 + p3 + p4 + p5)  / 5.0 / 128.0

    return 1.0 * pixel_loss + 3.0 * perceptual_loss

def mask_loss_function(real, pred):
    # L1 loss
    # return tf.reduce_mean(tf.keras.losses.BinaryCrossentropy()(real,pred))
    # return 1.0 * compute_mse_loss(real, pred)
    return 1.0 * tf.keras.losses.SparseCategoricalCrossentropy()(real, pred)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)

@tf.function
def train_step(x_batch_train, y_batch_train):
    with tf.device('/device:GPU:0'):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.

            # Name
            # "input_pose"
            # "input_body_mask"
            # "input_face_hair"
            # "input_cloth"
            # "output_image"
            # "output_cloth_mask"
            logits_human, logits_mask = model([x_batch_train], training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_function(y_batch_train["output_image"], logits_human)
            loss_mask_value = mask_loss_function(y_batch_train["output_cloth_mask"], logits_mask)
            loss = loss_value + loss_mask_value
            # loss = loss_value
            print(f"loss for this batch at step: {step + 1}: {loss }")

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return logits_human, logits_mask

# %%

model = get_very_simple_unet_model()

# %%

model.summary()
# Checkpoint path
checkpoint_path = "models/checkpoints/viton_12.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
if os.path.exists("models/checkpoints/viton_12.ckpt.index"):
    model.load_weights(checkpoint_path)
    print("Weights loaded!")


# %%

# Training the model

EPOCHS = 1
STEP_PER_EPOCHS = 100
# print(f"Dataset steps per epochs: {len(train_batch_ds) // BATCH_SIZE}")
with tf.device('/device:CPU:0'):
    for epoch in range(EPOCHS):
        print("\nStart of epoch %d" % (epoch + 1,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_batch_ds.take(STEP_PER_EPOCHS)):
            logits_human, logits_mask = train_step(x_batch_train, y_batch_train)
            if step % 20 == 0:
                display.clear_output(wait=True)
                print(f"Epoch {epoch + 1}, step {step + 1}:")
                print("Input cloth:")
                show_img(deprocess_img(x_batch_train["input_cloth"][0]))
                print("Predictions:")
                show_img(deprocess_img(logits_human[0]))
                show_img(create_mask(logits_mask[0]))
        # For each epoch, save checkpoint
        model.save_weights(checkpoint_path)
        print("Checkpoint saved!")
            # Log every 200 batches.
            # if step % 200 == 0:
            #     print(
            #         "Training loss (for one batch) at step %d: %.4f"
            #         % (step, float(loss_value))
            #     )
            #     print("Seen so far: %s samples" % ((step + 1) * 64))

# %%

model.save('models/viton-unet-30epochs')

# %%

# eval

sample_input_batch, sample_output_batch = next(it_batch)

#%%

# Evaluate
r = np.random.randint(0, BATCH_SIZE - 1)

print("Input: ")
show_img(deprocess_img(sample_input_batch["input_pose"][r]))
show_img(deprocess_img(sample_input_batch["input_body_mask"][r]))
show_img(deprocess_img(sample_input_batch["input_face_hair"][r]))
show_img(deprocess_img(sample_input_batch["input_cloth"][r]))

print('label:')
show_img(deprocess_img(sample_output_batch["output_image"][r]))
show_img(deprocess_img(sample_output_batch["output_cloth_mask"][r]))

print('pred:')
pred_img, pred_cloth = model([sample_input_batch])
show_img(deprocess_img(pred_img[r]))
show_img(create_mask(pred_cloth[r]))



