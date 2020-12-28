
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
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2

import mediapipe as mp

from .utils import *

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

# This is the U-net model

def get_wrap_vgg16_model(IMG_SHAPE=(256, 192, 3)):
    """
    Read this u-net article with the cute meow:
        https://towardsdatascience.com/u-net-b229b32b4a71
    """

    
    mobile_net_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE, 
        include_top=False)
    # mobile_net_model.summary()
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


    # inputs_img = tf.keras.Input(shape=(*IMG_SHAPE[:2], 7), name="inputs_img")

    inputs_cloth = tf.keras.Input(shape=(*IMG_SHAPE[:2], 3), name="inputs_cloth")
    # input_concat = tf.concat([inputs_img, inputs_cloth], axis=-1)
    # pre_conv = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(input_concat)

    out4, out3, out2, out1, out0 = wrap_mobile_net_model(inputs_cloth, training=False)

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

    # We don't use activation because we have to calculate mse, or we can use relu act
    out1 = tf.keras.layers.Conv2DTranspose(
        3, 3, strides=2,
        padding='same',
        activation='relu'
    ) (cat4_tensor)

    # out2 = tf.keras.layers.Conv2DTranspose(
    #     1, 3, strides=2,
    #     padding='same',
    #     activation='relu'
    # ) (cat4_tensor)

    # We will not use model, we will just use it to see the summary!
    # model = tf.keras.Model([inputs_img, inputs_cloth], [out1, out2])
    mask_model = tf.keras.Model(inputs_cloth, out1)
    # mask_model.summary()
    return mask_model


# Normal 2D auto encoder model with u net

# Build a simple Convolutional Autoencoder model. Don't use this model tho
# The reason why this model is deprecated is it's output shape is 3 dimesions
# so that we can just predict according to parsing 1 or 0 using binary crossentropy loss.
# Please use the simple parsing dataset for this model.

# inputs = tf.keras.layers.Input(shape=(*IMG_SHAPE[:2], 10))
def get_doubly_model(IMG_SHAPE=(256, 192, 3)):
    inputs_pose = tf.keras.Input(shape=(*IMG_SHAPE[:2], 3), name="inputs_pose")
    inputs_body_mask = tf.keras.Input(shape=(*IMG_SHAPE[:2], 1), name="inputs_body_mask")
    inputs_face_hair = tf.keras.Input(shape=(*IMG_SHAPE[:2], 3), name="inputs_face_hair")
    inputs_cloth = tf.keras.Input(shape=(*IMG_SHAPE[:2], 3), name="inputs_cloth")

    inputs_concat = tf.concat([inputs_pose, inputs_body_mask, inputs_face_hair, inputs_cloth], axis=-1)

    x = tf.keras.layers.Conv2D(
        filters=64, 
        kernel_size=(4, 4), 
        activation='relu', 
        padding='same'
    ) (inputs_concat)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    ) (x)

    x = tf.keras.layers.Conv2D(
        filters=128, 
        kernel_size=(4, 4), 
        activation='relu', 
        padding='same') (x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    ) (x)


    x = tf.keras.layers.Conv2D(
        filters=256, 
        kernel_size=(4, 4), 
        activation='relu', 
        padding='same') (x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    ) (x)


    x = tf.keras.layers.Conv2DTranspose(
        128, 3, strides=2,
        padding='same',
        activation='relu'
    ) (x)

    x = tf.keras.layers.Conv2DTranspose(
        64, 3, strides=2,
        padding='same',
        activation='relu'
    ) (x)

    # outputs = tf.keras.layers.Conv2D(
    #     filters=4, 
    #     kernel_size=(4, 4), 
    #     activation='relu',
    #     padding='same') (x)


    out1 = tf.keras.layers.Conv2DTranspose(
        3, 3, strides=2,
        padding='same',
        activation='relu'
    ) (x)

    # out2 = tf.keras.layers.Conv2DTranspose(
    #     1, 3, strides=2,
    #     padding='same',
    #     activation='relu'
    # ) (x)

    pre_out2 = tf.keras.layers.Conv2D(3, (4,4), padding='same')(inputs_concat)

    # out2 = mask_model(pre_out2)
    out2 = get_wrap_vgg16_model()(pre_out2)

    # We will not use model, we will just use it to see the summary!
    model = tf.keras.Model(
        [inputs_pose, inputs_body_mask, inputs_face_hair, inputs_cloth], 
        [out1, out2]
    )
    # model.summary()

    # model = tf.keras.Model(inputs, outputs)
    # model.summary()

    return model

# Deprecated model
def get_simple_unet_model(IMG_SHAPE=(256, 192, 3)):

    # Simple U-net architecture

    inputs_pose = tf.keras.Input(shape=(*IMG_SHAPE[:2], 3), name="inputs_pose")
    inputs_body_mask = tf.keras.Input(shape=(*IMG_SHAPE[:2], 1), name="inputs_body_mask")
    inputs_face_hair = tf.keras.Input(shape=(*IMG_SHAPE[:2], 3), name="inputs_face_hair")
    inputs_cloth = tf.keras.Input(shape=(*IMG_SHAPE[:2], 3), name="inputs_cloth")

    inputs_concat = tf.concat(
        [inputs_pose, inputs_body_mask, inputs_face_hair, inputs_cloth], 
        axis=-1
    )

    encoder1 = conv(inputs_concat, 32) # 256 x 192 x 32
    pool1 = max_pool(encoder1) # 128 x 96 x 32

    encoder2 = conv(pool1, 64) # 128 x 96 x 64
    encoder2_res = conv(encoder2, 64) # 128 x 96 x 64
    cat2 = tf.concat([encoder2, encoder2_res], axis=-1)
    pool2 = max_pool(cat2) # 64 x 48 x 64
    
    encoder3 = conv(pool2, 64) # 128 x 96 x 64
    encoder3_res = conv(encoder3, 64) # 128 x 96 x 64
    cat3 = tf.concat([encoder3, encoder3_res], axis=-1)
    pool3 = max_pool(cat3) # 64 x 48 x 64

    encoder4 = conv(pool3, 64) # 128 x 96 x 64
    encoder4_res = conv(encoder4, 64) # 128 x 96 x 64
    cat4 = tf.concat([encoder4, encoder4_res], axis=-1)
    pool4 = max_pool(cat4) # 64 x 48 x 64

    encoder5 = conv(pool4, 64) # 128 x 96 x 64
    encoder5_res = conv(encoder5, 64) # 128 x 96 x 64
    out_encoder = tf.concat([encoder5, encoder5_res], axis=-1)
    # pool5 = max_pool(tf.concat([encoder5, encoder5_res], axis=-1)) # 64 x 48 x 64
    
    # encoder3 = conv(pool2, 128) # 64 x 48 x 128
    # pool3 = max_pool(encoder3) # 32 x 24 x 128
    
    # encoder4 = conv(pool3, 256) # 32 x 24 x 256
    # pool4 = max_pool(encoder4) # 16 x 12 x 256

    # encoder5 = conv(pool4, 512) # 16 x 12 x 512
    # pool5 = max_pool(encoder5) # 8 x 6 x 512

    # up1_tensor = pix2pix.upsample(512, 4)(pool5) # 16 x 12 x 512

    # cat1_tensor = tf.keras.layers.concatenate([up1_tensor, encoder5]) # 16 x 12 x 512
    up2_tensor = pix2pix.upsample(256, 4)(out_encoder)  # 32 x 24 x 256

    cat2_tensor = tf.keras.layers.concatenate([up2_tensor, encoder4])  # 32 x 24 x 256
    up3_tensor = pix2pix.upsample(128, 4)(cat2_tensor)  # 64 x 48 x 128

    cat3_tensor = tf.keras.layers.concatenate([up3_tensor, encoder3]) # 64 x 48 x 128
    up4_tensor = pix2pix.upsample(64, 4)(cat3_tensor) # 128 x 96 x 64

    cat4_tensor = tf.keras.layers.concatenate([up4_tensor, encoder2]) # 128 x 96 x 64
    # up5_tensor = pix2pix.upsample(32, 4)(cat4_tensor) 

    # cat5_tensor = tf.keras.layers.concatenate([up5_tensor, encoder1])

    out1 = deconv(cat4_tensor, 32)
    out1 = final_conv(out1, 3)
    out2 = final_deconv(cat4_tensor, 1)
    # out2 = final_conv(out2, 1)

    model = tf.keras.Model(
        [inputs_pose, inputs_body_mask, inputs_face_hair, inputs_cloth], 
        [out1, out2]
    )

    return model


def get_res_unet_model():

    inputs_pose = tf.keras.Input(shape=(*IMG_SHAPE[:2], 3), name="inputs_pose")
    inputs_body_mask = tf.keras.Input(shape=(*IMG_SHAPE[:2], 1), name="inputs_body_mask")
    inputs_face_hair = tf.keras.Input(shape=(*IMG_SHAPE[:2], 3), name="inputs_face_hair")
    inputs_cloth = tf.keras.Input(shape=(*IMG_SHAPE[:2], 3), name="inputs_cloth")

    inputs_concat = tf.concat(
        [inputs_pose, inputs_body_mask, inputs_face_hair, inputs_cloth], 
        axis=-1
    )

    encoder1 = conv(inputs_concat, 32) # 256 x 192 x 32
    pool1 = max_pool(encoder1) # 128 x 96 x 32

    encoder2 = conv(pool1, 64) # 128 x 96 x 64
    encoder2_res = conv(encoder2, 64) # 128 x 96 x 64
    cat2 = tf.concat([encoder2, encoder2_res], axis=-1)
    pool2 = max_pool(cat2) # 64 x 48 x 64
    pool2 = dropout(pool2)

    encoder3 = conv(pool2, 128) # 128 x 96 x 64
    encoder3_res = conv(encoder3, 128) # 128 x 96 x 64
    cat3 = tf.concat([encoder3, encoder3_res], axis=-1)
    pool3 = max_pool(cat3) # 64 x 48 x 64
    pool3 = dropout(pool3)

    encoder4 = conv(pool3, 256) # 128 x 96 x 64
    encoder4_res = conv(encoder4, 256) # 128 x 96 x 64
    cat4 = tf.concat([encoder4, encoder4_res], axis=-1)
    pool4 = max_pool(cat4) # 64 x 48 x 64
    pool4 = dropout(pool4)

    encoder5 = conv(pool4, 512) # 128 x 96 x 64
    encoder5_res = conv(encoder5, 512) # 128 x 96 x 64
    out_encoder = tf.concat([encoder5, encoder5_res], axis=-1)
    
    bottleneck = conv(out_encoder, 1024)

    up2_tensor = pix2pix.upsample(256, 4)(bottleneck)  # 32 x 24 x 256
    up2_tensor = dropout(up2_tensor)

    cat2_tensor = tf.keras.layers.concatenate([up2_tensor, encoder4_res])  # 32 x 24 x 256
    cat2_tensor = conv(cat2_tensor, 512)
    
    up3_tensor = pix2pix.upsample(128, 4)(cat2_tensor)  # 64 x 48 x 128
    up3_tensor = dropout(up3_tensor)

    cat3_tensor = tf.keras.layers.concatenate([up3_tensor, encoder3_res]) # 64 x 48 x 128
    cat3_tensor = conv(cat3_tensor, 256)

    up4_tensor = pix2pix.upsample(64, 4)(cat3_tensor) # 128 x 96 x 64
    up4_tensor = dropout(up4_tensor)

    cat4_tensor = tf.keras.layers.concatenate([up4_tensor, encoder2_res]) # 128 x 96 x 64
    cat4_tensor = conv(cat4_tensor, 128)
    

    outputs = deconv(cat4_tensor, 32)
    outputs = conv(outputs, 32)
    outputs1 = conv(outputs, 3, 'tanh')
    outputs2 = conv(outputs, 2, 'sigmoid')
    # out2 = final_conv(out2, 1)

    model = tf.keras.Model(
        [inputs_pose, inputs_body_mask, inputs_face_hair, inputs_cloth], 
        [outputs1, outputs2]
    )

    return model


def get_very_simple_unet_model():

    inputs_pose = tf.keras.Input(shape=(*IMG_SHAPE[:2], 3), name="inputs_pose")
    inputs_body_mask = tf.keras.Input(shape=(*IMG_SHAPE[:2], 1), name="inputs_body_mask")
    inputs_face_hair = tf.keras.Input(shape=(*IMG_SHAPE[:2], 3), name="inputs_face_hair")
    inputs_cloth = tf.keras.Input(shape=(*IMG_SHAPE[:2], 3), name="inputs_cloth")

    inputs_concat = tf.concat(
        [inputs_pose, inputs_body_mask, inputs_face_hair, inputs_cloth], 
        axis=-1
    )

    encoder1 = conv(inputs_concat, 32) # 256 x 192 x 32
    pool1 = max_pool(encoder1) # 128 x 96 x 32

    encoder2 = conv(pool1, 64) # 128 x 96 x 64
    pool2 = max_pool(encoder2) # 64 x 48 x 64
    pool2 = dropout(pool2)

    encoder3 = conv(pool2, 128) # 64 x 48 x 128
    pool3 = max_pool(encoder3) # 32 x 24 x 128
    pool3 = dropout(pool3)

    encoder4 = conv(pool3, 256) # 32 x 24 x 256
    pool4 = max_pool(encoder4) # 16 x 12 x 256
    pool4 = dropout(pool4)

    encoder5 = conv(pool4, 512) # 16 x 12 x 512
    pool5 = max_pool(encoder5) # 8 x 6 x 512
    pool5 = dropout(pool5)

    up1_tensor = pix2pix.upsample(512, 4)(pool5)  # 16 x 12 x 512
    up1_tensor = dropout(up1_tensor)

    up2_tensor = pix2pix.upsample(256, 4)(tf.concat([
        up1_tensor, encoder5 # each 16 x 12 x 512
    ], axis=-1))  # 32 x 24 x 256
    up2_tensor = dropout(up2_tensor)

    up3_tensor = pix2pix.upsample(128, 4)(tf.concat([
        up2_tensor, encoder4 # each 32 x 24 x 256
    ], axis=-1))  # 64 x 48 x 128
    up3_tensor = dropout(up3_tensor)

    up4_tensor = pix2pix.upsample(64, 4)(tf.concat([
        up3_tensor, encoder3 # Each 64 x 48 x 128
    ], axis=-1))  # 128 x 96 x 64
    up4_tensor = dropout(up4_tensor)

    outputs1 = deconv(up4_tensor, 3, activation='tanh')
    outputs2 = deconv(up4_tensor, 2, activation='sigmoid')
    # out2 = final_conv(out2, 1)

    model = tf.keras.Model(
        [inputs_pose, inputs_body_mask, inputs_face_hair, inputs_cloth], 
        [outputs1, outputs2]
    )

    return model