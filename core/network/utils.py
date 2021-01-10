#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College
"""

from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow as tf

def make_dense_layer(out_channels, activation='relu'):
    return tf.keras.layers.Dense(out_channels, activation=activation)

def make_conv_layer(out_channels, strides=1, activation='relu', padding='same'):
    layer = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=(4, 4),
        activation=activation,
        padding=padding
    )
    return layer

def make_dropout_layer(rate=0.5):
    return tf.keras.layers.Dropout(rate)

def make_max_pooling_layer():
    return tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    )

def make_batch_norm_layer(**kwargs):
    return tf.keras.layers.BatchNormalization(**kwargs)

def make_down_conv_sequence(out_channels):
    return tf.keras.Sequential([
        make_conv_layer(out_channels),
        make_max_pooling_layer(),
        make_dropout_layer()
    ])


def conv(batch_input, out_channels, strides=1, activation='relu'):

    # padded_input = tf.pad(
    #     batch_input, 
    #     [[0, 0], [1, 1], [1, 1], [0, 0]], 
    #     mode="CONSTANT"
    # )

    out = tf.keras.layers.Conv2D(
        filters=out_channels, 
        kernel_size=(4, 4),
        activation=activation, 
        padding='same'
    )(batch_input)
    # print(out.shape)
    return out

def dropout(batch_input, rate=0.5):
    return tf.keras.layers.Dropout(
        rate
    ) (batch_input)

def max_pool(batch_input):
    return tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    ) (batch_input)

def final_conv(batch_input, out_channels):
    # out = tf.keras.layers.Conv2D(
    #     filters=out_channels, 
    #     kernel_size=(4, 4),
    #     activation='sigmoid', 
    #     padding='same'
    # )(batch_input)
    # return out
    return conv(batch_input, out_channels, activation='sigmoid')

def final_deconv(batch_input, out_channels):
    return tf.keras.layers.Conv2DTranspose(
        out_channels, 4, strides=2,
        padding='same',
        activation='sigmoid'
    ) (batch_input)

def deconv(batch_input, out_channels, activation='relu'):
    return tf.keras.layers.Conv2DTranspose(
        out_channels, 4, strides=2,
        padding='same',
        activation=activation
    ) (batch_input)

def upsampling(batch_input):
    pass