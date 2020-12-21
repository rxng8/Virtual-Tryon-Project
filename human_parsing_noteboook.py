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
import random

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
# tf.config.experimental.set_visible_devices(devices=gpus[0], device_type="GPU")
# tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

# Please configure the path to the dataset as accurate as possible!

# This is the very parent path to the human parsing dataset.
DATASET_PATH = Path("./dataset/human_parsing/")

# This is the path to the text file containing path to data
TRAIN_IDX = DATASET_PATH / "TrainVal_images" / "train_id.txt"

# Get all name of the training path to the TRAIN_NAME list.
TRAIN_NAME = []
with open(TRAIN_IDX, "r") as f:
    for line in f:
        TRAIN_NAME.append(line.strip())

# This is the path to the text file containing path to data
VAL_IDX = DATASET_PATH / "TrainVal_images" / "val_id.txt"

# Get all name of the validation path to the TRAIN_NAME list.
# name is just the name of the image file without extension. E.g: 8123583, not 8123583.jpg
VAL_NAME = []
with open(VAL_IDX, "r") as f:
    for line in f:
        VAL_NAME.append(line.strip())

# Since input have extension .jpg and the parsing has extension .png
INPUT_EXT = ".jpg"
PARSING_EXT = ".png"

# These are the path to the folder containing all the image files
TRAIN_INPUT_PATH = DATASET_PATH / \
    "TrainVal_images" / \
    "TrainVal_images" / \
    "train_images"

VAL_INPUT_PATH = DATASET_PATH / \
    "TrainVal_images" / \
    "TrainVal_images" / \
    "val_images"

# These are the path to the folder containing all the label (parsing) files
TRAIN_PARSING_PATH = DATASET_PATH / \
    "TrainVal_parsing_annotations" / \
    "TrainVal_parsing_annotations" / \
    "train_segmentations"
VAL_PARSING_PATH = DATASET_PATH / \
    "TrainVal_parsing_annotations" / \
    "TrainVal_parsing_annotations" / \
    "val_segmentations"

# Label name, the actual label in the parsing image. note that it is 1 channel
LABEL_NAME = {
    'Hat': 1,
    'Hair': 2,
    'Glove': 3,
    'Sunglasses': 4,
    'UpperClothes': 5,
    'Dress': 6,
    'Coat': 7,
    'Socks': 8,
    'Pants': 9,
    'Jumpsuits': 10,
    'Scarf': 11,
    'Skirt': 12,
    'Face': 13,
    'Left-arm': 14,
    'Right-arm': 15,
    'Left-leg': 16,
    'Right-leg': 17,
    'Left-shoe': 18,
    'Right-shoe': 19
}

IMG_SHAPE = (300, 300, 3)
PARSING_SHAPE = (288, 288, 1)

def show_img(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# %%

r = np.random.randint(0, 5000)

sample_input = np.asarray(Image.open(TRAIN_INPUT_PATH / (TRAIN_NAME[r] + INPUT_EXT)))
sample_label = np.asarray(Image.open(TRAIN_PARSING_PATH / (TRAIN_NAME[r] + PARSING_EXT)))

print(f"Shape: {sample_input.shape}")
show_img(sample_input)
show_img(sample_label)


# %%

def train_gen():
    for file_name in TRAIN_NAME:
        img = tf.convert_to_tensor(
            np.asarray(
                Image.open(
                    TRAIN_INPUT_PATH / (file_name + INPUT_EXT)
                )
            ),
            dtype=tf.float32)
        if len(img.shape) != 3:
            continue
        resized_img = tf.keras.layers.experimental.preprocessing.Resizing(*IMG_SHAPE[:2])(img)
        rescaled_img = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(resized_img)
        label = \
            np.expand_dims(
                np.asarray(
                    Image.open(
                        TRAIN_PARSING_PATH / (file_name + PARSING_EXT)
                    ),
                    dtype=float
                ), axis=2
            )
        if len(label.shape) != 3:
            continue
        label[label > 0] = 1
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        resized_label = tf.keras.layers.experimental.preprocessing.Resizing(*PARSING_SHAPE[:2])(label)
        # rescaled_label = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(resized_label)
        yield rescaled_img, resized_label
        # yield rescaled_img, rescaled_label

def get_input_and_label(input_path, label_path):
    test_o = tf.convert_to_tensor(
        np.asarray(
            Image.open(
                input_path
            )
        )
    )
    test_o = tf.keras.layers.experimental.preprocessing.Resizing(*IMG_SHAPE[:2])(test_o)
    test_o = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(test_o)
    label = \
        np.expand_dims(
            np.asarray(
                Image.open(
                    label_path
                ),
                dtype=float
            ), axis=2
        )
    label[label > 0] = 1
    label = tf.convert_to_tensor(label, dtype=tf.float32)
    resized_label = tf.keras.layers.experimental.preprocessing.Resizing(*PARSING_SHAPE[:2])(label)
    return test_o, resized_label


# Tensorflow dataset object
ds = tf.data.Dataset.from_generator(train_gen, output_signature=(
    tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
    tf.TensorSpec(shape=PARSING_SHAPE, dtype=tf.float32)
))

# Generator object
train_gen_obs = train_gen()
it = iter(ds)
batchs = ds.repeat().batch(20)
batchs

# %%

# a, b = next(train_gen_obs)
a, b = next(it)
show_img(a)
show_img(b)


# %%

# Build a simple Convolutional Autoencoder model. Don't use this model tho

inputs = tf.keras.layers.Input(shape=(300, 300, 3))
x = tf.keras.layers.Conv2D(
    filters=40, 
    kernel_size=(3, 3), 
    activation='relu', 
    padding='same'
) (inputs)
x = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    padding='same'
) (x)

x = tf.keras.layers.Conv2D(
    filters=20, 
    kernel_size=(3, 3), 
    activation='relu', 
    padding='same') (x)
x = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    padding='same'
) (x)


x = tf.keras.layers.Conv2D(
    filters=20, 
    kernel_size=(3, 3), 
    activation='relu', 
    padding='same') (x)
x = tf.keras.layers.UpSampling2D(
    (2, 2)
) (x)


x = tf.keras.layers.Conv2D(
    filters=40, 
    kernel_size=(3, 3), 
    activation='relu', 
    padding='same') (x)
x = tf.keras.layers.UpSampling2D(
    (2, 2)
) (x)

outputs = tf.keras.layers.Conv2D(
    filters=1, 
    kernel_size=(3, 3), 
    activation='softmax',
    padding='same') (x)


model = tf.keras.Model(inputs, outputs)
model.compile(loss='categorical_crossentropy', metrics=[tf.keras.metrics.MeanIoU(2)])
model.summary()

# %%

# Build another model

from tensorflow_examples.models.pix2pix import pix2pix
vgg16_model = tf.keras.applications.VGG16(
    input_shape=(300, 300, 3),
    include_top=False,
    weights='imagenet'
)
vgg16_model.trainable = False
# vgg16_model.summary()

inputs = tf.keras.Input(shape=(300, 300, 3))
x = vgg16_model(inputs, training=False)

x = pix2pix.upsample(512, 3)(x)
x = pix2pix.upsample(256, 3)(x)
x = pix2pix.upsample(128, 3)(x)
x = pix2pix.upsample(64, 3)(x)
x = pix2pix.upsample(32, 3)(x)
x = pix2pix.upsample(16, 3)(x)

out = tf.keras.layers.Conv2D(
    1, 3, strides=2,
    padding='same',
    activation='sigmoid'
) (x)


model = tf.keras.Model(inputs, out)
model.summary()
model.compile(
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    loss='binary_crossentropy',
    metrics=['acc']
)
# %%

# Training
history = model.fit(batchs, steps_per_epoch=20, epochs=20)


# %%

# Testing

r = np.random.randint(0, 5000)
test_o, test_gt = get_input_and_label(
    TRAIN_INPUT_PATH / (TRAIN_NAME[r] + INPUT_EXT),
    TRAIN_PARSING_PATH / (TRAIN_NAME[r] + PARSING_EXT)
)
print("Original image:")
show_img(test_o)
print("Ground truth:")
show_img(test_gt)
print("Predicted segmentation:")
test_a = model.predict(tf.expand_dims(test_o, axis=0))
show_img(np.asarray(test_a[0]))

ref = test_a[0] > 0.5

show_img(ref)

# %%

# Save model if you want to

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "checkpoints/human_parsing_cp-20epochs.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.save_weights(checkpoint_path)

model.save('models/human_parsing_cp-20epochs')

# %%

# load models. (If you have saved model)
# new_model = tf.keras.models.load_model('models/human_parsing_cp-20epochs')

# Check its architecture
# new_model.summary()

# %%
# Predict on new model
r = np.random.randint(0, 5000)
test_o, test_gt = get_input_and_label(
    TRAIN_INPUT_PATH / (TRAIN_NAME[r] + INPUT_EXT),
    TRAIN_PARSING_PATH / (TRAIN_NAME[r] + PARSING_EXT)
)
print("Original image:")
show_img(test_o)
print("Ground truth:")
show_img(test_gt)
print("Predicted segmentation:")
test_a = new_model.predict(tf.expand_dims(test_o, axis=0))
show_img(np.asarray(test_a[0]))

ref = test_a[0] > 0.5

show_img(ref)

# %%

# Try with the clothing dataset. This code has not been revised.

# lip_test = "./dataset/lip_mpv_dataset/MPV_192_256/0VB21E007/0VB21E007-T11@8=person_half_front.jpg"
# test_o = tf.convert_to_tensor(np.asarray(Image.open(lip_test)))
# test_o = tf.keras.layers.experimental.preprocessing.Resizing(300, 300)(test_o)
# test_o = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(test_o)

# show_img(test_o)

# test_a = model.predict(tf.expand_dims(test_o, axis=0))
# show_img(np.asarray(test_a[0]))

# ref = test_a[0] > 0.64

# show_img(ref)
