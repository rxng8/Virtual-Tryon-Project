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
    'Background': 0,
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

IMG_SHAPE = (256, 192, 3)
PARSING_SHAPE = (256, 192)

# The reason why I named it single is because we use this to parse binary. I.e whether
# it is the person or not.
PARSING_SINGLE_SHAPE = (256, 192, 1)

# %%

def plot_history(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    plt.figure()
    plt.plot(history.history['acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()


def show_img(img):
    if len(img.shape) == 3:
        plt.figure()
        plt.imshow(img)
        plt.show()
    elif len(img.shape) == 2:
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.show()

def plot_parsing_map(test_predict):
    width=15
    height=10
    rows = 4
    cols = 5
    axes=[]
    fig=plt.figure(figsize=(width, height))

    for i, (label_name, label_channel) in enumerate(LABEL_NAME.items()):
        
        axes.append(fig.add_subplot(rows, cols, i+1))
        subplot_title=(label_name)
        axes[-1].set_title(subplot_title)
        axes[-1].axis('off')
        plt.imshow(test_predict[0, :, :, label_channel] > 0.5, cmap='gray')
    fig.tight_layout()
    plt.show()


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def train_gen_single_parsing():
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
        label[label > 0] = 1
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        resized_label = tf.keras.layers.experimental.preprocessing.Resizing(*PARSING_SINGLE_SHAPE[:2])(label)
        yield rescaled_img, resized_label

def train_gen_multi_parsing():
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
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        resized_label = tf.keras.layers.experimental.preprocessing.Resizing(*PARSING_SHAPE[:2])(label)
        resized_label = tf.reshape(resized_label, PARSING_SHAPE)
        yield rescaled_img, resized_label

def val_gen_multi_parsing():
    for file_name in VAL_NAME:
        img = tf.convert_to_tensor(
            np.asarray(
                Image.open(
                    VAL_INPUT_PATH / (file_name + INPUT_EXT)
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
                        VAL_PARSING_PATH / (file_name + PARSING_EXT)
                    ),
                    dtype=float
                ), axis=2
            )
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        resized_label = tf.keras.layers.experimental.preprocessing.Resizing(*PARSING_SHAPE[:2])(label)
        resized_label = tf.reshape(resized_label, PARSING_SHAPE)
        yield rescaled_img, resized_label

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
    label = tf.convert_to_tensor(label, dtype=tf.float32)
    resized_label = tf.keras.layers.experimental.preprocessing.Resizing(*PARSING_SHAPE[:2])(label)
    return test_o, resized_label


# %%

# See the actual un reshaped data

r = np.random.randint(0, 5000)

sample_input = np.asarray(Image.open(TRAIN_INPUT_PATH / (TRAIN_NAME[r] + INPUT_EXT)))
sample_label = np.asarray(Image.open(TRAIN_PARSING_PATH / (TRAIN_NAME[r] + PARSING_EXT)))

print(f"Shape: {sample_input.shape}")
show_img(sample_input)
show_img(sample_label)

# %%

# Uncomment this to see the data with dataset generating binary parsing

# ds = tf.data.Dataset.from_generator(train_gen_single_parsing, output_signature=(
#     tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
#     tf.TensorSpec(shape=PARSING_SINGLE_SHAPE, dtype=tf.float32)
# ))

# # Generator object
# train_gen_obs = train_gen_single_parsing()
# it = iter(ds)
# batchs = ds.repeat().batch(20)
# batchs


# %%

# See the data in dataset generator

# Tensorflow dataset object
ds = tf.data.Dataset.from_generator(train_gen_multi_parsing, output_signature=(
    tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
    tf.TensorSpec(shape=PARSING_SHAPE, dtype=tf.float32)
))

# Generator object
train_gen_obs = train_gen_multi_parsing()
it = iter(ds)
batchs = ds.repeat().batch(20)
batchs

# %%

# See some sample data

a, b = next(it)
show_img(a)
show_img(b)


# %%

# Build a simple Convolutional Autoencoder model. Don't use this model tho
# The reason why this model is deprecated is it's output shape is 3 dimesions
# so that we can just predict according to parsing 1 or 0 using binary crossentropy loss.
# Please use the simple parsing dataset for this model.

inputs = tf.keras.layers.Input(shape=IMG_SHAPE)
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

# Train this model
# create dataset
ds = tf.data.Dataset.from_generator(train_gen_single_parsing, output_signature=(
    tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
    tf.TensorSpec(shape=PARSING_SINGLE_SHAPE, dtype=tf.float32)
))

# Generator object
batchs = ds.repeat().batch(20)

# Train
history = model.fit(batchs, steps_per_epoch=20, epochs=1)


# %%

# Build another model with VGG-16
# The reason why this model is deprecated is it's output shape is 3 dimesions
# so that we can just predict according to parsing 1 or 0 using binary crossentropy loss.
# Please use the simple parsing dataset for this model.

from tensorflow_examples.models.pix2pix import pix2pix
vgg16_model = tf.keras.applications.VGG16(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
vgg16_model.trainable = False
# vgg16_model.summary()

inputs = tf.keras.Input(shape=IMG_SHAPE)
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

# Train this model
# create dataset
ds = tf.data.Dataset.from_generator(train_gen_single_parsing, output_signature=(
    tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
    tf.TensorSpec(shape=PARSING_SINGLE_SHAPE, dtype=tf.float32)
))

# Generator object
batchs = ds.repeat().batch(20)

# Train
history = model.fit(batchs, steps_per_epoch=20, epochs=1)



# %%

# This is the U-net model

"""
Read this u-net article with the cute meow:
    https://towardsdatascience.com/u-net-b229b32b4a71
"""

from tensorflow_examples.models.pix2pix import pix2pix
mobile_net_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, 
    include_top=False)
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


inputs = tf.keras.Input(shape=IMG_SHAPE)
out4, out3, out2, out1, out0 = wrap_mobile_net_model(inputs, training=False)

up1_tensor = pix2pix.upsample(512, 3)(out0)

cat1_tensor = tf.keras.layers.concatenate([up1_tensor, out1])
up2_tensor = pix2pix.upsample(256, 3)(cat1_tensor)

cat2_tensor = tf.keras.layers.concatenate([up2_tensor, out2])
up3_tensor = pix2pix.upsample(128, 3)(cat2_tensor)

cat3_tensor = tf.keras.layers.concatenate([up3_tensor, out3])
up4_tensor = pix2pix.upsample(64, 3)(cat3_tensor)

cat4_tensor = tf.keras.layers.concatenate([up4_tensor, out4])

# There are 20 integer category (not one-hot-encoded), so, n channels (or neurons, or feature vectors) is 20
out = tf.keras.layers.Conv2DTranspose(
    20, 3, strides=2,
    padding='same',
    activation='sigmoid'
) (cat4_tensor)

model = tf.keras.Model(inputs, out)
model.summary()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['acc']
)

# %%

# Training model

# Tensorflow dataset object
train_ds = tf.data.Dataset.from_generator(train_gen_multi_parsing, output_signature=(
    tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
    tf.TensorSpec(shape=PARSING_SHAPE, dtype=tf.float32)
))
train_ds = train_ds.repeat().batch(20)

val_ds = tf.data.Dataset.from_generator(val_gen_multi_parsing, output_signature=(
    tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
    tf.TensorSpec(shape=PARSING_SHAPE, dtype=tf.float32)
))
val_ds = val_ds.repeat().batch(20)

history = model.fit(train_ds, steps_per_epoch=20, epochs=50, validation_data=val_ds)

plot_history(history)

# %%

# Testing for binary classification. Uncomment to test

# r = np.random.randint(0, 5000)
# test_o, test_gt = get_input_and_label(
#     VAL_INPUT_PATH / (VAL_NAME[r] + INPUT_EXT),
#     VAL_PARSING_PATH / (VAL_NAME[r] + PARSING_EXT)
# )
# print("Original image:")
# show_img(test_o)
# print("Ground truth:")
# show_img(test_gt)
# print("Predicted segmentation:")
# test_a = model.predict(tf.expand_dims(test_o, axis=0))
# show_img(np.asarray(test_a[0]))

# ref = test_a[0] > 0.5

# show_img(ref)


# %%

# Testing for multi label classification.

r = np.random.randint(0, 5000)
test_img, test_gt = get_input_and_label(
    VAL_INPUT_PATH / (VAL_NAME[r] + INPUT_EXT),
    VAL_PARSING_PATH / (VAL_NAME[r] + PARSING_EXT)
)
print("Original image:")
show_img(test_img)
print("Ground truth:")
show_img(test_gt)
print("Predicted segmentation:")
# Test predict will now contain 20 channel corressponding to 20 classification labels
test_predict = model.predict(tf.expand_dims(test_img, axis=0))

plot_parsing_map(test_predict)

# %%

# evaluate on lip dataset
lip_test = "./dataset/lip_mpv_dataset/MPV_192_256/0VB21E007/0VB21E007-T11@8=person_half_front.jpg"
test_img = tf.convert_to_tensor(
    np.asarray(
        Image.open(
            lip_test
        )
    )
)
test_img = tf.keras.layers.experimental.preprocessing.Resizing(*IMG_SHAPE[:2])(test_img)
test_img = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(test_img)
print("Original image:")
show_img(test_img)
print("Predicted segmentation:")
# Test predict will now contain 20 channel corressponding to 20 classification labels
test_predict = model.predict(tf.expand_dims(test_img, axis=0))
plot_parsing_map(test_predict)


# %%

# Save model if you want to

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "checkpoints/human_parsing_mbv2-50epochs.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.save_weights(checkpoint_path)

# Save the entire model
model.save('models/human_parsing_mbv2-50epochs')

# %%

# load models. (If you have saved model)
new_model = tf.keras.models.load_model('models/human_parsing_cp-20epochs')

# Check its architecture
new_model.summary()

