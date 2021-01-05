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
from core.utils import *
BATCH_SIZE = 32
# TODO: Resize image using tf.image.ResizeMethod.NEAREST_NEIGHBOR

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
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
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def show_img(img):
    if len(img.shape) == 3:
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    elif len(img.shape) == 2:
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()

def plot_parsing_map(test_predict):
    # Test pred shape 4d
    # mask is shape (256, 192, 1). Range[0, 19]
    mask = create_mask(test_predict)[0]
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
        plt.imshow(mask == label_channel, cmap='gray')
    fig.tight_layout()
    plt.show()

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

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
        img = np.asarray(
                Image.open(
                    TRAIN_INPUT_PATH / (file_name + INPUT_EXT)
                )
            )
        if len(img.shape) < 3:
            continue
        rescaled_img = preprocess_image(img)

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
        resized_label = tf.image.resize(
            label, 
            IMG_SHAPE[:2],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        resized_label = tf.reshape(resized_label, PARSING_SHAPE)
        yield rescaled_img, resized_label

def val_gen_multi_parsing():
    for file_name in VAL_NAME:
        img = np.asarray(
                Image.open(
                    VAL_INPUT_PATH / (file_name + INPUT_EXT)
                )
            )
        if len(img.shape) < 3:
            continue
        rescaled_img = preprocess_image(img)

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
        resized_label = tf.image.resize(
            label, 
            IMG_SHAPE[:2],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
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
batchs = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
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
# history = model.fit(batchs, steps_per_epoch=20, epochs=1)


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
# history = model.fit(batchs, steps_per_epoch=20, epochs=1)



# %%

model = get_unet_model_for_human_parsing()

# %%

# Training model

# Tensorflow dataset object
train_ds = tf.data.Dataset.from_generator(train_gen_multi_parsing, output_signature=(
    tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
    tf.TensorSpec(shape=PARSING_SHAPE, dtype=tf.float32)
))
train_ds = train_ds.repeat().batch(BATCH_SIZE)

val_ds = tf.data.Dataset.from_generator(val_gen_multi_parsing, output_signature=(
    tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
    tf.TensorSpec(shape=PARSING_SHAPE, dtype=tf.float32)
))
val_ds = val_ds.repeat().batch(BATCH_SIZE)

# %%

history = model.fit(
    train_ds,
    steps_per_epoch=1000, 
    epochs=2, 
    validation_data=val_ds,
    validation_steps=10)
# %%


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

pred_mask = create_mask(test_predict)[0]
show_img(pred_mask)

# %%

# evaluate on lip dataset
lip_test2 = "./dataset/lip_mpv_dataset/MPV_192_256/0VB21E007/0VB21E007-T11@8=person_half_front.jpg"
lip_test3 = "./dataset/lip_mpv_dataset/MPV_192_256/ZX121DA0I/ZX121DA0I-Q11@16=person_half_front.jpg"
lip_test = "./dataset/lip_mpv_dataset/MPV_192_256/ZX121DA0P/ZX121DA0P-K11@8=person_half_front.jpg"
test_img = tf.convert_to_tensor(
    np.asarray(
        Image.open(
            lip_test2
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
pred_mask = create_mask(test_predict)[0]
show_img(pred_mask)

# %%

# Save model if you want to

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "checkpoints/human_parsing_mbv3-50epochs.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.save_weights(checkpoint_path)

# Save the entire model
model.save('models/human_parsing_mbv3-50epochs')

# %%

# load models. (If you have saved model)
model = tf.keras.models.load_model('models/human_parsing_mbv2-50epochs')

# Check its architecture
model.summary()

# %%

# Perform export prediction images

import mediapipe as mp

# this is the path to the root of the dataset
LIP_DATASET_PATH = Path("./dataset/lip_mpv_dataset/")

# This is the path to the folder contain actual data
LIP_DATASET_SRC = LIP_DATASET_PATH / "MPV_192_256"

# This is the name of the file where it contains path to data
LIP_DATASET_FILE = "all_poseA_poseB_clothes.txt"

DATASET_OUT_PATH = LIP_DATASET_PATH / "preprocessed"

LABEL_NAME_LIST = ['body_mask', 'face_hair', 'clothing_mask', 'pose']

LABEL_FOLDER_PATH = [DATASET_OUT_PATH / d for d in LABEL_NAME_LIST]
for i, name in enumerate(LABEL_FOLDER_PATH):
    if not os.path.exists(name):
        os.mkdir(name)

# print(LABEL_FOLDER_PATH[0])
def get_data_path_raw():
    train_half_front = []
    test_half_front = []

    with open(LIP_DATASET_PATH / LIP_DATASET_FILE, 'r') as f:
        for line in f:
            elems = line.split("\t")
            assert len(elems) == 4, "Unexpected readline!"
            if "train" in line:
                if "person_half_front.jpg" in line and "cloth_front.jpg" in line:
                    tmp_person = ""
                    tmp_cloth = ""
                    for elem in elems:
                        if "person_half_front.jpg" in elem:
                            tmp_person = str(elem)
                        if "cloth_front.jpg" in elem:
                            tmp_cloth = str(elem)
                    train_half_front.append([tmp_person, tmp_cloth])
                else:
                    continue
            elif "test" in line:
                if "person_half_front.jpg" in line and "cloth_front.jpg" in line:
                    tmp_person = ""
                    tmp_cloth = ""
                    for elem in elems:
                        if "person_half_front.jpg" in elem:
                            tmp_person = str(elem)
                        if "cloth_front.jpg" in elem:
                            tmp_cloth = str(elem)
                    test_half_front.append([tmp_person, tmp_cloth])
                else:
                    continue
            else:
                print("Unexpected behavior!")

    return np.asarray(train_half_front), np.asarray(test_half_front)

def get_data_path():
    train_half_front = []
    test_half_front = []

    with open(LIP_DATASET_PATH / LIP_DATASET_FILE, 'r') as f:
        for line in f:
            elems = line.split("\t")
            assert len(elems) == 4, "Unexpected readline!"
            if "train" in line:
                if "person_half_front.jpg" in line and "cloth_front.jpg" in line:
                    tmp_person = ""
                    tmp_cloth = ""
                    for elem in elems:
                        if "person_half_front.jpg" in elem:
                            tmp_person = str(LIP_DATASET_SRC / elem)
                        if "cloth_front.jpg" in elem:
                            tmp_cloth = str(LIP_DATASET_SRC / elem)
                    train_half_front.append([tmp_person, tmp_cloth])
                else:
                    continue
            elif "test" in line:
                if "person_half_front.jpg" in line and "cloth_front.jpg" in line:
                    tmp_person = ""
                    tmp_cloth = ""
                    for elem in elems:
                        if "person_half_front.jpg" in elem:
                            tmp_person = str(LIP_DATASET_SRC / elem)
                        if "cloth_front.jpg" in elem:
                            tmp_cloth = str(LIP_DATASET_SRC / elem)
                    test_half_front.append([tmp_person, tmp_cloth])
                else:
                    continue
            else:
                print("Unexpected behavior!")

    return np.asarray(train_half_front), np.asarray(test_half_front)

TRAIN_PATH, TEST_PATH = get_data_path_raw()

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

def get_pose_map(path) -> np.ndarray:
    """ given a path, return a pose map

    Args:
        img (np.ndarray): 

    Returns:
        np.ndarray: [description]
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5)
    image = cv2.imread(path)
    image_height, image_width, n_channels = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    RED_COLOR = (0, 0, 255)
    if results.pose_landmarks:
        annotated_image = np.zeros(shape=(image_height, image_width, n_channels))
        
        mp_drawing.draw_landmarks(
            annotated_image, 
            results.pose_landmarks,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                # color=RED_COLOR,
                thickness=4,
                circle_radius=1
            ))
        pose.close()
        return tf.image.resize(np.asarray(annotated_image) / 255.0, IMG_SHAPE[:2])
    pose.close()
    return np.zeros(shape=IMG_SHAPE)

def get_human_parsing(img):
    assert img.shape == IMG_SHAPE, "Wrong image shape"
    prediction = model.predict(tf.expand_dims(img, axis=0))[0]
    return prediction

def create_mask(pred_mask):
    # pred_mask shape 3d, not 4d
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def random_prediction():
    r = random.randint(0, TRAIN_PATH.shape[0] - 1)
    img = np.asarray(
        Image.open(
            LIP_DATASET_SRC / TRAIN_PATH[r, 0]
        )
    )
    print("Original:")
    show_img(img)
    rescaled_img = preprocess_image(img)
    test_predict = model.predict(tf.expand_dims(rescaled_img, axis=0))
    plot_parsing_map(test_predict)
    pred_mask = create_mask(test_predict)[0]
    show_img(pred_mask)

# %%

# LIP Random prediction

random_prediction()



# %%
""" Write files

Writw a lot of files

"""

import re
from tqdm import tqdm
import time

r_str = r"\/.*\.jpg$"
with tqdm(total=TRAIN_PATH.shape[0]) as pbar:
    for (idx, [img_path, cloth_path]) in enumerate(TRAIN_PATH):

        # time.sleep(0.1)

        # Eg: img path is raw. I.e: does not contains the full path (dataset/lip_mpv_dataset/preprocessed)
        # img_path: RE321D05M\RE321D05M-Q11@8=person_half_front.jpg
        # img_path_name: RE321D05M-Q11@8=person_half_front.jpg
        # img_folder_name: RE321D05M
        img_path_name = re.findall(r_str, img_path)[0]
        img_folder_name = img_path[:(len(img_path) - len(img_path_name))]
        cloth_path_name = re.findall(r_str, cloth_path)[0]
        cloth_folder_name = cloth_path[:(len(cloth_path) - len(cloth_path_name))]

        assert img_folder_name == cloth_folder_name, "Unexpected behavior"

        for i, name in enumerate(LABEL_FOLDER_PATH):
            if not os.path.exists(name / img_folder_name):
                os.mkdir(name / img_folder_name)
            if not os.path.exists(name / cloth_folder_name):
                os.mkdir(name / cloth_folder_name)

        sample_cloth = tf.image.resize(
            np.asarray(
                Image.open(
                    LIP_DATASET_SRC / cloth_path
                )
            ), IMG_SHAPE[:2]
        ) / 255.0

        # sample_img shape (256, 192, 3). Range [0, 1]
        sample_img = tf.image.resize(
            np.asarray(
                Image.open(
                    LIP_DATASET_SRC / img_path
                )
            ), IMG_SHAPE[:2]
        ) / 255.0
        
        if not os.path.exists(LABEL_FOLDER_PATH[3] / img_path):
            # sample_pose shape (256, 192, 3). Range [0, 1]
            sample_pose = get_pose_map(str(LIP_DATASET_SRC / img_path))
            tf.keras.preprocessing.image.save_img(
                LABEL_FOLDER_PATH[3] / img_path, sample_pose
            )

        # sample_pose shape (256, 192, 20). Range [0, 1]
        sample_parsing = get_human_parsing(sample_img)
        # sample_body_mask shape (256, 192, 1). Range [0, 19]. Representing classes.
        sample_mask = create_mask(sample_parsing)

        if not os.path.exists(LABEL_FOLDER_PATH[0] / img_path):
            body_masking_channels = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
            sample_body_mask = [sample_mask == c for c in body_masking_channels]
            # sample_body_mask shape (len(body_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
            sample_body_mask = tf.reduce_any(sample_body_mask, axis=0)
            sample_body_mask = tf.cast(sample_body_mask, dtype=tf.float32)
            # print(sample_body_mask.shape)
            # show_img(sample_body_mask)
            # print(os.path.join(LABEL_FOLDER_PATH[0], img_path_name))
            tf.keras.preprocessing.image.save_img(
                LABEL_FOLDER_PATH[0] / img_path, sample_body_mask
            )

        if not os.path.exists(LABEL_FOLDER_PATH[1] / img_path):
            face_hair_masking_channels = [1, 2, 13]
            sample_face_hair_mask = [sample_mask == c for c in face_hair_masking_channels]
            # sample_face_hair_mask shape (len(face_hair_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
            sample_face_hair_mask = tf.reduce_any(sample_face_hair_mask, axis=0)
            sample_face_hair_mask = tf.cast(sample_face_hair_mask, dtype=tf.float32)

            sample_face_hair = sample_img * sample_face_hair_mask
            # print(sample_face_hair.shape)
            # show_img(sample_face_hair)
            tf.keras.preprocessing.image.save_img(
                LABEL_FOLDER_PATH[1] / img_path, sample_face_hair
            )

        if not os.path.exists(LABEL_FOLDER_PATH[2] / img_path):
            # Take everything except for the background, face, and hair
            clothing_masking_channels = [5, 6, 7, 12]
            sample_clothing_mask = [sample_mask == c for c in clothing_masking_channels]
            # sample_clothing_mask shape (len(face_hair_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
            sample_clothing_mask = tf.reduce_any(sample_clothing_mask, axis=0)
            sample_clothing_mask = tf.cast(sample_clothing_mask, dtype=tf.float32)
            # print(sample_clothing_mask.shape)
            # show_img(sample_clothing_mask)
            tf.keras.preprocessing.image.save_img(
                LABEL_FOLDER_PATH[2] / cloth_path, sample_clothing_mask
            )

        pbar.update(1)
    



# %%
