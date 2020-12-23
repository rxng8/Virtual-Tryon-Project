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

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2

import mediapipe as mp


# config

# GPU config first
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

# Please config the path as accurate as possible

# this is the path to the root of the dataset
DATASET_PATH = Path("./dataset/lip_mpv_dataset/")

# This is the path to the folder contain actual data
DATASET_SRC = DATASET_PATH / "MPV_192_256"

# This is the name of the file where it contains path to data
DATASET_FILE = "all_poseA_poseB_clothes.txt"

BATCH_SIZE = 16
STEP_PER_EPOCHS = 20
IMG_SHAPE = (256, 192, 3)

MASK_THRESHOLD = 0.9

# load human parsing model that has been trained in human_parsing notebook.
parsing_model = tf.keras.models.load_model('models/human_parsing_mbv2-50epochs')

def get_data_path_raw():
    train_half_front = []
    test_half_front = []

    with open(DATASET_PATH / DATASET_FILE, 'r') as f:
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

    with open(DATASET_PATH / DATASET_FILE, 'r') as f:
        for line in f:
            elems = line.split("\t")
            assert len(elems) == 4, "Unexpected readline!"
            if "train" in line:
                if "person_half_front.jpg" in line and "cloth_front.jpg" in line:
                    tmp_person = ""
                    tmp_cloth = ""
                    for elem in elems:
                        if "person_half_front.jpg" in elem:
                            tmp_person = str(DATASET_SRC / elem)
                        if "cloth_front.jpg" in elem:
                            tmp_cloth = str(DATASET_SRC / elem)
                    train_half_front.append([tmp_person, tmp_cloth])
                else:
                    continue
            elif "test" in line:
                if "person_half_front.jpg" in line and "cloth_front.jpg" in line:
                    tmp_person = ""
                    tmp_cloth = ""
                    for elem in elems:
                        if "person_half_front.jpg" in elem:
                            tmp_person = str(DATASET_SRC / elem)
                        if "cloth_front.jpg" in elem:
                            tmp_cloth = str(DATASET_SRC / elem)
                    test_half_front.append([tmp_person, tmp_cloth])
                else:
                    continue
            else:
                print("Unexpected behavior!")

    return np.asarray(train_half_front), np.asarray(test_half_front)

TRAIN_PATH, TEST_PATH = get_data_path()

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

# %%

# Some definition

def get_pose_map_generator(path_list: np.ndarray) -> None:
    """ given a path, return a pose map

    Args:
        img (np.ndarray): 

    Returns:
        np.ndarray: [description]
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    pose = mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5)
    for idx, f in enumerate(path_list):
        image = cv2.imread(f)
        image_height, image_width, n_channels = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            continue
        # Draw pose landmarks on the image.
        annotated_image = np.zeros(shape=(image_height, image_width, n_channels))
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks)
        yield np.asarray(annotated_image) / 255.0
    pose.close()

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
    prediction = parsing_model.predict(tf.expand_dims(img, axis=0))[0]
    return prediction

def create_mask(pred_mask):
    # pred_mask shape 3d, not 4d
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

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

# %%

# Sample data
with tf.device('/device:GPU:0'):
    r = np.random.randint(0, TRAIN_PATH.shape[0] - 1)

    # sample_img shape (256, 192, 3). Range [0, 1]
    sample_cloth = tf.image.resize(
        np.asarray(
            Image.open(
                TRAIN_PATH[r, 1]
            )
        ), IMG_SHAPE[:2]
    ) / 255.0
    show_img(sample_cloth)

    # sample_img shape (256, 192, 3). Range [0, 1]
    sample_img = tf.image.resize(
        np.asarray(
            Image.open(
                TRAIN_PATH[r, 0]
            )
        ), IMG_SHAPE[:2]
    ) / 255.0

    show_img(sample_img)

    # sample_pose shape (256, 192, 3). Range [0, 1]
    sample_pose = get_pose_map(TRAIN_PATH[r, 0])
    show_img(sample_pose)

    # sample_pose shape (256, 192, 20). Range [0, 1]
    sample_parsing = get_human_parsing(sample_img)
    # sample_body_mask shape (256, 192, 1). Range [0, 19]. Representing classes.
    sample_mask = create_mask(sample_parsing)

    body_masking_channels = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
    sample_body_mask = [sample_mask == c for c in body_masking_channels]
    # sample_body_mask shape (len(body_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
    sample_body_mask = tf.reduce_any(sample_body_mask, axis=0)
    sample_body_mask = tf.cast(sample_body_mask, dtype=tf.float32)
    show_img(sample_body_mask)

    face_hair_masking_channels = [1, 2, 13]
    sample_face_hair_mask = [sample_mask == c for c in face_hair_masking_channels]
    # sample_face_hair_mask shape (len(face_hair_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
    sample_face_hair_mask = tf.reduce_any(sample_face_hair_mask, axis=0)
    sample_face_hair_mask = tf.cast(sample_face_hair_mask, dtype=tf.float32)
    # show_img(sample_face_hair_mask)

    sample_face_hair = sample_img * sample_face_hair_mask
    show_img(sample_face_hair)

    # Take everything except for the background, face, and hair
    clothing_masking_channels = [5, 6, 7, 12]
    sample_clothing_mask = [sample_mask == c for c in clothing_masking_channels]
    # sample_clothing_mask shape (len(face_hair_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
    sample_clothing_mask = tf.reduce_any(sample_clothing_mask, axis=0)
    sample_clothing_mask = tf.cast(sample_clothing_mask, dtype=tf.float32)
    show_img(sample_clothing_mask)

# %%

# After sampling data, we now can build the dataset

def train_generator():
    for (idx, [img_path, cloth_path]) in enumerate(TRAIN_PATH):
        sample_cloth = tf.image.resize(
            np.asarray(
                Image.open(
                    cloth_path
                )
            ), IMG_SHAPE[:2]
        ) / 255.0

        # sample_img shape (256, 192, 3). Range [0, 1]
        sample_img = tf.image.resize(
            np.asarray(
                Image.open(
                    img_path
                )
            ), IMG_SHAPE[:2]
        ) / 255.0

        # sample_pose shape (256, 192, 3). Range [0, 1]
        sample_pose = get_pose_map(img_path)

        # sample_pose shape (256, 192, 20). Range [0, 1]
        sample_parsing = get_human_parsing(sample_img)
        # sample_body_mask shape (256, 192, 1). Range [0, 19]. Representing classes.
        sample_mask = create_mask(sample_parsing)

        body_masking_channels = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
        sample_body_mask = [sample_mask == c for c in body_masking_channels]
        # sample_body_mask shape (len(body_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
        sample_body_mask = tf.reduce_any(sample_body_mask, axis=0)
        sample_body_mask = tf.cast(sample_body_mask, dtype=tf.float32)

        face_hair_masking_channels = [1, 2, 13]
        sample_face_hair_mask = [sample_mask == c for c in face_hair_masking_channels]
        # sample_face_hair_mask shape (len(face_hair_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
        sample_face_hair_mask = tf.reduce_any(sample_face_hair_mask, axis=0)
        sample_face_hair_mask = tf.cast(sample_face_hair_mask, dtype=tf.float32)
        # show_img(sample_face_hair_mask)

        sample_face_hair = sample_img * sample_face_hair_mask

        # Take everything except for the background, face, and hair
        clothing_masking_channels = [5, 6, 7, 12]
        sample_clothing_mask = [sample_mask == c for c in clothing_masking_channels]
        # sample_clothing_mask shape (len(face_hair_masking_channels), 256, 192, 1). Range [0, 19]. Representing classes.
        sample_clothing_mask = tf.reduce_any(sample_clothing_mask, axis=0)
        sample_clothing_mask = tf.cast(sample_clothing_mask, dtype=tf.float32)
        
        yield tf.concat([sample_pose, sample_body_mask, sample_face_hair, sample_cloth], axis=2), \
            tf.concat([sample_img, sample_clothing_mask], axis=2)

train_ds = tf.data.Dataset.from_generator(
    train_generator,
    output_signature=(
        tf.TensorSpec(shape=(*IMG_SHAPE[:2], 10), dtype=tf.float32),
        tf.TensorSpec(shape=(*IMG_SHAPE[:2], 4), dtype=tf.float32)
    )
)
train_batch_ds = train_ds.batch(BATCH_SIZE)
it = iter(train_ds)

# %%

# test dataset
sample_input, sample_output = next(it)
print(sample_input.shape)
print(sample_output.shape)

# %%



# %%
# from shutil import copyfile
# train_path_raw, test_path_raw = get_data_path_raw()

# # Copy the segmentation model to input folder
# r_str = r"\/.*\.jpg$"

# for i, line in enumerate(train_path_raw):
#     # each line is 2 links, one is the person, 1 ius the cloth
#     # copy only th cloth to the input folder to segment.
#     url = line[0]
#     name_list = re.findall(r_str, url)
#     if len(name_list) == 0:
#         print(url)
#     else:
#         name = name_list[0][1:]
#         copyfile(DATASET_SRC / url, Path("./reference/inputs") / name)

# for i, line in enumerate(test_path_raw):
#     # each line is 2 links, one is the person, 1 ius the cloth
#     # copy only th cloth to the input folder to segment.
#     url = line[0]
#     name_list = re.findall(r_str, url)
#     if len(name_list) == 0:
#         print(url)
#     else:
#         name = name_list[0][1:]
#         copyfile(DATASET_SRC / url, Path("./reference/inputs") / name)

# %%

# Test Human pose




# %%

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
mobile_net_model.summary()
mobile_net_model.trainable = True
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
wrap_mobile_net_model.trainable = True


inputs = tf.keras.Input(shape=(*IMG_SHAPE[:2], 10))

pre_conv = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(inputs)

out4, out3, out2, out1, out0 = wrap_mobile_net_model(pre_conv, training=True)

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
out = tf.keras.layers.Conv2DTranspose(
    4, 3, strides=2,
    padding='same',
    activation='relu'
) (cat4_tensor)

# We will not use model, we will just use it to see the summary!
model = tf.keras.Model(inputs, out)
model.summary()

# %%

# Normal 2D auto encoder model with u net

# Build a simple Convolutional Autoencoder model. Don't use this model tho
# The reason why this model is deprecated is it's output shape is 3 dimesions
# so that we can just predict according to parsing 1 or 0 using binary crossentropy loss.
# Please use the simple parsing dataset for this model.

inputs = tf.keras.layers.Input(shape=(*IMG_SHAPE[:2], 10))
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
    filters=4, 
    kernel_size=(3, 3), 
    activation='relu',
    padding='same') (x)


model = tf.keras.Model(inputs, outputs)
model.summary()



# %%
# Deprecated
class PerceptualVGG16(tf.keras.Model):
    layers = [
        'block1_conv2', 
        'block2_conv2', 
        'block3_conv2', 
        'block4_conv2',
        'block5_conv2'
    ]
    def __init__(self):
        self.vgg16 = tf.keras.applications.VGG16(
            include_top=False, 
            weights='imagenet',
            input_shape=IMG_SHAPE
        )
        self.vgg16.trainable = False
    
    def __call__(self, x: tf.Tensor):
        assert x.shape == IMG_SHAPE, "Wrong shape!"
        return [self.vgg16.get_layer()]

# %%

vgg16 = tf.keras.applications.VGG16(
    include_top=False, 
    weights='imagenet',
    input_shape=IMG_SHAPE
)
vgg16.trainable = False
layer_names = [
    'block1_conv2', # 256 x 192 x 64
    'block2_conv2', # 128 x 96 x 128
    'block3_conv2', # 64 x 48 x 256
    'block4_conv2', # 32 x 24 x 512
    'block5_conv2'  # 16 x 12 x 512
]
layers = [vgg16.get_layer(name).output for name in layer_names]

# Create the feature extraction model
wrap_vgg16_model = tf.keras.Model(inputs=vgg16.input, outputs=layers)
wrap_vgg16_model.trainable = False

def loss_function(real, pred):
    # Read about perceptual loss here:
    # https://towardsdatascience.com/perceptual-losses-for-real-time-style-transfer-and-super-resolution-637b5d93fa6d
    # Also, tensorflow losses only compute loss across the last dimension. so we 
    # have to reduce mean to a constant

    # Perceptual loss eval
    out_real = wrap_vgg16_model(real[:,:,:,:3], training=False)
    out_pred = wrap_vgg16_model(pred[:,:,:,:3], training=False)

    # pixel-pise loss, RGB predicted value
    pixel_loss = tf.reduce_mean(tf.math.abs(real[:,:,:,:3] - pred[:,:,:,:3]))

    perceptual_loss = 0
    # Perceptual loss
    for real_features, pred_features in zip(out_real, out_pred):
        perceptual_loss += tf.reduce_mean(tf.math.abs(real_features - pred_features))
    # perceptual_loss /= len(out_real)

    # L1 loss
    mask_loss = tf.reduce_mean(tf.math.abs(real[:,:,:,3:] - pred[:,:,:,3:]))

    return pixel_loss + perceptual_loss + mask_loss

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)

def train_step(person_reprs, clothings, labels):
    # Use gradient tape
    pass



# %%

# Training the model

EPOCHS = 1

for epoch in range(EPOCHS):
    print("\nStart of epoch %d" % (epoch + 1,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_batch_ds.take(STEP_PER_EPOCHS)):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            
            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_function(y_batch_train, logits)
            print(f"loss for this batch at step: {step + 1}: {loss_value}")
            gc.collect()
            torch.cuda.empty_cache()
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        # grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        # optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        # if step % 200 == 0:
        #     print(
        #         "Training loss (for one batch) at step %d: %.4f"
        #         % (step, float(loss_value))
        #     )
        #     print("Seen so far: %s samples" % ((step + 1) * 64))

# %%

model.save('models/viton-mbv2-10epochs')

# %%

# eval

sample_input_batch, sample_output_batch = next(iter(train_batch_ds.take(1)))

#%%

# Evaluate
r = np.random.randint(0, BATCH_SIZE - 1)

print("Input: ")
show_img(sample_input_batch[r, :, :, 0:3])
show_img(sample_input_batch[r, :, :, 3])
show_img(sample_input_batch[r, :, :, 4:7])
show_img(sample_input_batch[r, :, :, 7:10])

print('label:')
show_img(sample_output_batch[r, :, :, 0:3])
show_img(sample_output_batch[r, :, :, 3])

print('pred:')
pred = model(sample_input_batch)
show_img(pred[r, :, :, 0:3])
show_img(pred[r, :, :, 3])

# %%




# %%



