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

from core.utils import *
from core.models import *

# config

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

# Please config the path as accurate as possible

# this is the path to the root of the dataset
DATASET_PATH = Path("./dataset/lip_mpv_dataset/")

# This is the path to the folder contain actual data
DATASET_SRC = DATASET_PATH / "MPV_192_256"

# This is the name of the file where it contains path to data
DATASET_FILE = "all_poseA_poseB_clothes.txt"

DATASET_OUT_PATH = DATASET_PATH / "preprocessed"

LABEL_NAME_LIST = ['body_mask', 'face_hair', 'clothing_mask', 'pose']

LABEL_FOLDER_PATH = [DATASET_OUT_PATH / d for d in LABEL_NAME_LIST]

BATCH_SIZE = 8
STEP_PER_EPOCHS = 20
IMG_SHAPE = (256, 192, 3)

MASK_THRESHOLD = 0.9

# Deprecated approach
# load human parsing model that has been trained in human_parsing notebook.
# parsing_model = tf.keras.models.load_model('models/human_parsing_mbv2-50epochs')

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

def get_human_parsing(img):
    assert img.shape == IMG_SHAPE, "Wrong image shape"
    prediction = parsing_model.predict(tf.expand_dims(img, axis=0))[0]
    return prediction

# %%

# Sample data
r = np.random.randint(0, TRAIN_PATH.shape[0] - 1)

sample_cloth = preprocess_image(
    np.asarray(
        Image.open(
            DATASET_SRC / TRAIN_PATH[r,1]
        )
    ), IMG_SHAPE[:2]
)
print(f"Min val: {tf.reduce_min(sample_cloth)}, max val: {tf.reduce_max(sample_cloth)}")
show_img(deprocess_img(sample_cloth))

# sample_img shape (256, 192, 3). Range [0, 1]
sample_img = preprocess_image(
    np.asarray(
        Image.open(
                DATASET_SRC / TRAIN_PATH[r,0]
        )
    ), IMG_SHAPE[:2]
)
print(f"Min val: {tf.reduce_min(sample_img)}, max val: {tf.reduce_max(sample_img)}")
show_img(deprocess_img(sample_img))

# sample_pose shape (256, 192, 3). Range [0, 1].
sample_pose = preprocess_image (
    np.asarray(Image.open(LABEL_FOLDER_PATH[3] / TRAIN_PATH[r,0])),
    IMG_SHAPE[:2]
)
print(f"Min val: {tf.reduce_min(sample_pose)}, max val: {tf.reduce_max(sample_pose)}")
show_img(deprocess_img(sample_pose))

# sample_body_mask shape (256, 192, 1).
sample_body_mask =  preprocess_image(
    tf.expand_dims(np.asarray(Image.open(LABEL_FOLDER_PATH[0] / TRAIN_PATH[r,0])), axis=2),
    IMG_SHAPE[:2],
    resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
)
print(f"Min val: {tf.reduce_min(sample_body_mask)}, max val: {tf.reduce_max(sample_body_mask)}")
show_img(deprocess_img(sample_body_mask))

# sample_face_hair shape (256, 192, 1).
sample_face_hair = preprocess_image(
    np.asarray(Image.open(LABEL_FOLDER_PATH[1] / TRAIN_PATH[r,0])),
    IMG_SHAPE[:2]
)
print(f"Min val: {tf.reduce_min(sample_face_hair)}, max val: {tf.reduce_max(sample_face_hair)}")
show_img(deprocess_img(sample_face_hair))

# sample_clothing_mask shape (256, 192, 1).
sample_clothing_mask = preprocess_image(
    tf.expand_dims(np.asarray(Image.open(LABEL_FOLDER_PATH[2] / TRAIN_PATH[r,1])), axis=2),
    IMG_SHAPE[:2],
    resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
)
print(f"Min val: {tf.reduce_min(sample_clothing_mask)}, max val: {tf.reduce_max(sample_clothing_mask)}")
show_img(deprocess_img(sample_clothing_mask))

# %%

# After sampling data, we now can build the dataset
# dEPRECATED GENErator
def train_generator_deprecated():
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


def train_generator():
    for (idx, [img_path, cloth_path]) in enumerate(TRAIN_PATH):
        sample_cloth = preprocess_image(
            np.asarray(
                Image.open(
                    DATASET_SRC / cloth_path
                )
            ), IMG_SHAPE[:2]
        )
        # show_img(sample_cloth)

        # sample_img shape (256, 192, 3). Range [0, 1]
        sample_img = preprocess_image(
            np.asarray(
                Image.open(
                     DATASET_SRC / img_path
                )
            ), IMG_SHAPE[:2]
        )
        # show_img(sample_img)

        # sample_pose shape (256, 192, 3). Range [0, 1].
        sample_pose = preprocess_image (
            np.asarray(Image.open(LABEL_FOLDER_PATH[3] / img_path)),
            IMG_SHAPE[:2]
        )
        # show_img(sample_pose)

        # sample_body_mask shape (256, 192, 1).
        sample_body_mask =  preprocess_image(
            tf.expand_dims(np.asarray(Image.open(LABEL_FOLDER_PATH[0] / img_path)), axis=2),
            IMG_SHAPE[:2],
            resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        # show_img(sample_body_mask)

        # sample_face_hair shape (256, 192, 1).
        sample_face_hair = preprocess_image(
            np.asarray(Image.open(LABEL_FOLDER_PATH[1] / img_path)),
            IMG_SHAPE[:2]
        )
        # show_img(sample_face_hair)

        # sample_clothing_mask shape (256, 192, 1).
        sample_clothing_mask = preprocess_image(
            tf.expand_dims(np.asarray(Image.open(LABEL_FOLDER_PATH[2] / cloth_path)), axis=2),
            IMG_SHAPE[:2],
            resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        # show_img(sample_clothing_mask)

        yield {
                "input_pose": sample_pose,
                "input_body_mask": sample_body_mask,
                "input_face_hair":sample_face_hair,
                "input_cloth": sample_cloth
            }, \
            {
                "output_image": sample_img,
                "output_cloth_mask": sample_clothing_mask
            }

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
train_batch_ds = train_ds.batch(BATCH_SIZE)
it = iter(train_ds)

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
    return 1.0 * compute_mse_loss(real, pred)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)

def train_step(person_reprs, clothings, labels):
    # Use gradient tape
    pass

# %%

model = get_res_unet_model()

# %%

model.summary()
# Checkpoint path
checkpoint_path = "models/checkpoints/viton_9.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
if os.path.exists("models/checkpoints/viton_9.ckpt.index"):
    model.load_weights(checkpoint_path)
    print("Weights loaded!")


# %%

# Training the model

EPOCHS = 1
with tf.device('/device:CPU:0'):
    for epoch in range(EPOCHS):
        print("\nStart of epoch %d" % (epoch + 1,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_batch_ds.shuffle(4).take(STEP_PER_EPOCHS)):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape, tf.device('/device:GPU:0'):

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
                print("Input cloth:")
                show_img(deprocess_img(x_batch_train["input_cloth"][0]))
                print("Predictions:")
                show_img(deprocess_img(logits_human[0]))
                show_img(deprocess_img(logits_mask[0]))

                # Compute the loss value for this minibatch.
                loss_value = loss_function(y_batch_train["output_image"], logits_human)
                loss_mask_value = mask_loss_function(y_batch_train["output_cloth_mask"], logits_mask)
                loss = loss_value + loss_mask_value
                print(f"loss for this batch at step: {step + 1}: {loss }")

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

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

model.save('models/viton-mbv2-10epochs')

# %%

# eval

sample_input_batch, sample_output_batch = next(iter(train_batch_ds.take(1)))

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
show_img(deprocess_img(pred_cloth[r]))

# %%




# %%



