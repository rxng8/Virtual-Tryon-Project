import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .utils import show_img, preprocess_image

class Dataset():
    def __init__(self, root, img_shape=(256, 192, 3), batch_size: int=16, steps_per_epoch: int=20):
        self.root = root
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.img_shape = img_shape

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def set_steps_per_epoch(self, steps_per_epoch: int):
        self.steps_per_epoch = steps_per_epoch

class MHPDataset(Dataset):
    INT2LABEL = {
        0:  'background',
        1:  'hat',
        2:  'hair',
        3:  'sunglass',
        4:  'upper-clothes',
        5:  'skirt',
        6:  'pants',
        7:  'dress',
        8:  'belt',
        9:  'left-shoe',
        10: 'right-shoe',
        11: 'face',
        12: 'left-leg',
        13: 'right-leg',
        14: 'left-arm',
        15: 'right-arm',
        16: 'bag',
        17: 'scarf',
        18: 'torso-skin'
    }

    LABEL2INT = {v: k for k, v in INT2LABEL.items()}

    def __init__(self, root, batch_size: int=16):
        super().__init__(root, batch_size)
        self.label_dir = os.path.join(self.root, "annotations")
        self.images_dir = os.path.join(self.root, "images")
        self.train_list = os.path.join(self.root, "train_list.txt")
        self.test_list = os.path.join(self.root, "test_list.txt")
        self.train_name, self.test_name = self.__get_data_path_raw()
    
    def __get_data_path_raw(self):
        train = []
        test = []

        with open(self.train_list, 'r') as f:
            for line in f:
                train.append(line.strip())
        with open(self.test_list, 'r') as f:
            for line in f:
                test.append(line.strip())
        return np.asarray(train), np.asarray(test)

    def show_random_img(self):
        r = random.randint(0, self.train_name.shape[0])
        img = np.asarray(Image.open(os.path.join(self.images_dir, self.train_name[r])))
        print(f"Image shape: {img.shape}")
        show_img(img)
        label = np.asarray(Image.open(os.path.join(self.label_dir, self.train_name[r])))
        print(f"Label Shape: {label.shape}")
        show_img(label)

    def train_generator(self):
        pass


class ATRDataset(Dataset):

    LABEL2INT = {
        'background'     :0,
        'hat'            :1,
        'hair'           :2,
        'sunglass'       :3,
        'upper-clothes'  :4,
        'skirt'          :5,
        'pants'          :6,
        'dress'          :7,
        'belt'           :8,
        'left-shoe'      :9,
        'right-shoe'     :10,
        'face'           :11,
        'left-leg'       :12,
        'right-leg'      :13,
        'left-arm'       :14,
        'right-arm'      :15,
        'bag'            :16,
        'scarf'          :17
    }

    INT2LABEL = {v: k for k, v in LABEL2INT.items()}

    IMG_EXT = '.jpg'
    LABEL_EXT = '.png'

    def __init__(self, root: str, img_shape=(256, 192, 3)):
        super().__init__(root, img_shape=img_shape)
        self.label_dir = os.path.join(self.root, "SegmentationClassAug")
        self.image_dir = os.path.join(self.root, "JPEGImages")
        self.train_name, self.test_name = self.__get_data_path_raw()
        self.set_steps_per_epoch(self.train_name.shape[0] // self.batch_size)

    def __get_data_path_raw(self, test_split: float=0.25) -> np.ndarray:
        data = []
        for img_name, label_name in zip(
            os.listdir(self.image_dir), 
            os.listdir(self.label_dir)
        ):
            assert img_name[:-4] == label_name[:-4], "Wrong file name"
            data.append([img_name, label_name])
        data = np.asarray(data)
        train, test = train_test_split(data, test_size=test_split)
        return train, test

    def show_random_img(self):
        r = random.randint(0, self.train_name.shape[0])
        img = np.asarray(Image.open(os.path.join(self.image_dir, self.train_name[r])))
        print(f"Image shape: {img.shape}")
        show_img(img)
        label = np.asarray(Image.open(os.path.join(self.label_dir, self.train_name[r])))
        print(f"Label Shape: {label.shape}")
        show_img(label)

    def __train_generator(self):
        # Return a tensorflow batch dataset
        for id, [img_name, label_name] in enumerate(self.train_name):
            # print(img_name)
            pil_img = Image.open(
                os.path.join(self.image_dir, img_name)
            )
            img = np.asarray(pil_img)
            
            # If the image is mono-color, expand dims and convert it to rgb
            if len(img.shape) == 2:
                # replace alpha channel with white color
                img = Image.new('RGB', pil_img.size, (255, 255, 255))
                img.paste(pil_img)
                img = np.asarray(img)

            # Preprocess image
            img = preprocess_image(
                img, self.img_shape[:2]
            )

            # LAbel is 2d so we broadcast it
            label = np.expand_dims(
                Image.open(
                    os.path.join(self.label_dir, label_name)
                ),
                axis=-1
            )
            label = tf.image.resize(
                label,
                self.img_shape[:2],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            yield img, label
    
    def __test_generator(self):
        # Return a tensorflow batch dataset
        for id, [img_name, label_name] in enumerate(self.test_name):
            # print(img_name)

            pil_img = Image.open(
                os.path.join(self.image_dir, img_name)
            )
            img = np.asarray(pil_img)
            
            # If the image is mono-color, expand dims and convert it to rgb
            if len(img.shape) == 2:
                # replace alpha channel with white color
                img = Image.new('RGB', pil_img.size, (255, 255, 255))
                img.paste(pil_img)
                img = np.asarray(img)

            # Preprocess image
            img = preprocess_image(
                img, self.img_shape[:2]
            )

            # LAbel is 2d so we broadcast it
            label = np.expand_dims(
                Image.open(
                    os.path.join(self.label_dir, label_name)
                ),
                axis=-1
            )
            label = tf.image.resize(
                label,
                self.img_shape[:2],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            yield img, label
    
    def get_tf_dataset(self, dataset='train'):
        if dataset == 'train':
            dataset = tf.data.Dataset.from_generator(
                self.__train_generator,
                output_signature=(
                    tf.TensorSpec(shape=self.img_shape, dtype=tf.float32),
                    tf.TensorSpec(shape=(*self.img_shape[:2], 1), dtype=tf.float32)
                )
            )
            return dataset
        elif dataset == 'test':
            dataset = tf.data.Dataset.from_generator(
                self.__test_generator,
                output_signature=(
                    tf.TensorSpec(shape=self.img_shape, dtype=tf.float32),
                    tf.TensorSpec(shape=(*self.img_shape[:2], 1), dtype=tf.float32)
                )
            )
            return dataset
        else:
            print("Wrong dataset name")

    def get_tf_train_batch_dataset(self):
        dataset = self.get_tf_dataset()
        batch_dataset = dataset.batch(self.batch_size).repeat()
        return batch_dataset

    def get_tf_test_batch_dataset(self):
        dataset = self.get_tf_dataset(dataset='test')
        batch_dataset = dataset.batch(self.batch_size).repeat()
        return batch_dataset

    def get_tf_train_dataset_iter(self):
        dataset = self.get_tf_dataset()
        it = iter(dataset)
        return it
    
    def get_tf_test_dataset_iter(self):
        dataset = self.get_tf_dataset(dataset='test')
        it = iter(dataset)
        return it

class VtonPrep(Dataset):
    
    """[summary]

    Returns:
        [type]: [description]
    """

    LABEL2INT = {
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

    INT2LABEL = {v: k for k, v in LABEL2INT.items()}

    FEATURE_DIR_NAME = ['cloth', 'cloth-mask', 'image', 'image-parse', 'pose']

    def __init__(self, root, img_shape=(256, 192, 3)):
        self.root = root
        self.img_shape = img_shape
        self.train_dir = os.path.join(root, "train")
        self.test_dir = os.path.join(root, "test")
        self.train_path = self.__get_data_path("train")
        self.test_path = self.__get_data_path("test")

    def __get_image_path_only(self, dir_name):
        names = []
        for f_name in os.listdir(os.path.join(self.root, dir_name, "image")):
            names.append(os.path.join(self.root, dir_name, "image", f_name))
        return np.asarray(names)

    def __get_image_name_only(self, dir_name):
        names = []
        for f_name in os.listdir(os.path.join(self.root, dir_name, "image")):
            names.append(f_name)
        return np.asarray(names)

    def __get_data_path(self, dir_name: str):
        data = []
        file_names = self.__get_image_name_only(dir_name)
        for img_name in file_names:
            tmp = {k: os.path.join(self.root, dir_name, k, img_name) for k in VtonPrep.FEATURE_DIR_NAME}
            # Hotfix extension: Change .jpg to .png
            tmp['image-parse'] = tmp['image-parse'][:-3] + "png"
            data.append(tmp)
        return data

    def __generator(self, pipeline="train"):

        paths = None
        if pipeline == "train":
            paths = self.train_path
        elif pipeline == "test":
            paths = self.test_path
        else:
            print("Wrong pipeline. Only 'train' or 'test' can be passed")
            return None

        # Return a tensorflow batch dataset
        for _, sample_data in enumerate(paths):

            sample_img = np.asarray(Image.open(sample_data['image']))
            sample_img = preprocess_image(sample_img)

            sample_cloth = np.asarray(Image.open(sample_data['cloth']))
            sample_cloth = preprocess_image(sample_cloth)

            sample_cloth_mask = np.asarray(Image.open(sample_data['cloth-mask']))
            sample_cloth_mask = preprocess_image(sample_cloth_mask, tanh_range=False)

            sample_parse = tf.expand_dims(np.asarray(Image.open(sample_data['image-parse'])), axis=-1)

            # sample_body_mask shape (256, 192, 1). Range [0, 19]. Representing classes.
            body_masking_channels = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
            sample_body_mask = [sample_parse == c for c in body_masking_channels]
            sample_body_mask = tf.reduce_any(sample_body_mask, axis=0)
            sample_body_mask = tf.cast(sample_body_mask, dtype=tf.float32)

            # Take face hair
            face_hair_masking_channels = [1, 2, 13]
            sample_face_hair_mask = [sample_parse == c for c in face_hair_masking_channels]
            sample_face_hair_mask = tf.reduce_any(sample_face_hair_mask, axis=0)
            sample_face_hair_mask = tf.cast(sample_face_hair_mask, dtype=tf.float32)
            sample_face_hair = sample_img * sample_face_hair_mask

            # Take actual clothing mask in the person
            clothing_masking_channels = [5, 6, 7, 12]
            sample_clothing_mask = [sample_parse == c for c in clothing_masking_channels]
            sample_clothing_mask = tf.reduce_any(sample_clothing_mask, axis=0)
            sample_clothing_mask = tf.cast(sample_clothing_mask, dtype=tf.float32)

            # Pose
            sample_pose = np.asarray(Image.open(sample_data['pose']))
            sample_pose = preprocess_image(sample_pose)

            yield {
                'image': sample_img,
                'cloth': sample_cloth,
                'cloth-mask': sample_cloth_mask,
                'body-mask': sample_body_mask,
                'face-hair': sample_face_hair,
                'actual-cloth-mask': sample_clothing_mask,
                'pose': sample_pose
            }

    def get_tf_train_dataset(self):
        ds = tf.data.Dataset.from_generator(
            self.__generator,
            args=("train"),
            output_signature=({
                'image': tf.TensorSpec(self.img_shape),
                'cloth': tf.TensorSpec(self.img_shape),
                'cloth-mask': tf.TensorSpec((*self.img_shape[:2], 1)),
                'body-mask': tf.TensorSpec((*self.img_shape[:2], 1)),
                'face-hair': tf.TensorSpec(self.img_shape),
                'actual-cloth-mask': tf.TensorSpec((*self.img_shape[:2], 1)),
                'pose': tf.TensorSpec(self.img_shape),
            })
        )
        return ds

    def get_tf_test_dataset(self):
        ds = tf.data.Dataset.from_generator(
            self.__generator,
            args=("test"),
            output_signature=({
                'image': tf.TensorSpec(self.img_shape),
                'cloth': tf.TensorSpec(self.img_shape),
                'cloth-mask': tf.TensorSpec((*self.img_shape[:2], 1)),
                'body-mask': tf.TensorSpec((*self.img_shape[:2], 1)),
                'face-hair': tf.TensorSpec(self.img_shape),
                'actual-cloth-mask': tf.TensorSpec((*self.img_shape[:2], 1)),
                'pose': tf.TensorSpec(self.img_shape),
            })
        )
        return ds
    
    def get_random_data(self, dir_name="train"):
        if dir_name == "train":
            r = random.randint(0, len(self.train_path) - 1)
            return self.train_path[r]
        elif dir_name == "test":
            r = random.randint(0, len(self.test_path) - 1)
            return self.test_path[r]
        print("Wrong forlder format")
        return None
    
    def visualize_random_data(self):
        sample_data = self.get_random_data()

        sample_img = np.asarray(Image.open(sample_data['image']))
        sample_img = preprocess_image(sample_img)
        print(f"Shape: {sample_img.shape}")
        print(f"Range: [{tf.reduce_min(sample_img)}, {tf.reduce_max(sample_img)}]")
        show_img(sample_img)

        sample_cloth = np.asarray(Image.open(sample_data['cloth']))
        sample_cloth = preprocess_image(sample_cloth)
        print(f"Shape: {sample_cloth.shape}")
        print(f"Range: [{tf.reduce_min(sample_cloth)}, {tf.reduce_max(sample_cloth)}]")
        show_img(sample_cloth)

        sample_cloth_mask = np.asarray(Image.open(sample_data['cloth-mask']))
        sample_cloth_mask = preprocess_image(sample_cloth_mask, tanh_range=False)
        print(f"Shape: {sample_cloth_mask.shape}")
        print(f"Range: [{tf.reduce_min(sample_cloth_mask)}, {tf.reduce_max(sample_cloth_mask)}]")
        show_img(sample_cloth_mask)

        sample_parse = tf.expand_dims(np.asarray(Image.open(sample_data['image-parse'])), axis=-1)
        print(f"Shape: {sample_parse.shape}")
        print(f"Range: [{tf.reduce_min(sample_parse)}, {tf.reduce_max(sample_parse)}]")
        show_img(sample_parse)

        # sample_body_mask shape (256, 192, 1). Range [0, 19]. Representing classes.
        body_masking_channels = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
        sample_body_mask = [sample_parse == c for c in body_masking_channels]
        sample_body_mask = tf.reduce_any(sample_body_mask, axis=0)
        sample_body_mask = tf.cast(sample_body_mask, dtype=tf.float32)
        show_img(sample_body_mask)

        # Take face hair
        face_hair_masking_channels = [1, 2, 13]
        sample_face_hair_mask = [sample_parse == c for c in face_hair_masking_channels]
        sample_face_hair_mask = tf.reduce_any(sample_face_hair_mask, axis=0)
        sample_face_hair_mask = tf.cast(sample_face_hair_mask, dtype=tf.float32)
        sample_face_hair = sample_img * sample_face_hair_mask
        show_img(sample_face_hair)

        # Take cloth mask
        clothing_masking_channels = [5, 6, 7, 12]
        sample_clothing_mask = [sample_parse == c for c in clothing_masking_channels]
        sample_clothing_mask = tf.reduce_any(sample_clothing_mask, axis=0)
        sample_clothing_mask = tf.cast(sample_clothing_mask, dtype=tf.float32)
        show_img(sample_clothing_mask)

        # Pose
        sample_pose = np.asarray(Image.open(sample_data['pose']))
        sample_pose = preprocess_image(sample_pose)
        print(f"Shape: {sample_pose.shape}")
        print(f"Range: [{tf.reduce_min(sample_pose)}, {tf.reduce_max(sample_pose)}]")
        show_img(sample_pose)