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
    