#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/10 15:44
# @File    : train.py
# @Author  : NUS_LuoKe

from keras.preprocessing.image import ImageDataGenerator
import keras
from flower_seg import fcn_models

# pre-settings
target_size = (64, 64)
batch_size = 32
epochs = 50
# load model
input_shape = (64, 64, 3)
model = fcn_models.unet(input_shape=input_shape)

# we create two instances with the same arguments
data_gen_args = dict(rescale=1. / 255,
                     rotation_range=90.,
                     horizontal_flip=True,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1

image_generator = image_datagen.flow_from_directory(
    directory=train_dir,
    target_size=target_size,
    batch_size=batch_size,
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
