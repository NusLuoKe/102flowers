#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/10 15:44
# @File    : train.py
# @Author  : NUS_LuoKe

from keras.preprocessing.image import ImageDataGenerator
from flower_seg import fcn_models
import os
import time
from flower_seg.visualization_util import plot_acc_loss
import random
import itertools

# pre-settings
# Define training directories (raw training images and their corresponding masks)
train_image_dir = "../data_set/input/train_flower/"
train_mask_dir = "../data_set/input/train_mask/"

# Define testing directories (raw testing/validation images and their corresponding masks)
test_image_dir = "../data_set/input/test_flower/"
test_mask_dir = "../data_set/input/test_mask/"

target_size = (64, 64)
batch_size = 32
epochs = 50

# load model
input_shape = (64, 64, 1)
model = fcn_models.get_unet_128(input_shape=input_shape)

# we create two instances with the same arguments
train_data_gen_args = dict(rescale=1. / 255,
                           rotation_range=90.,
                           horizontal_flip=True,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.2)

test_data_gen_args = dict(rescale=1. / 255)

train_image_datagen = ImageDataGenerator(**train_data_gen_args)
train_mask_datagen = ImageDataGenerator(**train_data_gen_args)
test_image_datagen = ImageDataGenerator(**test_data_gen_args)
test_mask_datagen = ImageDataGenerator(**test_data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = random.randint(1, 100)

train_image_generator = train_image_datagen.flow_from_directory(
    directory=train_image_dir,
    color_mode="grayscale",
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    seed=seed)

train_mask_generator = train_mask_datagen.flow_from_directory(
    directory=train_mask_dir,
    color_mode="grayscale",
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(train_image_generator, train_mask_generator)

# test image generator
test_image_generator = test_image_datagen.flow_from_directory(
    directory=test_image_dir,
    color_mode="grayscale",
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    seed=seed)

test_mask_generator = test_mask_datagen.flow_from_directory(
    directory=test_mask_dir,
    color_mode="grayscale",
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
test_generator = zip(test_image_generator, test_mask_generator)

start = time.time()
# TODO：genetator是空的
h = model.fit_generator(generator=train_generator, steps_per_epoch=2000, epochs=50, validation_data=test_generator,
                        validation_steps=5, verbose=1)

model_dir = './models'
model_name = 'model_1.h5'
weights_path = os.path.join(model_dir, model_name)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

model.save(weights_path)
end = time.time()
time_spend = end - start
print('@ Overall time spend is %.2f seconds.' % time_spend)

# plot figures of accuracy and loss of every epoch and a visible test result
plot_acc_loss(h, epochs)
