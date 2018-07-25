#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/10 15:44
# @File    : train.py
# @Author  : NUS_LuoKe

import math
import os
import random
import time

from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from flower_seg import fcn_models
from flower_seg.visualization_util import plot_acc_loss


def dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# GPU limit
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define training directories (raw training images and their corresponding masks)
train_image_dir = "../data_set/input/train_flower/"
train_mask_dir = "../data_set/input/train_mask/"

# Define testing directories (raw testing/validation images and their corresponding masks)
test_image_dir = "../data_set/input/test_flower/"
test_mask_dir = "../data_set/input/test_mask/"

early_stopping_patience = 10
input_shape = (256, 256, 1)
batch_size = 32
epochs = 100
validation_steps = 20
steps_per_epoch = math.ceil(7500 / batch_size)
learning_rate = 1e-4
cont_training = True

# load model
if cont_training:
    try:
        model_path = "../flower_1"
        model = load_model(model_path)
        print("load pre-trained model")
    except:
        model = fcn_models.get_unet_128(input_shape=input_shape, learning_rate=learning_rate, loss_func=dice_coef_loss,
                                        metrics=[dice_coef])
        print("Pre-trained model not found, initialized a new model!")
else:
    model = fcn_models.get_unet_128(input_shape=input_shape, learning_rate=learning_rate, loss_func=dice_coef_loss,
                                    metrics=[dice_coef])
    print("new model ready!")

target_size = (input_shape[0], input_shape[1])
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

if __name__ == '__main__':
    start = time.time()

    early_stopping_monitor = EarlyStopping(patience=early_stopping_patience)
    h = model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                            validation_data=test_generator, validation_steps=validation_steps, verbose=1,
                            callbacks=[early_stopping_monitor])

    model_dir = './models'
    model_name = 'model_1.h5'
    weights_path = os.path.join(model_dir, model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model.save(weights_path)
    end = time.time()
    time_spend = end - start
    print('@ Overall time spend is %.2f seconds.' % time_spend)

    # # plot figures of accuracy and loss of every epoch and a visible test result
    # plot_acc_loss(h, epochs)
