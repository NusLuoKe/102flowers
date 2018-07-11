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

# pre-settings
train_dir = ""
mask_dir = ""
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
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    directory=mask_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

start = time.time()
h = model.fit_generator(train_generator, epochs=50, verbose=1)

model_path = './models'
model_name = 'model_1.h5'
weights_path = os.path.join(model_path, model_name)
if not os.path.isdir(model_path):
    os.makedirs(model_path)

model.save(weights_path)
end = time.time()
time_spend = end - start
print('@ Overall time spend is %.2f seconds.' % time_spend)

# plot figures of accuracy and loss of every epoch and a visible test result
plot_acc_loss(h, epochs)
