#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/11 17:58
# @File    : seg_in_image.py
# @Author  : NUS_LuoKe

import os

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage.transform import resize
from skimage import io
import matplotlib.pyplot as plt

image_path = ""
target_size = (64, 64)

img_arr = io.imread(image_path)
img_arr = img_arr / 255
re_img_arr = resize(img_arr, target_size)
plt.imshow(re_img_arr)
plt.show()

model_dir = './models'
model_name = 'model_1.h5'
model_path = os.path.join(model_dir, model_name)
model = load_model(model_path)
print('model loaded')

model.predict(image_path, batch_size=1)
