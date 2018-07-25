#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/11 17:58
# @File    : inference_one_image.py
# @Author  : NUS_LuoKe

import os

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.models import load_model
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


if __name__ == '__main__':
    image_path = "../readme_img/3.jpg"
    target_size = (256, 256)

    img_arr = io.imread(image_path)
    img_arr = img_arr / 255
    re_img_arr = resize(img_arr, target_size)
    plt.imshow(re_img_arr)
    plt.show()
    gray_img = rgb2gray(re_img_arr)

    model_input = np.reshape(gray_img, (1, 256, 256, 1))
    plt.imshow(gray_img, cmap="gray")
    plt.show()

    model_dir = '../models'
    model_name = 'model_1.h5'
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss, "dice_coef": dice_coef})
    print('model loaded')

    pred = model.predict(model_input, batch_size=1)
    output_img = np.squeeze(pred)
    plt.imshow(output_img, cmap="gray")
    plt.show()
