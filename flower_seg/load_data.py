#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/10 10:28
# @File    : load_data.py
# @Author  : NUS_LuoKe

import os

from skimage import io
from skimage.transform import resize


def seg2mask(segmentation_dir, mask_save_dir):
    '''
    segmentation given by the data set is flower instead of mask of the flower.
    generate mask from their corresponding segmentation.
    '''
    if not os.path.isdir(mask_save_dir):
        os.mkdir(mask_save_dir)

    for seg in os.listdir(segmentation_dir):
        seg_path = os.path.join(segmentation_dir, seg)
        count = os.path.basename(seg_path).split("_")[1]
        seg_array = io.imread(seg_path, as_gray=True)

        # the pixel value of the background in the gray image is 0.07181725490196078
        # Can use skimage.color.rgb2gray()
        seg_array[seg_array > 0.15] = 1
        io.imsave(fname=os.path.join(mask_save_dir, "mask_{}".format(count)), arr=seg_array)


def resize_image(image_dir, resized_image_save_dir, prefix, output_shape):
    if not os.path.isdir(resized_image_save_dir):
        os.mkdir(resized_image_save_dir)
    for image in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image)
        count = os.path.basename(image_path).split("_")[1]

        img_arr = io.imread(image_path)
        re_img_arr = resize(img_arr, output_shape)
        io.imsave(fname=os.path.join(resized_image_save_dir, prefix + "_{}".format(count)), arr=re_img_arr)


if __name__ == '__main__':
    # if mask does not exist, generate mask from segmentation
    segmentation_dir = "../data_set/segmentation"
    mask_save_dir = "../data_set/mask"
    if not os.path.exists(mask_save_dir):
        seg2mask(segmentation_dir, mask_save_dir)

    #################################################
    mask_dir = "../data_set/mask"
    resized_mask_save_dir = "../data_set/resized_mask"
    if not os.path.exists(resized_mask_save_dir):
        resize_image(mask_dir, resized_mask_save_dir, prefix="resized_mask", output_shape=(400, 400))

    #################################################
    flower_dir = "../data_set/ori_image"
    resized_flower_save_dir = "../data_set/resized_flower"
    if not os.path.exists(resized_flower_save_dir):
        resize_image(flower_dir, resized_flower_save_dir, prefix="resized_flower", output_shape=(400, 400))
