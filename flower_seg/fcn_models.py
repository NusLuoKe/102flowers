#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/10 15:42
# @File    : fcn_models.py
# @Author  : NUS_LuoKe

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import Adam


def get_unet_128(input_shape=(128, 128, 1), num_classes=1, learning_rate=1e-4, loss_func='binary_crossentropy',
                 metrics=["accuracy"]):
    inputs = Input(shape=input_shape)
    # 128

    down1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    down1 = Conv2D(64, (3, 3), activation='relu', padding='same')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), activation='relu', padding='same')(down1_pool)
    down2 = Conv2D(128, (3, 3), activation='relu', padding='same')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), activation='relu', padding='same')(down2_pool)
    down3 = Conv2D(256, (3, 3), activation='relu', padding='same')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), activation='relu', padding='same')(down3_pool)
    down4 = Conv2D(512, (3, 3), activation='relu', padding='same')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), activation='relu', padding='same')(down4_pool)
    center = Conv2D(1024, (3, 3), activation='relu', padding='same')(center)
    # center

    up4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), activation='relu', padding='same')(up4)
    up4 = Conv2D(512, (3, 3), activation='relu', padding='same')(up4)
    # 16

    up3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    # 32

    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    # 64

    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    # 128

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=Adam(lr=learning_rate), loss=loss_func, metrics=metrics)

    return model


def get_unet_128_bn(input_shape=(128, 128, 1), num_classes=1, learning_rate=1e-4, loss_func='binary_crossentropy',
                    metrics=["accuracy"]):
    inputs = Input(shape=input_shape)
    # 128

    down1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), activation='relu', padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), activation='relu', padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), activation='relu', padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), activation='relu', padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), activation='relu', padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), activation='relu', padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), activation='relu', padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), activation='relu', padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), activation='relu', padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), activation='relu', padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), activation='relu', padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=Adam(lr=learning_rate), loss=loss_func, metrics=metrics)

    return model


def get_unet_256(input_shape=(128, 128, 1), num_classes=1, learning_rate=1e-4, loss_func='binary_crossentropy',
                 metrics=["accuracy"]):
    inputs = Input(shape=input_shape)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(inputs)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    # center

    up4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    # 16

    up3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    # 32

    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    # 64

    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    # 128

    up0 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    # 256

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=Adam(lr=learning_rate), loss=loss_func, metrics=metrics)

    return model
