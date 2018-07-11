#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/11 17:51
# @File    : visualization_util.py
# @Author  : NUS_LuoKe

import matplotlib.pyplot as plt


def plot_acc_loss(h, nb_epoch):
    '''
    :param h: history, it is the return value of "fit()", h = model.fit()
    :param nb_epoch: number of epochs
    :return: plot a figure of accuracy and loss of very epoch
    '''
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(nb_epoch), acc, label='Train')
    plt.plot(range(nb_epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(nb_epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(nb_epoch), loss, label='Train')
    plt.plot(range(nb_epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(nb_epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()
