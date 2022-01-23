import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os, time


def random_flip_rotate(data):
    X, y = data
    # x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],  # previous without this, hard-dice=83.7
    #                         axis=0, is_random=True) # up down
    X, y = tl.prepro.flip_axis_multi([X, y],
                            axis=1, is_random=True) # left right

    return X,y
