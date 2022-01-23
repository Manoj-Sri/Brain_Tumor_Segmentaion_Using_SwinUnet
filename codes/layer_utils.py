
from __future__ import absolute_import


from tensorflow import expand_dims
from tensorflow.compat.v1 import image
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D, Conv2DTranspose, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply, add
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU, Softmax



def CONV_output(X, n_labels, kernel_size=1, activation='Softmax', name='conv_output'):

    X = Conv2D(n_labels, kernel_size, padding='same', use_bias=True, name=name)(X)

    if activation:

        if activation == 'Sigmoid':
            X = Activation('sigmoid', name='{}_activation'.format(name))(X)

        else:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)

    return X
