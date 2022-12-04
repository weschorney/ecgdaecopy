#============================================================
#
#  Deep Learning BLW Filtering
#  Deep Learning models
#
#  author: Francisco Perdigon Romero and Wesley Chorney
#  email: wes.chorney@gmail.com
#  github id: weschorney
#
#===========================================================


import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Dropout, BatchNormalization,\
                         concatenate, Input, Conv2DTranspose, Lambda, LSTM,\
                         Layer, MaxPool1D, Conv1DTranspose, Flatten

import keras.backend as K

def Conv1DTranspose2(input_tensor, filters, kernel_size, strides=2, activation='relu', padding='same'):
    """
        https://stackoverflow.com/a/45788699

        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters,
                        kernel_size=(kernel_size, 1),
                        activation=activation,
                        strides=(strides, 1),
                        padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

##########################################################################

###### MODULES #######

def LFilter_module(x, layers):
    LB0 = Conv1D(filters=int(layers / 4),
                 kernel_size=3,
                 activation='linear',
                 strides=1,
                 padding='same')(x)
    LB1 = Conv1D(filters=int(layers / 4),
                kernel_size=5,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB2 = Conv1D(filters=int(layers / 4),
                kernel_size=9,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB3 = Conv1D(filters=int(layers / 4),
                kernel_size=15,
                activation='linear',
                strides=1,
                padding='same')(x)


    x = concatenate([LB0, LB1, LB2, LB3])

    return x


def NLFilter_module(x, layers):

    NLB0 = Conv1D(filters=int(layers / 4),
                  kernel_size=3,
                  activation='relu',
                  strides=1,
                  padding='same')(x)
    NLB1 = Conv1D(filters=int(layers / 4),
                kernel_size=5,
                activation='relu',
                strides=1,
                padding='same')(x)
    NLB2 = Conv1D(filters=int(layers / 4),
                kernel_size=9,
                activation='relu',
                strides=1,
                padding='same')(x)
    NLB3 = Conv1D(filters=int(layers / 4),
                kernel_size=15,
                activation='relu',
                strides=1,
                padding='same')(x)


    x = concatenate([NLB0, NLB1, NLB2, NLB3])

    return x


def LANLFilter_module(x, layers):
    LB0 = Conv1D(filters=int(layers / 8),
                 kernel_size=3,
                 activation='linear',
                 strides=1,
                 padding='same')(x)
    LB1 = Conv1D(filters=int(layers / 8),
                kernel_size=5,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB2 = Conv1D(filters=int(layers / 8),
                kernel_size=9,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB3 = Conv1D(filters=int(layers / 8),
                kernel_size=15,
                activation='linear',
                strides=1,
                padding='same')(x)

    NLB0 = Conv1D(filters=int(layers / 8),
                  kernel_size=3,
                  activation='relu',
                  strides=1,
                  padding='same')(x)
    NLB1 = Conv1D(filters=int(layers / 8),
                 kernel_size=5,
                 activation='relu',
                 strides=1,
                 padding='same')(x)
    NLB2 = Conv1D(filters=int(layers / 8),
                 kernel_size=9,
                 activation='relu',
                 strides=1,
                 padding='same')(x)
    NLB3 = Conv1D(filters=int(layers / 8),
                 kernel_size=15,
                 activation='relu',
                 strides=1,
                 padding='same')(x)

    x = concatenate([LB0, LB1, LB2, LB3, NLB0, NLB1, NLB2, NLB3])

    return x


def LANLFilter_module_dilated(x, layers):
    LB1 = Conv1D(filters=int(layers / 6),
                kernel_size=5,
                activation='linear',
                dilation_rate=3,
                padding='same')(x)
    LB2 = Conv1D(filters=int(layers / 6),
                kernel_size=9,
                activation='linear',
                dilation_rate=3,
                padding='same')(x)
    LB3 = Conv1D(filters=int(layers / 6),
                kernel_size=15,
                dilation_rate=3,
                activation='linear',
                padding='same')(x)

    NLB1 = Conv1D(filters=int(layers / 6),
                 kernel_size=5,
                 activation='relu',
                 dilation_rate=3,
                 padding='same')(x)
    NLB2 = Conv1D(filters=int(layers / 6),
                 kernel_size=9,
                 activation='relu',
                 dilation_rate=3,
                 padding='same')(x)
    NLB3 = Conv1D(filters=int(layers / 6),
                 kernel_size=15,
                 dilation_rate=3,
                 activation='relu',
                 padding='same')(x)

    x = concatenate([LB1, LB2, LB3, NLB1, NLB2, NLB3])
    # x = BatchNormalization()(x)

    return x


###### MODELS #######

def deep_filter_vanilla_linear(signal_size=512):  # signal_size=None to use any size input

    model = Sequential()

    model.add(Conv1D(filters=64,
                     kernel_size=9,
                     activation='linear',
                     input_shape=(signal_size, 1),
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=64,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=32,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=32,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=16,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=16,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=1,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))
    return model


def deep_filter_vanilla_Nlinear(signal_size=512):
    model = Sequential()

    model.add(Conv1D(filters=64,
                     kernel_size=9,
                     activation='relu',
                     input_shape=(signal_size, 1),
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=64,
                     kernel_size=9,
                     activation='relu',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=32,
                     kernel_size=9,
                     activation='relu',
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=32,
                     kernel_size=9,
                     activation='relu',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=16,
                     kernel_size=9,
                     activation='relu',
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=16,
                     kernel_size=9,
                     activation='relu',
                     strides=1,
                     padding='same'))


    model.add(Conv1D(filters=1,
                     kernel_size=9,
                     activation='linear',
                     strides=1,
                     padding='same'))
    return model


def deep_filter_I_linear(signal_size=512):
    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)

    tensor = LFilter_module(input, 64)
    tensor = LFilter_module(tensor, 64)
    tensor = LFilter_module(tensor, 32)
    tensor = LFilter_module(tensor, 32)
    tensor = LFilter_module(tensor, 16)
    tensor = LFilter_module(tensor, 16)
    predictions = Conv1D(filters=1,
                         kernel_size=9,
                         activation='linear',
                         strides=1,
                         padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model


def deep_filter_I_Nlinear(signal_size=512):
    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)

    tensor = NLFilter_module(input, 64)
    tensor = NLFilter_module(tensor, 64)
    tensor = NLFilter_module(tensor, 32)
    tensor = NLFilter_module(tensor, 32)
    tensor = NLFilter_module(tensor, 16)
    tensor = NLFilter_module(tensor, 16)
    predictions = Conv1D(filters=1,
                         kernel_size=9,
                         activation='linear',
                         strides=1,
                         padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model


def deep_filter_I_LANL(signal_size=512):
    # TODO: Make the doc

    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)

    tensor = LANLFilter_module(input, 64)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 64)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 32)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 32)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 16)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 16)
    tensor = BatchNormalization()(tensor)
    predictions = Conv1D(filters=1,
                    kernel_size=9,
                    activation='linear',
                    strides=1,
                    padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model


def deep_filter_model_I_LANL_dilated(signal_size=512):
    # TODO: Make the doc

    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)

    tensor = LANLFilter_module(input, 64)
    tensor = Dropout(0.4)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module_dilated(tensor, 64)
    tensor = Dropout(0.4)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 32)
    tensor = Dropout(0.4)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module_dilated(tensor, 32)
    tensor = Dropout(0.4)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 16)
    tensor = Dropout(0.4)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module_dilated(tensor, 16)
    tensor = Dropout(0.4)(tensor)
    tensor = BatchNormalization()(tensor)
    predictions = Conv1D(filters=1,
                    kernel_size=9,
                    activation='linear',
                    strides=1,
                    padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model


def FCN_DAE(signal_size=512):
    # Implementation of FCN_DAE approach presented in
    # Chiang, H. T., Hsieh, Y. Y., Fu, S. W., Hung, K. H., Tsao, Y., & Chien, S. Y. (2019).
    # Noise reduction in ECG signals using fully convolutional denoising autoencoders.
    # IEEE Access, 7, 60806-60813.

    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)

    x = Conv1D(filters=40,
               input_shape=(512, 1),
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(input)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=40,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=1,
               kernel_size=16,
               activation='elu',
               strides=1,
               padding='same')(x)

    x = BatchNormalization()(x)

    # Keras has no 1D Traspose Convolution, instead we use Conv2DTranspose function
    # in a souch way taht is mathematically equivalent
    x = Conv1DTranspose2(input_tensor=x,
                        filters=1,
                        kernel_size=16,
                        activation='elu',
                        strides=1,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=40,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=40,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    predictions = Conv1DTranspose2(input_tensor=x,
                        filters=1,
                        kernel_size=16,
                        activation='linear',
                        strides=1,
                        padding='same')

    model = Model(inputs=[input], outputs=predictions)
    return model

def CNN_DAE(signal_size=512):
    # Implementation of FCN_DAE approach presented in
    # Chiang, H. T., Hsieh, Y. Y., Fu, S. W., Hung, K. H., Tsao, Y., & Chien, S. Y. (2019).
    # Noise reduction in ECG signals using fully convolutional denoising autoencoders.
    # IEEE Access, 7, 60806-60813.

    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)

    x = Conv1D(filters=40,
               input_shape=(512, 1),
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(input)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=40,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=1,
               kernel_size=16,
               activation='elu',
               strides=1,
               padding='same')(x)

    x = BatchNormalization()(x)

    # Keras has no 1D Traspose Convolution, instead we use Conv2DTranspose function
    # in a souch way taht is mathematically equivalent
    x = Conv1DTranspose2(input_tensor=x,
                        filters=1,
                        kernel_size=16,
                        activation='elu',
                        strides=1,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=40,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=1,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')
    x = Flatten()(x)
    x = BatchNormalization()(x)

    x = Dense(signal_size // 2,
              activation='elu')(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    
    predictions = Dense(signal_size, activation='linear')(x)

    model = Model(inputs=[input], outputs=predictions)
    return model

def DRRN_denoising(signal_size=512):
    # Implementation of DRNN approach presented in
    # Antczak, K. (2018). Deep recurrent neural networks for ECG signal denoising.
    # arXiv preprint arXiv:1807.11551.

    model = Sequential()
    model.add(LSTM(64, input_shape=(signal_size, 1), return_sequences=True))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))

    return model

########################################################
############## IMPLEMENT SPATIAL #######################
########################################################

class SpatialGate(Layer):
    def __init__(self, filters, kernel_size, input_shape=None, activation='sigmoid', transpose=False):
        super(SpatialGate, self).__init__()
        self.transpose = transpose
        self.conv = Conv1D(filters, kernel_size, input_shape=input_shape, activation=activation)

    def call(self, x):
        #if transpose, switch the data to (batch, steps, channels)
        if self.transpose:
            x = tf.transpose(x, [0, 2, 1])
        avg_ = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_ = tf.reduce_max(x, axis=-1, keepdims=True)
        x = tf.concat([avg_, max_], axis=1)
        out = self.conv(x)
        if self.transpose:
            out = tf.transpose(out, [0, 2, 1])
        return out

########################################################
############## IMPLEMENT CHANNEL #######################
########################################################

class ChannelGate(Layer):
    def __init__(self, filters, kernel_size, input_shape=None, activation='sigmoid', transpose=False):
        super(ChannelGate, self).__init__()
        self.transpose = transpose
        self.conv = Conv1D(filters, kernel_size,
                           input_shape=input_shape,
                           activation=activation,
                           padding='same')

    def call(self, x):
        #if transpose, switch the data to (batch, steps, channels)
        if self.transpose:
            x = tf.transpose(x, [0, 2, 1])
        x = tf.reduce_mean(x, axis=1, keepdims=True)
        x = tf.transpose(x, [0, 2, 1])
        out = self.conv(x)
        out = tf.transpose(out, [0, 2, 1])
        if self.transpose:
            out = tf.transpose(out, [0, 2, 1])
        return out

########################################################
################# IMPLEMENT CBAM #######################
########################################################

class CBAM(ks.layers.Layer):
    def __init__(self, c_filters, c_kernel, c_input, c_transpose,
                 s_filters, s_kernel, s_input, s_transpose, spatial=True):
        super(CBAM, self).__init__()
        self.spatial = spatial
        self.channel_attention = ChannelGate(c_filters, c_kernel, input_shape=c_input, transpose=c_transpose)
        self.spatial_attention = SpatialGate(s_filters, s_kernel, input_shape=s_input, transpose=s_transpose)

    def call(self, x):
        channel_mask = self.channel_attention(x)
        x = channel_mask * x
        if self.spatial:
            spatial_mask = self.spatial_attention(x)
            x = spatial_mask * x
        return x

class AttentionBlock(Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None, activation='LeakyReLU'):
        super(AttentionBlock, self).__init__()
        if input_size is not None:
            self.conv = Conv1D(
                channels,
                kernel_size,
                input_shape=input_size,
                padding='same',
                activation=activation
            )
        else:
            self.conv = Conv1D(
                channels,
                kernel_size,
                padding='same',
                activation=activation
            )
        self.attention = CBAM(
            1,
            3,
            (channels, 1),
            False,
            1,
            signal_size + 1,
            (signal_size, 1),
            False
        )
        self.maxpool = MaxPool1D(
            padding='same',
            strides=2
        )

    def call(self, x):
        output = self.conv(x)
        output = self.attention(output)
        output = self.maxpool(output)
        return output

class AttentionBlockBN(Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None):
        super(AttentionBlockBN, self).__init__()
        if input_size is not None:
            self.conv = Conv1D(
                channels,
                kernel_size,
                input_shape=input_size,
                padding='same',
                activation=None
            )
        else:
            self.conv = Conv1D(
                channels,
                kernel_size,
                padding='same',
                activation=None
            )
        self.activation = tf.keras.layers.LeakyReLU()
        self.bn = BatchNormalization()
        self.attention = CBAM(
            1,
            3,
            (channels, 1),
            False,
            1,
            signal_size + 1,
            (signal_size, 1),
            False
        )
        self.maxpool = MaxPool1D(
            padding='same',
            strides=2
        )

    def call(self, x):
        output = self.conv(x)
        output = self.activation(self.bn(output))
        output = self.attention(output)
        output = self.maxpool(output)
        return output

class EncoderBlock(Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None, activation='LeakyReLU'):
        super(EncoderBlock, self).__init__()
        if input_size is not None:
            self.conv = Conv1D(
                channels,
                kernel_size,
                input_shape=input_size,
                padding='same',
                activation=activation
            )
        else:
            self.conv = Conv1D(
                channels,
                kernel_size,
                padding='same',
                activation=activation
            )
        self.maxpool = MaxPool1D(
            padding='same',
            strides=2
        )

    def call(self, x):
        output = self.conv(x)
        output = self.maxpool(output)
        return output

class AttentionDeconv(tf.keras.layers.Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None, activation='LeakyReLU',
                 strides=2, padding='same'):
        super(AttentionDeconv, self).__init__()
        self.deconv = Conv1DTranspose(
            channels,
            kernel_size,
            strides=strides,
            padding=padding,
            activation=activation
        )
        self.attention = CBAM(
            1,
            3,
            (channels, 1),
            False,
            1,
            signal_size + 1,
            (signal_size, 1),
            False
        )

    def call(self, x):
        output = self.attention(self.deconv(x))
        return output

class AttentionDeconvBN(tf.keras.layers.Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None, activation='LeakyReLU',
                 strides=2, padding='same'):
        super(AttentionDeconvBN, self).__init__()
        self.deconv = Conv1DTranspose(
            channels,
            kernel_size,
            strides=strides,
            padding=padding,
        )
        self.bn = BatchNormalization()
        if activation == 'LeakyReLU':
            self.activation = tf.keras.layers.LeakyReLU()
        else:
            self.activation = None
        self.attention = CBAM(
            1,
            3,
            (channels, 1),
            False,
            1,
            signal_size + 1,
            (signal_size, 1),
            False
        )

    def call(self, x):
        output = self.deconv(x)
        output = self.bn(output)
        if self.activation is not None:
            output = self.activation(output)
        output = self.attention(output)
        return output


class AttentionDeconvECA(tf.keras.layers.Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None, activation='LeakyReLU',
                 strides=2, padding='same'):
        super(AttentionDeconvECA, self).__init__()
        self.deconv = Conv1DTranspose(
            channels,
            kernel_size,
            strides=strides,
            padding=padding,
            activation=activation
        )
        self.attention = CBAM(
            1,
            3,
            (channels, 1),
            False,
            1,
            signal_size + 1,
            (signal_size, 1),
            False,
            spatial=False
        )

    def call(self, x):
        output = self.attention(self.deconv(x))
        return output

class AttentionSkipDAE(tf.keras.Model):
    def __init__(self, signal_size=512):
        super(AttentionSkipDAE, self).__init__()
        self.b1 = AttentionBlock(signal_size, 16, input_size=(signal_size, 1))
        self.b2 = AttentionBlock(signal_size//2, 32)
        self.b3 = AttentionBlock(signal_size//4, 64)
        self.b4 = AttentionBlock(signal_size//8, 64)
        self.b5 = AttentionBlock(signal_size//16, 1) #32
        self.d5 = AttentionDeconv(signal_size//16, 64)
        self.d4 = AttentionDeconv(signal_size//8, 64)
        self.d3 = AttentionDeconv(signal_size//4, 32)
        self.d2 = AttentionDeconv(signal_size//2, 16)
        self.d1 = AttentionDeconv(signal_size, 1, activation='linear')

    def encode(self, x):
        encoded = self.b1(x)
        encoded = self.b2(encoded)
        encoded = self.b3(encoded)
        encoded = self.b4(encoded)
        encoded = self.b5(encoded)
        return encoded

    def decode(self, x):
        decoded = self.d5(x)
        decoded = self.d4(decoded)
        decoded = self.d3(decoded)
        decoded = self.d2(decoded)
        decoded = self.d1(decoded)
        return decoded

    def call(self, x):
        enc1 = self.b1(x)
        enc2 = self.b2(enc1)
        enc3 = self.b3(enc2)
        enc4 = self.b4(enc3)
        enc5 = self.b5(enc4)
        dec5 = self.d5(enc5)
        dec4 = self.d4(dec5 + enc4)
        dec3 = self.d3(dec4 + enc3)
        dec2 = self.d2(dec3 + enc2)
        dec1 = self.d1(dec2 + enc1)
        return dec1

class ECASkipDAE(tf.keras.Model):
    def __init__(self, signal_size=512):
        super(ECASkipDAE, self).__init__()
        self.b1 = EncoderBlock(signal_size, 16, input_size=(signal_size, 1), kernel_size=13)
        self.b2 = EncoderBlock(signal_size//2, 32, kernel_size=7)
        self.b3 = EncoderBlock(signal_size//4, 64, kernel_size=7)
        self.b4 = EncoderBlock(signal_size//8, 64, kernel_size=7)
        self.b5 = EncoderBlock(signal_size//16, 1, kernel_size=7) #32
        self.d5 = AttentionDeconvECA(signal_size//16, 64, kernel_size=7)
        self.d4 = AttentionDeconvECA(signal_size//8, 64, kernel_size=7)
        self.d3 = AttentionDeconvECA(signal_size//4, 32, kernel_size=7)
        self.d2 = AttentionDeconvECA(signal_size//2, 16, kernel_size=7)
        self.d1 = AttentionDeconvECA(signal_size, 1, activation='linear', kernel_size=13)
        self.dense = ks.layers.Dense(signal_size)

    def encode(self, x):
        encoded = self.b1(x)
        encoded = self.b2(encoded)
        encoded = self.b3(encoded)
        encoded = self.b4(encoded)
        encoded = self.b5(encoded)
        return encoded

    def decode(self, x):
        decoded = self.d5(x)
        decoded = self.d4(decoded)
        decoded = self.d3(decoded)
        decoded = self.d2(decoded)
        decoded = self.d1(decoded)
        return decoded

    def call(self, x):
        enc1 = self.b1(x)
        enc2 = self.b2(enc1)
        enc3 = self.b3(enc2)
        enc4 = self.b4(enc3)
        enc5 = self.b5(enc4)
        dec5 = self.d5(enc5)
        dec4 = self.d4(dec5 + enc4)
        dec3 = self.d3(dec4 + enc3)
        dec2 = self.d2(dec3 + enc2)
        dec1 = self.d1(dec2 + enc1)
        return dec1

class VanillaAutoencoder(tf.keras.Model):
    def __init__(self, signal_size=512, activation='LeakyReLU'):
        super(VanillaAutoencoder, self).__init__()
        self.model = Sequential([
                Flatten(),
                Dense(signal_size // 2, activation=activation),
                Dense(signal_size // 4, activation=activation),
                Dense(signal_size // 8, activation=activation),
                Dense(signal_size // 16, activation=activation),
                Dense(signal_size // 8, activation=activation),
                Dense(signal_size // 4, activation=activation),
                Dense(signal_size // 2, activation=activation),
                Dense(signal_size, activation=activation)
                ])

    def call(self, x):
        return self.model(x)

class AttentionSkipDAE2(tf.keras.Model):
    def __init__(self, signal_size=512):
        super(AttentionSkipDAE2, self).__init__()
        self.b1 = AttentionBlockBN(signal_size, 16, input_size=(signal_size, 1))
        self.b2 = AttentionBlockBN(signal_size//2, 32)
        self.b3 = AttentionBlockBN(signal_size//4, 64)
        self.b4 = AttentionBlockBN(signal_size//8, 64)
        self.b5 = AttentionBlockBN(signal_size//16, 1) #32
        self.d5 = AttentionDeconvBN(signal_size//16, 64)
        self.d4 = AttentionDeconvBN(signal_size//8, 64)
        self.d3 = AttentionDeconvBN(signal_size//4, 32)
        self.d2 = AttentionDeconvBN(signal_size//2, 16)
        self.d1 = AttentionDeconvBN(signal_size, 1, activation='linear')

    def encode(self, x):
        encoded = self.b1(x)
        encoded = self.b2(encoded)
        encoded = self.b3(encoded)
        encoded = self.b4(encoded)
        encoded = self.b5(encoded)
        return encoded

    def decode(self, x):
        decoded = self.d5(x)
        decoded = self.d4(decoded)
        decoded = self.d3(decoded)
        decoded = self.d2(decoded)
        decoded = self.d1(decoded)
        return decoded

    def call(self, x):
        enc1 = self.b1(x)
        enc2 = self.b2(enc1)
        enc3 = self.b3(enc2)
        enc4 = self.b4(enc3)
        enc5 = self.b5(enc4)
        dec5 = self.d5(enc5)
        dec4 = self.d4(dec5 + enc4)
        dec3 = self.d3(dec4 + enc3)
        dec2 = self.d2(dec3 + enc2)
        dec1 = self.d1(dec2 + enc1)
        return dec1
