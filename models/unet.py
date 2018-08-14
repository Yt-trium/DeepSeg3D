# ------------------------------------------------------------ #
#
# file : models/unet.py
# author : CM
# Keras unet models
#
# ------------------------------------------------------------ #
import tensorflow as tf

from keras import Input, Model
from keras.layers import Conv3D, MaxPooling3D, Dropout, UpSampling3D, concatenate, BatchNormalization, Cropping3D, \
    regularizers, SpatialDropout3D


class Unet_1:
    def __init__(self):
        self.a = 1
        # nothing

    def __call__(self, X:tf.Tensor):
        with tf.name_scope("Block-1"):
            conv_01 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(X)
            conv_01 = Dropout(0.2)(conv_01)
            conv_01 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_01)
            pool_01 = MaxPooling3D((2, 2, 2))(conv_01)

        with tf.name_scope("Block-2"):
            conv_02 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_01)
            conv_02 = Dropout(0.2)(conv_02)
            conv_02 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_02)
            pool_02 = MaxPooling3D((2, 2, 2))(conv_02)

        with tf.name_scope("Block-3"):
            conv_03 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_02)
            conv_03 = Dropout(0.2)(conv_03)
            conv_03 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_03)
            pool_03 = MaxPooling3D((2, 2, 2))(conv_03)

        with tf.name_scope("Block-4"):
            conv_04 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_03)
            conv_04 = Dropout(0.2)(conv_04)
            conv_04 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_04)

        with tf.name_scope("Block-5"):
            up_11 = UpSampling3D(size=(2, 2, 2))(conv_04)
            up_11 = concatenate([conv_03, up_11], axis=4)
            conv_11 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_11)
            conv_11 = Dropout(0.2)(conv_11)
            conv_11 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_11)

        with tf.name_scope("Block-6"):
            up_12 = UpSampling3D(size=(2, 2, 2))(conv_11)
            up_12 = concatenate([conv_02, up_12], axis=4)
            conv_12 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_12)
            conv_12 = Dropout(0.2)(conv_12)
            conv_12 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_12)

        with tf.name_scope("Block-7"):
            up_13 = UpSampling3D(size=(2, 2, 2))(conv_12)
            up_13 = concatenate([conv_01, up_13], axis=4)
            conv_13 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_13)
            conv_13 = Dropout(0.2)(conv_13)
            conv_13 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_13)

        with tf.name_scope("Block-8"):
            conv_14 = Conv3D(2, (1, 1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_13)
            conv_14 = BatchNormalization(axis=4)(conv_14)
            conv_15 = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv_14)

        tf.summary.histogram("conv_01", conv_01)
        tf.summary.histogram("pool_01", pool_01)
        tf.summary.histogram("conv_02", conv_02)
        tf.summary.histogram("pool_02", pool_02)
        tf.summary.histogram("conv_03", conv_03)
        tf.summary.histogram("pool_03", pool_03)
        tf.summary.histogram("conv_04", conv_04)
        tf.summary.histogram("up_11", up_11)
        tf.summary.histogram("conv_11", conv_11)
        tf.summary.histogram("up_12", up_12)
        tf.summary.histogram("conv_12", conv_12)
        tf.summary.histogram("up_13", up_13)
        tf.summary.histogram("conv_13", conv_13)
        tf.summary.histogram("conv_14", conv_14)
        tf.summary.histogram("conv_15", conv_15)

        return conv_15

# unet model
def unet_1(size_x, size_y, size_z):
    # Input layer
    input = Input(shape=(size_x, size_y, size_z, 1))

    # convolutional layer 1
    conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(32, 32, 32, 1), name="conv_1_c1")(input)
    conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_1_c2")(conv_1)
    pool_1 = MaxPooling3D((2, 2, 2), name="pool_1_p1")(conv_1)

    # convolutional layer 2
    conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2_c1")(pool_1)
    conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2_c2")(conv_2)
    pool_2 = MaxPooling3D((2, 2, 2), name="pool_2_p1")(conv_2)

    # convolutional layer 3
    conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_3_c1")(pool_2)
    conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_3_c2")(conv_3)
    pool_3 = MaxPooling3D((2, 2, 2), name="pool_3_p1")(conv_3)

    # convolutional layer 4
    conv_4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_4_c1")(pool_3)
    conv_4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_4_c2")(conv_4)
    conv_4 = Dropout(0.4, name="conv_4_d1")(conv_4)
    pool_4 = MaxPooling3D((2, 2, 2), name="pool_4_p1")(conv_4)

    # convolutional layer 5
    conv_5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_5_c1")(pool_4)
    conv_5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_5_c2")(conv_5)
    conv_5 = Dropout(0.4, name="conv_5_d1")(conv_5)

    # Up sampling 5
    up_5 = UpSampling3D((2, 2, 2), name="up_5_u1")(conv_5)
    up_5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="up_5_c1")(up_5)
    # Merge layer 4 and up 5
    merge_4 = concatenate([conv_4, up_5], axis=4, name="merge_4")

    # convolutional layer 6
    conv_6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_6_c1")(merge_4)
    conv_6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_6_c2")(conv_6)

    # Up sampling 4
    up_4 = UpSampling3D((2, 2, 2), name="up_4_u1")(conv_6)
    up_4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="up_4_c1")(up_4)
    # Merge layer 3 and up 4
    merge_3 = concatenate([conv_3, up_4], axis=4, name="merge_3")

    # convolutional layer 7
    conv_7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_7_c1")(merge_3)
    conv_7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_7_c2")(conv_7)

    # Up sampling 3
    up_3 = UpSampling3D((2, 2, 2), name="up_3_u1")(conv_7)
    up_3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="up_3_c1")(up_3)
    # Merge layer 2 and up 3
    merge_2 = concatenate([conv_2, up_3], axis=4, name="merge_2")

    # convolutional layer 8
    conv_8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_8_c1")(merge_2)
    conv_8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_8_c2")(conv_8)

    # Up sampling 2
    up_2 = UpSampling3D((2, 2, 2), name="up_2_u1")(conv_8)
    up_2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="up_2_c1")(up_2)
    # Merge layer 1 and up 2
    merge_1 = concatenate([conv_1, up_2], axis=4, name="merge_1")

    # convolutional layer 9
    conv_9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_9_c1")(merge_1)
    conv_9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_9_c2")(conv_9)
    conv_9 = Conv3D(2, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name="conv_9_c3")(conv_9)

    # convolutional layer 10
    conv_10 = Conv3D(1, 1, activation = 'sigmoid', name="conv_10_c1")(conv_9)

    model = Model(inputs=input, outputs=conv_10)

    return model

# unet model
def unet_2(size_x, size_y, size_z):
    # Input layer
    input = Input(shape=(size_x, size_y, size_z, 1))

    #
    conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                    input_shape=(size_x, size_y, size_z, 1))(input)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_1)
    pool_1 = MaxPooling3D((2, 2, 2))(conv_1)

    #
    conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_1)
    conv_2 = Dropout(0.2)(conv_2)
    conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_2)
    pool_2 = MaxPooling3D((2, 2, 2))(conv_2)

    #
    conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_2)
    conv_3 = Dropout(0.2)(conv_3)
    conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_3)

    #
    up_1 = UpSampling3D(size=(2, 2, 2))(conv_3)
    up_1 = concatenate([conv_2, up_1], axis=4)
    conv_4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_1)
    conv_4 = Dropout(0.2)(conv_4)
    conv_4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_4)

    #
    up_2 = UpSampling3D(size=(2, 2, 2))(conv_4)
    up_2 = concatenate([conv_1, up_2], axis=4)
    conv_5 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_2)
    conv_5 = Dropout(0.2)(conv_5)
    conv_5 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_5)

    #
    conv_6 = Conv3D(2, (1, 1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_5)
    conv_6 = BatchNormalization(axis=4)(conv_6)
    conv_7 = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv_6)

    model = Model(inputs=input, outputs=conv_7)

    return model

# unet model
def unet_3(size_x, size_y, size_z):
    # Input layer
    input = Input(shape=(size_x, size_y, size_z, 1))

    #
    conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                    input_shape=(size_x, size_y, size_z, 1))(input)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_1)
    pool_1 = MaxPooling3D((2, 2, 2))(conv_1)

    #
    conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_1)
    conv_2 = Dropout(0.2)(conv_2)
    conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_2)
    pool_2 = MaxPooling3D((2, 2, 2))(conv_2)

    #
    conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_2)
    conv_3 = Dropout(0.2)(conv_3)
    conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_3)
    pool_3 = MaxPooling3D((2, 2, 2))(conv_3)

    #
    conv_4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_3)
    conv_4 = Dropout(0.2)(conv_4)
    conv_4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_4)
    pool_4 = MaxPooling3D((2, 2, 2))(conv_4)

    #
    conv_5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_4)
    conv_5 = Dropout(0.2)(conv_5)
    conv_5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_5)

    #
    up_1 = UpSampling3D(size=(2, 2, 2))(conv_5)
    up_1 = concatenate([conv_4, up_1], axis=4)
    conv_6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_1)
    conv_6 = Dropout(0.2)(conv_6)
    conv_6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_6)

    #
    up_2 = UpSampling3D(size=(2, 2, 2))(conv_6)
    up_2 = concatenate([conv_3, up_2], axis=4)
    conv_7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_2)
    conv_7 = Dropout(0.2)(conv_7)
    conv_7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_7)

    #
    up_3 = UpSampling3D(size=(2, 2, 2))(conv_7)
    up_3 = concatenate([conv_2, up_3], axis=4)
    conv_8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_3)
    conv_8 = Dropout(0.2)(conv_8)
    conv_8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_8)

    #
    up_4 = UpSampling3D(size=(2, 2, 2))(conv_8)
    up_4 = concatenate([conv_1, up_4], axis=4)
    conv_9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_4)
    conv_9 = Dropout(0.2)(conv_9)
    conv_9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_9)

    #
    conv_10 = Conv3D(2, (1, 1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_9)
    conv_10 = BatchNormalization(axis=4)(conv_10)
    conv_11 = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv_10)

    model = Model(inputs=input, outputs=conv_11)

    return model

# unet model
def unet_3_cropping(size_x, size_y, size_z):
    # Input layer
    input = Input(shape=(size_x, size_y, size_z, 1))

    #
    conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                    input_shape=(size_x, size_y, size_z, 1))(input)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_1)
    pool_1 = MaxPooling3D((2, 2, 2))(conv_1)

    #
    conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_1)
    conv_2 = Dropout(0.2)(conv_2)
    conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_2)
    pool_2 = MaxPooling3D((2, 2, 2))(conv_2)

    #
    conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_2)
    conv_3 = Dropout(0.2)(conv_3)
    conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_3)
    pool_3 = MaxPooling3D((2, 2, 2))(conv_3)

    #
    conv_4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_3)
    conv_4 = Dropout(0.2)(conv_4)
    conv_4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_4)
    pool_4 = MaxPooling3D((2, 2, 2))(conv_4)

    #
    conv_5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_4)
    conv_5 = Dropout(0.2)(conv_5)
    conv_5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_5)

    #
    up_1 = UpSampling3D(size=(2, 2, 2))(conv_5)
    up_1 = concatenate([conv_4, up_1], axis=4)
    conv_6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_1)
    conv_6 = Dropout(0.2)(conv_6)
    conv_6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_6)

    #
    up_2 = UpSampling3D(size=(2, 2, 2))(conv_6)
    up_2 = concatenate([conv_3, up_2], axis=4)
    conv_7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_2)
    conv_7 = Dropout(0.2)(conv_7)
    conv_7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_7)

    #
    up_3 = UpSampling3D(size=(2, 2, 2))(conv_7)
    up_3 = concatenate([conv_2, up_3], axis=4)
    conv_8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_3)
    conv_8 = Dropout(0.2)(conv_8)
    conv_8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_8)

    #
    up_4 = UpSampling3D(size=(2, 2, 2))(conv_8)
    up_4 = concatenate([conv_1, up_4], axis=4)
    conv_9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_4)
    conv_9 = Dropout(0.2)(conv_9)
    conv_9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_9)

    #
    conv_10 = Conv3D(2, (1, 1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_9)
    conv_10 = BatchNormalization(axis=4)(conv_10)
    conv_11 = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv_10)

    xc = int(size_x/4)
    yc = int(size_y/4)
    zc = int(size_z/4)
    conv_11 = Cropping3D(cropping=((xc, xc), (yc, yc), (zc, zc)))(conv_11)

    model = Model(inputs=input, outputs=conv_11)

    return model

# unet model
def unet_3_light(size_x, size_y, size_z):
    # Input layer
    input = Input(shape=(size_x, size_y, size_z, 1))

    #
    conv_1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                    input_shape=(size_x, size_y, size_z, 1))(input)
    conv_1 = SpatialDropout3D(0.2)(conv_1)
    conv_1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_1)
    pool_1 = MaxPooling3D((2, 2, 2))(conv_1)

    #
    conv_2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_1)
    conv_2 = SpatialDropout3D(0.2)(conv_2)
    conv_2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_2)
    pool_2 = MaxPooling3D((2, 2, 2))(conv_2)

    #
    conv_3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_2)
    conv_3 = SpatialDropout3D(0.2)(conv_3)
    conv_3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_3)
    pool_3 = MaxPooling3D((2, 2, 2))(conv_3)

    #
    conv_4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_3)
    conv_4 = SpatialDropout3D(0.2)(conv_4)
    conv_4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_4)

    #
    up_1 = UpSampling3D(size=(2, 2, 2))(conv_4)
    up_1 = concatenate([conv_3, up_1], axis=4)
    conv_6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_1)
    conv_6 = SpatialDropout3D(0.2)(conv_6)
    conv_6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_6)

    #
    up_2 = UpSampling3D(size=(2, 2, 2))(conv_6)
    up_2 = concatenate([conv_2, up_2], axis=4)
    conv_7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_2)
    conv_7 = SpatialDropout3D(0.2)(conv_7)
    conv_7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_7)

    #
    up_3 = UpSampling3D(size=(2, 2, 2))(conv_7)
    up_3 = concatenate([conv_1, up_3], axis=4)
    conv_8 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_3)
    conv_8 = SpatialDropout3D(0.2)(conv_8)
    conv_8 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_8)

    #
    conv_10 = Conv3D(2, (1, 1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_8)
    conv_10 = BatchNormalization(axis=4)(conv_10)
    conv_11 = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv_10)

    model = Model(inputs=input, outputs=conv_11)

    return model

# Cutted unet (output shape = input / 2)
def cunet_1(size_x, size_y, size_z):
    #
    input = Input(shape=(size_x, size_y, size_z, 1))

    #
    conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                    input_shape=(size_x, size_y, size_z, 1))(input)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_1)
    pool_1 = MaxPooling3D((2, 2, 2))(conv_1)

    #
    conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_1)
    conv_2 = Dropout(0.2)(conv_2)
    conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_2)
    pool_2 = MaxPooling3D((2, 2, 2))(conv_2)

    #
    conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_2)
    conv_3 = Dropout(0.2)(conv_3)
    conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_3)
    pool_3 = MaxPooling3D((2, 2, 2))(conv_3)

    #
    conv_4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_3)
    conv_4 = Dropout(0.2)(conv_4)
    conv_4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_4)

    #
    up_1 = UpSampling3D(size=(2, 2, 2))(conv_4)
    up_1 = concatenate([pool_2, up_1], axis=4)
    conv_5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_1)
    conv_5 = Dropout(0.2)(conv_5)
    conv_5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_5)

    #
    up_2 = UpSampling3D(size=(2, 2, 2))(conv_5)
    up_2 = concatenate([pool_1, up_2], axis=4)
    conv_6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_2)
    conv_6 = Dropout(0.2)(conv_6)
    conv_6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_6)

    #
    conv_7 = Conv3D(2, (1, 1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_6)
    conv_7 = BatchNormalization(axis=4)(conv_7)
    conv_8 = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv_7)

    model = Model(inputs=input, outputs=conv_8)

    return model

def unet_exp_1(size_x, size_y, size_z):
    filters_mult = 1
    kernel_size = 3
    dropout = 0.2
    dropout_3d = 0

    input = Input(shape=(size_x, size_y, size_z, 1))

    #
    conv_1 = Conv3D(int(16*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv_1 = Dropout(dropout)(conv_1)
    conv_1 = SpatialDropout3D(dropout_3d)(conv_1)
    conv_1 = Conv3D(int(16*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_1)
    pool_1 = MaxPooling3D(2)(conv_1)

    #
    conv_2 = Conv3D(int(32*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(pool_1)
    conv_2 = Dropout(dropout)(conv_2)
    conv_2 = SpatialDropout3D(dropout_3d)(conv_2)
    conv_2 = Conv3D(int(32*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_2)
    pool_2 = MaxPooling3D(2)(conv_2)

    #
    conv_3 = Conv3D(int(64*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(pool_2)
    conv_3 = Dropout(dropout)(conv_3)
    conv_3 = SpatialDropout3D(dropout_3d)(conv_3)
    conv_3 = Conv3D(int(64*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_3)
    pool_3 = MaxPooling3D(2)(conv_3)

    #
    conv_4 = Conv3D(int(128*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(pool_3)
    conv_4 = Dropout(dropout)(conv_4)
    conv_4 = SpatialDropout3D(dropout_3d)(conv_4)
    conv_4 = Conv3D(int(128*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_4)

    #
    up_1 = UpSampling3D(size=2)(conv_4)
    up_1 = concatenate([conv_3, up_1], axis=4)
    conv_6 = Conv3D(int(64*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(up_1)
    conv_6 = Dropout(dropout)(conv_6)
    conv_6 = SpatialDropout3D(dropout_3d)(conv_6)
    conv_6 = Conv3D(int(64*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_6)

    #
    up_2 = UpSampling3D(size=2)(conv_6)
    up_2 = concatenate([conv_2, up_2], axis=4)
    conv_7 = Conv3D(int(32*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(up_2)
    conv_7 = Dropout(dropout)(conv_7)
    conv_7 = SpatialDropout3D(dropout_3d)(conv_7)
    conv_7 = Conv3D(int(32*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_7)

    #
    up_3 = UpSampling3D(size=2)(conv_7)
    up_3 = concatenate([conv_1, up_3], axis=4)
    conv_8 = Conv3D(int(16*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(up_3)
    conv_8 = Dropout(dropout)(conv_8)
    conv_8 = SpatialDropout3D(dropout_3d)(conv_8)
    conv_8 = Conv3D(int(16*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_8)

    #
    conv_10 = Conv3D(2, (1, 1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_8)
    conv_10 = BatchNormalization(axis=4)(conv_10)
    conv_11 = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv_10)

    model = Model(inputs=input, outputs=conv_11)

    return model

def unet_exp_2(size_x, size_y, size_z):
    filters_mult = 1
    kernel_size = 3
    dropout = 0.2
    dropout_3d = 0

    input = Input(shape=(size_x, size_y, size_z, 1))

    #
    conv_1 = Conv3D(int(8*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv_1 = Dropout(dropout)(conv_1)
    conv_1 = SpatialDropout3D(dropout_3d)(conv_1)
    conv_1 = Conv3D(int(8*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_1)
    pool_1 = MaxPooling3D(2)(conv_1)

    #
    conv_2 = Conv3D(int(16*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(pool_1)
    conv_2 = Dropout(dropout)(conv_2)
    conv_2 = SpatialDropout3D(dropout_3d)(conv_2)
    conv_2 = Conv3D(int(16*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_2)
    pool_2 = MaxPooling3D(2)(conv_2)

    #
    conv_3 = Conv3D(int(32*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(pool_2)
    conv_3 = Dropout(dropout)(conv_3)
    conv_3 = SpatialDropout3D(dropout_3d)(conv_3)
    conv_3 = Conv3D(int(32*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_3)

    #
    up_1 = UpSampling3D(size=2)(conv_3)
    up_1 = concatenate([conv_2, up_1], axis=4)
    conv_4 = Conv3D(int(16*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(up_1)
    conv_4 = Dropout(dropout)(conv_4)
    conv_4 = SpatialDropout3D(dropout_3d)(conv_4)
    conv_4 = Conv3D(int(16*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_4)

    #
    up_2 = UpSampling3D(size=2)(conv_4)
    up_2 = concatenate([conv_1, up_2], axis=4)
    conv_5 = Conv3D(int(8*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(up_2)
    conv_5 = Dropout(dropout)(conv_5)
    conv_5 = SpatialDropout3D(dropout_3d)(conv_5)
    conv_5 = Conv3D(int(8*filters_mult), kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv_5)

    #
    conv_6 = Conv3D(2, (1, 1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_5)
    conv_6 = BatchNormalization(axis=4)(conv_6)
    conv_7 = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv_6)

    model = Model(inputs=input, outputs=conv_7)

    return model
