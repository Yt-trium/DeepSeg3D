# ------------------------------------------------------------ #
#
# file : fcnn.py
# author : CM
# Keras fcnn models
#
# ------------------------------------------------------------ #
from keras import Input, Model
from keras.layers import Conv3D, PReLU, Cropping3D, concatenate, Reshape, Permute, Activation

def fcnn_1(size_x, size_y, size_z, num_classes) :
    init_input = Input((size_x, size_y, size_z, 1))

    x = Conv3D(25, kernel_size=(3, 3, 3))(init_input)
    x = PReLU()(x)
    x = Conv3D(25, kernel_size=(3, 3, 3))(x)
    x = PReLU()(x)
    x = Conv3D(25, kernel_size=(3, 3, 3))(x)
    x = PReLU()(x)

    y = Conv3D(50, kernel_size=(3, 3, 3))(x)
    y = PReLU()(y)
    y = Conv3D(50, kernel_size=(3, 3, 3))(y)
    y = PReLU()(y)
    y = Conv3D(50, kernel_size=(3, 3, 3))(y)
    y = PReLU()(y)

    z = Conv3D(75, kernel_size=(3, 3, 3))(y)
    z = PReLU()(z)
    z = Conv3D(75, kernel_size=(3, 3, 3))(z)
    z = PReLU()(z)

    x_crop = Cropping3D(cropping=((5, 5), (5, 5), (5, 5)))(x)
    y_crop = Cropping3D(cropping=((2, 2), (2, 2), (2, 2)))(y)

    concat = concatenate([x_crop, y_crop, z], axis=4)

    fc = Conv3D(400, kernel_size=(1, 1, 1))(concat)
    fc = PReLU()(fc)
    fc = Conv3D(200, kernel_size=(1, 1, 1))(fc)
    fc = PReLU()(fc)
    fc = Conv3D(150, kernel_size=(1, 1, 1))(fc)
    fc = PReLU()(fc)

    pred = Conv3D(num_classes, kernel_size=(1, 1, 1))(fc)
    pred = PReLU()(pred)
    pred = Reshape((num_classes, 16*16*16))(pred)
    pred = Permute((2, 1))(pred)
    pred = Activation('softmax')(pred)

    model = Model(inputs=init_input, outputs=pred)

    return model