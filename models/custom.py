# ------------------------------------------------------------ #
#
# file : models/custom.py
# author : CM
# Keras custom models
#
# ------------------------------------------------------------ #
from keras import Input, Model
from keras.layers import Conv3D, Dropout, MaxPooling3D, SpatialDropout3D, UpSampling3D, concatenate, BatchNormalization
from keras.utils import plot_model


def custom_model_1(size_x, size_y, size_z):
    # Input layer
    input = Input(shape=(size_x, size_y, size_z, 1))

    # Hyperparameter
    dropoutRD_value = 0.05
    dropout3D_value = 0.1

    # 1
    conv_1_1_1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv_1_1_1 = Dropout(dropoutRD_value)(conv_1_1_1)
    conv_1_1_2 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_1_1_1)
    conv_1_1_2 = Dropout(dropoutRD_value)(conv_1_1_2)
    pool_1_1_3 = MaxPooling3D((2, 2, 2))(conv_1_1_2)

    conv_1_2_1 = Conv3D(16, (5, 5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv_1_2_1 = Dropout(dropoutRD_value)(conv_1_2_1)
    pool_1_2_2 = MaxPooling3D((2, 2, 2))(conv_1_2_1)

    conc_1_1 = concatenate([conv_1_1_2, conv_1_2_1], axis=4)
    conc_1_2 = concatenate([pool_1_1_3, pool_1_2_2], axis=4)

    # 2
    conv_2_1_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conc_1_2)
    conv_2_1_1 = Dropout(dropoutRD_value)(conv_2_1_1)
    conv_2_1_2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_2_1_1)
    conv_2_1_2 = Dropout(dropoutRD_value)(conv_2_1_2)
    pool_2_1_3 = MaxPooling3D((2, 2, 2))(conv_2_1_2)

    conv_2_2_1 = Conv3D(32, (5, 5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(conc_1_2)
    conv_2_2_1 = Dropout(dropoutRD_value)(conv_2_2_1)
    pool_2_2_2 = MaxPooling3D((2, 2, 2))(conv_2_2_1)

    conc_2_1 = concatenate([conv_2_1_2, conv_2_2_1], axis=4)
    conc_2_2 = concatenate([pool_2_1_3, pool_2_2_2], axis=4)

    # 3
    conv_3_1_1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conc_2_2)
    conv_3_1_1 = Dropout(dropoutRD_value)(conv_3_1_1)
    conv_3_1_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_3_1_1)
    conv_3_1_2 = Dropout(dropoutRD_value)(conv_3_1_2)

    # 4
    upsa_4_1_1 = UpSampling3D(size=(2, 2, 2))(conv_3_1_2)
    conc_4_1_2 = concatenate([conc_2_1, upsa_4_1_1], axis=4)
    conv_4_1_3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conc_4_1_2)
    conv_4_1_3 = Dropout(dropoutRD_value)(conv_4_1_3)
    conv_4_1_3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_4_1_3)
    conv_4_1_3 = Dropout(dropoutRD_value)(conv_4_1_3)

    # 5
    upsa_5_1_1 = UpSampling3D(size=(2, 2, 2))(conv_4_1_3)
    conc_5_1_2 = concatenate([conc_1_1, upsa_5_1_1], axis=4)
    conv_5_1_3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conc_5_1_2)
    conv_5_1_3 = Dropout(dropoutRD_value)(conv_5_1_3)
    conv_5_1_3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_5_1_3)
    conv_5_1_3 = Dropout(dropoutRD_value)(conv_5_1_3)

    # out
    conv_out = Conv3D(2, (1, 1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_5_1_3)
    conv_out = BatchNormalization(axis=4)(conv_out)
    conv_out = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv_out)

    model = Model(inputs=input, outputs=conv_out)

    return model



def custom_model_2(size_x, size_y, size_z):
    # Input layer
    input = Input(shape=(size_x, size_y, size_z, 1))

    # Hyperparameter
    dropoutRD_value = 0.05
    dropout3D_value = 0.1

    # 1
    conv_1_1_1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv_1_1_1 = Dropout(dropoutRD_value)(conv_1_1_1)
    conv_1_1_2 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_1_1_1)
    conv_1_1_2 = Dropout(dropoutRD_value)(conv_1_1_2)
    pool_1_1_3 = MaxPooling3D((2, 2, 2))(conv_1_1_2)

    conv_1_2_1 = Conv3D(16, (5, 5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv_1_2_1 = Dropout(dropoutRD_value)(conv_1_2_1)
    pool_1_2_2 = MaxPooling3D((2, 2, 2))(conv_1_2_1)

    conv_1_3_1 = Conv3D(16, (8, 8, 8), activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv_1_3_1 = Dropout(dropoutRD_value)(conv_1_3_1)
    pool_1_3_2 = MaxPooling3D((2, 2, 2))(conv_1_3_1)

    conc_1_1 = concatenate([conv_1_1_2, conv_1_2_1, conv_1_3_1], axis=4)
    conc_1_2 = concatenate([pool_1_1_3, pool_1_2_2, pool_1_3_2], axis=4)

    # 2
    conv_2_1_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conc_1_2)
    conv_2_1_1 = Dropout(dropoutRD_value)(conv_2_1_1)
    conv_2_1_2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_2_1_1)
    conv_2_1_2 = Dropout(dropoutRD_value)(conv_2_1_2)
    pool_2_1_3 = MaxPooling3D((2, 2, 2))(conv_2_1_2)

    conv_2_2_1 = Conv3D(32, (5, 5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(conc_1_2)
    conv_2_2_1 = Dropout(dropoutRD_value)(conv_2_2_1)
    pool_2_2_2 = MaxPooling3D((2, 2, 2))(conv_2_2_1)

    conv_2_3_1 = Conv3D(32, (8, 8, 8), activation='relu', padding='same', kernel_initializer='he_normal')(conc_1_2)
    conv_2_3_1 = Dropout(dropoutRD_value)(conv_2_3_1)
    pool_2_3_2 = MaxPooling3D((2, 2, 2))(conv_2_3_1)

    conc_2_1 = concatenate([conv_2_1_2, conv_2_2_1, conv_2_3_1], axis=4)
    conc_2_2 = concatenate([pool_2_1_3, pool_2_2_2, pool_2_3_2], axis=4)

    # 3
    conv_3_1_1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conc_2_2)
    conv_3_1_1 = Dropout(dropoutRD_value)(conv_3_1_1)
    conv_3_1_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_3_1_1)
    conv_3_1_2 = Dropout(dropoutRD_value)(conv_3_1_2)

    # 4
    upsa_4_1_1 = UpSampling3D(size=(2, 2, 2))(conv_3_1_2)
    conc_4_1_2 = concatenate([conc_2_1, upsa_4_1_1], axis=4)
    conv_4_1_3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conc_4_1_2)
    conv_4_1_3 = Dropout(dropoutRD_value)(conv_4_1_3)
    conv_4_1_3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_4_1_3)
    conv_4_1_3 = Dropout(dropoutRD_value)(conv_4_1_3)

    # 5
    upsa_5_1_1 = UpSampling3D(size=(2, 2, 2))(conv_4_1_3)
    conc_5_1_2 = concatenate([conc_1_1, upsa_5_1_1], axis=4)
    conv_5_1_3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conc_5_1_2)
    conv_5_1_3 = Dropout(dropoutRD_value)(conv_5_1_3)
    conv_5_1_3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_5_1_3)
    conv_5_1_3 = Dropout(dropoutRD_value)(conv_5_1_3)

    # out
    conv_out = Conv3D(2, (1, 1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_5_1_3)
    conv_out = BatchNormalization(axis=4)(conv_out)
    conv_out = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv_out)

    model = Model(inputs=input, outputs=conv_out)

    return model


if __name__ == '__main__':
    #model_1 = custom_model_1(64, 64, 64)
    #model_1.summary()
    #print(model_1.input_shape, model_1.output_shape)
    #plot_model(model_1, to_file='model_1.png')

    model_2 = custom_model_2(64, 64, 64)
    model_2.summary()
    print(model_2.input_shape, model_2.output_shape)
    plot_model(model_2, to_file='model_2.png')