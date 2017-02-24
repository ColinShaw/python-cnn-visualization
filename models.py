from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Lambda

def behavior_cloning():
    return Sequential([
        Lambda(lambda x: x / 255.0 - 0.5, input_shape=(120,240,3)),
        Convolution2D(3, 1, 1),
        Convolution2D(32, 3, 3, activation='elu', init='glorot_normal'),
        Convolution2D(32, 3, 3, activation='elu', init='glorot_normal'),
        MaxPooling2D((2,2), strides=(2,2)),
        Dropout(0.5),
        Convolution2D(64, 3, 3, activation='elu', init='glorot_normal'),
        Convolution2D(64, 3, 3, activation='elu', init='glorot_normal'),
        MaxPooling2D((2,2), strides=(2,2)),
        Dropout(0.5),
        Convolution2D(128, 3, 3, activation='elu', init='glorot_normal'),
        Convolution2D(128, 3, 3, activation='elu', init='glorot_normal'),
        MaxPooling2D((2,2), strides=(2,2)),
        Dropout(0.5),
        Flatten(),
        Dense(2048, activation='elu', init='he_normal'),
        Dropout(0.5),
        Dense(256, activation='elu', init='he_normal'),
        Dropout(0.25),
        Dense(32, activation='elu', init='he_normal'),
        Dense(1, activation='tanh') 
    ])
