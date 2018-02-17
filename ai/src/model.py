# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synopsis: defines the model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization
from keras.utils import np_utils

model = Sequential([
    Conv2D(8, kernel_size=3, input_shape=(75, 75, 1), activation='relu', use_bias=False),
    BatchNormalization(),
    Conv2D(8, kernel_size=3, activation='relu', use_bias=False),
    BatchNormalization(),
    Conv2D(8, kernel_size=3, activation='relu', use_bias=False),
    BatchNormalization(),
    Conv2D(8, kernel_size=3, activation='relu', use_bias=False),
    BatchNormalization(),
    Flatten(),
    Dense(8),
    Dense(1)
])
