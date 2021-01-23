from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def create_mlp(input_dim, layers = [8, 4]):

    model = Sequential()
    for i, layer in enumerate(layers):
        if i == 0:
            model.add(Dense(layer, input_dim = input_dim, activation = "relu"))
        else:
            model.add(Dense(layer, activation = "relu"))

    return model

def create_cnn(img_height, img_width, depth = 1, filters = [16, 32, 64], kernel_sizes = [4, 4, 4],
               strides = [8, 8, 8], activation = "relu", pool_size = (2, 2),
               dropout = 0.5, final_sizes = [16, 4]):

    input_shape = (img_height, img_width, depth)
    inputs = Input(shape = input_shape)
    # loop over the number of filters
    for i, (f, k, s) in enumerate(zip(filters, kernel_sizes, strides)):
        # first layer
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(filters = f, kernel_size = k, strides = s, padding = "same")(x)
        x = Activation(activation)(x)
        x = BatchNormalization(axis = -1)(x)
        x = MaxPooling2D(pool_size = pool_size)(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(final_sizes[0])(x)
    x = Activation(activation)(x)
    x = BatchNormalization(axis = -1)(x)
    x = Dropout(dropout)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(final_sizes[1])(x)
    x = Activation(activation)(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model