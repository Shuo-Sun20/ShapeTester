import keras
import numpy as np

def call_func(inputs, cropping=((1, 1), (1, 1), (1, 1)), data_format=None):
    layer = keras.layers.Cropping3D(cropping=cropping, data_format=data_format)
    return layer(inputs)

input_tensor = np.random.random((2, 28, 28, 10, 3)).astype(np.float32)
example_output = call_func(inputs=input_tensor, cropping=(2, 4, 2))