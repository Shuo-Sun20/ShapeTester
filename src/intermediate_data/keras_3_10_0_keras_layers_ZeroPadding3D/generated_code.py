import keras
import numpy as np

def call_func(padding=((1, 1), (1, 1), (1, 1)), data_format=None, inputs=None):
    layer = keras.layers.ZeroPadding3D(padding=padding, data_format=data_format)
    output = layer(inputs)
    return output

input_tensor = np.random.randn(2, 4, 6, 8, 3).astype(np.float32)
example_output = call_func(padding=((1, 1), (2, 2), (3, 3)), inputs=input_tensor)