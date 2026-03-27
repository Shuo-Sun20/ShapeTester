import keras
import numpy as np

def call_func(inputs, mask=None, data_format=None, keepdims=False):
    layer = keras.layers.GlobalAveragePooling1D(data_format=data_format, keepdims=keepdims)
    if mask is not None:
        return layer(inputs, mask=mask)
    else:
        return layer(inputs)

x = np.random.rand(2, 3, 4)
example_output = call_func(x)