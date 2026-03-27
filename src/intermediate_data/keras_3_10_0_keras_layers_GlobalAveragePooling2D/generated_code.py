import keras
import numpy as np

def call_func(inputs, data_format=None, keepdims=False):
    layer = keras.layers.GlobalAveragePooling2D(data_format=data_format, keepdims=keepdims)
    return layer(inputs)

x = np.random.rand(2, 4, 5, 3).astype(np.float32)
example_output = call_func(x)