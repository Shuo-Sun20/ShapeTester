import numpy as np
import keras

def call_func(data_format, keepdims, inputs):
    layer = keras.layers.GlobalAveragePooling3D(data_format=data_format, keepdims=keepdims)
    if isinstance(inputs, list):
        output = layer(*inputs)
    else:
        output = layer(inputs)
    return output

x = np.random.rand(2, 4, 5, 4, 3).astype(np.float32)
example_output = call_func(data_format='channels_last', keepdims=False, inputs=x)