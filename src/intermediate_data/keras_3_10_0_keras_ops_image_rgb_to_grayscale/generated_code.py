import numpy as np
import keras.ops

def call_func(inputs, data_format=None):
    return keras.ops.image.rgb_to_grayscale(images=inputs, data_format=data_format)

x = np.random.random((2, 4, 4, 3))
example_output = call_func(x)