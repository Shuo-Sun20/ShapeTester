import keras
import numpy as np

def call_func(inputs, data_format=None):
    return keras.ops.image.hsv_to_rgb(images=inputs, data_format=data_format)

x = np.random.random((2, 4, 4, 3)).astype(np.float32)
example_output = call_func(x)