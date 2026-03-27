import keras
import numpy as np

def call_func(inputs, data_format=None):
    return keras.ops.image.rgb_to_hsv(images=inputs, data_format=data_format)

example_output = call_func(np.random.random((4, 4, 3)).astype(np.float32))