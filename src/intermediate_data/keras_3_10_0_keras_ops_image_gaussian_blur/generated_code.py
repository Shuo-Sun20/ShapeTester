import keras
import numpy as np

def call_func(inputs, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None):
    return keras.ops.image.gaussian_blur(inputs, kernel_size, sigma, data_format)

example_output = call_func(np.random.random((2, 64, 80, 3)).astype(np.float32))