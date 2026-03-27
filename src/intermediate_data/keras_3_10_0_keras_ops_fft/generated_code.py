import keras
import numpy as np

def call_func(inputs):
    real_part, imag_part = inputs
    return keras.ops.fft((real_part, imag_part))

real_tensor = keras.ops.convert_to_tensor(np.random.random((2,)))
imag_tensor = keras.ops.convert_to_tensor(np.random.random((2,)))
example_output = list(call_func([real_tensor, imag_tensor]))