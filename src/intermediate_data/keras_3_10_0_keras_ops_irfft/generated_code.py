import keras
import numpy as np

def call_func(inputs, fft_length=None):
    real_part, imag_part = inputs[0], inputs[1]
    return keras.ops.irfft((real_part, imag_part), fft_length)

np.random.seed(42)
real_tensor = keras.ops.convert_to_tensor(np.random.randn(8).astype(np.float32))
imag_tensor = keras.ops.convert_to_tensor(np.random.randn(8).astype(np.float32))
example_output = call_func([real_tensor, imag_tensor], fft_length=16)