import keras
import numpy as np

def call_func(inputs):
    return keras.ops.angle(inputs)

# Generate random complex tensor using numpy
real_part = np.random.randn(2, 2).astype(np.float32)
imag_part = np.random.randn(2, 2).astype(np.float32)
complex_tensor = keras.ops.convert_to_tensor(real_part + 1j * imag_part)

example_output = call_func(complex_tensor)